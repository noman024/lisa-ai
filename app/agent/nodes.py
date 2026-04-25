"""LangGraph node implementations: router, retriever, prompt builder, LLM, validator."""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from openai import OpenAIError

from app.agent.context import AgentContext
from app.agent.state import FALLBACK_MESSAGE, GraphState
from app.llm.client import chat_text
from app.rag.retriever import RetrievedChunk
from app.utils.grounding import grounding_score

_log = logging.getLogger(__name__)

# Strips any residual section-label prefixes (e.g. "Definition:") before grounding scoring.
_SECTION_LABEL = re.compile(
    r"^(Definition|Benefits|Eligibility|Claims)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)


def _classify_query_type(message: str) -> str:
    """Classify the question with regex — no extra LLM call."""
    t = message.lower()
    if re.search(
        r"\b(claim|claims|filing|file a claim|payout|death certificate|notify)\b", t
    ):
        return "claims"
    if re.search(
        r"\b(elig|qualif|underwrit|age|health exam|medical|tobacco|insurable)\b", t
    ):
        return "eligibility"
    if re.search(r"\b(compare|vs\.?|versus|difference between|or whole|or term)\b", t):
        return "comparison"
    return "informational"


def _ser_chunks(chunks: Sequence[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "text": c.text,
            "source_section": c.source_section,
            "score": c.score,
        }
        for c in chunks
    ]


def _strip_for_grounding(answer: str) -> str:
    """Normalise answer text before word-overlap scoring."""
    text = _SECTION_LABEL.sub(" ", answer)
    return re.sub(r"\s+", " ", text).strip()


def _trim_context(context: str, max_chars: int = 2000) -> str:
    """Fit retrieved text within a character budget, keeping whole paragraphs."""
    if len(context) <= max_chars:
        return context
    sections = context.split("\n\n")
    kept: list[str] = []
    used = 0
    for s in sections:
        if used + len(s) + 2 > max_chars:
            break
        kept.append(s)
        used += len(s) + 2
    return "\n\n".join(kept) if kept else context[:max_chars]


def _collapse_repeated_fallback(text: str) -> str:
    """Collapse a response that is mostly repeated 'I don't know.' lines to one."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    fallback_count = sum(1 for ln in lines if ln.rstrip(".") == FALLBACK_MESSAGE.rstrip("."))
    if lines and fallback_count / max(len(lines), 1) > 0.5:
        return FALLBACK_MESSAGE
    return text


def _sources_from_retrieved(data: list[dict[str, Any]], max_n: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for d in data:
        s = d.get("source_section") or "knowledge"
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def _build_system_prompt() -> str:
    return (
        "You are LISA, a knowledgeable life insurance support assistant. "
        "Answer the user's question using ONLY the facts found in the provided Context section. "
        "Be clear, accurate, and helpful. "
        f'If the context does not contain enough information to answer, respond with exactly: "{FALLBACK_MESSAGE}" '
        "Do not invent policy terms, rates, or guarantees."
    )


class NodeBundle:
    """Node functions bound to :class:`AgentContext`, compiled into the LangGraph."""

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    def router_node(self, state: GraphState) -> dict[str, Any]:
        q = state.get("user_message", "")
        return {"query_type": _classify_query_type(q)}

    async def retriever_node(self, state: GraphState) -> dict[str, Any]:
        s = self._ctx.settings
        r = self._ctx.retriever
        msg = state.get("user_message", "")
        chunks, best = r.search(msg, k=s.retriever_top_k)
        low = (not chunks) or (best < s.retrieval_min_score) or (not r.is_ready)
        return {
            "retrieved": _ser_chunks(chunks),
            "best_score": best,
            "context_text": r.format_context(chunks) if chunks else "",
            "low_confidence": low,
        }

    def prompt_builder_node(self, state: GraphState) -> dict[str, Any]:
        user_q = state.get("user_message", "")
        low = state.get("low_confidence", False)
        retrieved = state.get("retrieved", [])

        if low or not retrieved:
            context_block = "No relevant information found in the knowledge base."
        else:
            parts = [r.get("text", "") for r in retrieved[:3] if r.get("text")]
            context_block = _trim_context("\n\n".join(parts))

        # "Answer:" suffix elicits a direct response without preamble.
        user_payload = (
            f"Question: {user_q}\n\n"
            f"Context:\n{context_block}\n\n"
            f"Answer:"
        )
        return {
            "system_prompt": _build_system_prompt(),
            "user_payload": user_payload,
        }

    async def llm_node(self, state: GraphState) -> dict[str, Any]:
        if not self._ctx.retriever.is_ready or state.get("low_confidence", False):
            return {"raw_llm_response": FALLBACK_MESSAGE}
        try:
            out = await chat_text(
                self._ctx.llm,
                self._ctx.settings,
                system=state.get("system_prompt", _build_system_prompt()),
                user=state.get("user_payload", ""),
            )
        except OpenAIError as exc:
            _log.warning("LLM request failed: %s", exc)
            return {"raw_llm_response": FALLBACK_MESSAGE}
        return {"raw_llm_response": out or FALLBACK_MESSAGE}

    def validator_node(self, state: GraphState) -> dict[str, Any]:
        s = self._ctx.settings
        low = bool(state.get("low_confidence", False)) or (not self._ctx.retriever.is_ready)
        raw = (state.get("raw_llm_response") or "").strip() or FALLBACK_MESSAGE
        ctx = state.get("context_text", "")
        src = _sources_from_retrieved(state.get("retrieved", []))

        if low or not ctx.strip():
            return {"final_response": FALLBACK_MESSAGE, "grounded_ok": False, "sources": []}

        clean = _collapse_repeated_fallback(raw)

        if clean == FALLBACK_MESSAGE:
            return {"final_response": FALLBACK_MESSAGE, "grounded_ok": True, "sources": src}

        g = grounding_score(_strip_for_grounding(clean), ctx)
        if g >= s.grounding_min_overlap or FALLBACK_MESSAGE in clean:
            return {"final_response": clean, "grounded_ok": True, "sources": src}

        return {"final_response": FALLBACK_MESSAGE, "grounded_ok": False, "sources": []}


__all__ = ["NodeBundle", "_classify_query_type"]
