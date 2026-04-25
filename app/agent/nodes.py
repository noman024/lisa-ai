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

# Stricter similarity bar when the user message looks underspecified (vague / very short).
_VAGUE_RETRIEVAL_SCORE_MULT = 1.15

_INSURANCE_HINT = re.compile(
    r"\b("
    r"term|whole|universal|life|insur|policy|policies|claim|claims|premium|premiums|"
    r"benefici|coverage|death|underwrit|underwriting|rider|riders|cash\s+value|"
    r"annuity|lapse|contestab|face\s+amount|death\s+benefit"
    r")\b",
    re.IGNORECASE,
)

_OFF_TOPIC = re.compile(
    r"\b("
    r"weather|forecast|rain|snow|temperature|humidity|hurricane|tornado|"
    r"nfl|nba|mlb|fifa|super\s+bowl|world\s+cup|olympics|"
    r"recipe|ingredients|\bcook\b|\bbake\b|"
    r"javascript|typescript|react\.js|node\.js|debug\s+my\s+code|stack\s+overflow|"
    r"bitcoin|ethereum|cryptocurrency|\bcrypto\b"
    r")\b",
    re.IGNORECASE,
)


def _is_off_topic(message: str) -> bool:
    """Heuristic non-insurance topics — skipped when an insurance term is present."""
    if _INSURANCE_HINT.search(message):
        return False
    return bool(_OFF_TOPIC.search(message))


def _is_vague_query(message: str) -> bool:
    """
    Underspecified questions: very short, or few tokens without insurance vocabulary.

    Follow-ups like \"Does it build cash value?\" are not vague (enough tokens; may use history).
    """
    s = message.strip()
    if not s:
        return True
    if _INSURANCE_HINT.search(s):
        return False
    if len(s) < 12:
        return True
    if len(s.split()) <= 2:
        return True
    return False


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


def _should_collapse_to_fallback(clean: str) -> bool:
    """
    True when the model started with \"I don't know\" but added a non-answer explanation.

    The substring check ``FALLBACK_MESSAGE in clean`` incorrectly accepted those as
    grounded and kept irrelevant KB sources; we collapse to the canonical fallback.
    """
    t = clean.strip()
    if not t or t == FALLBACK_MESSAGE:
        return False
    for pattern in (r"^I don't know[.!]?(\s+|$)", r"^I do not know[.!]?(\s+|$)"):
        m = re.match(pattern, t, re.IGNORECASE)
        if not m:
            continue
        rest = t[m.end() :].strip()
        if not rest:
            return False
        if not _INSURANCE_HINT.search(rest):
            return True
        return False
    return False


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
        "If a \"Recent conversation\" block is present, use it only to understand follow-up questions "
        "(e.g. what \"it\" refers to). Do not treat the conversation as a source of factual claims. "
        "Be clear, accurate, and helpful. "
        f'If the context does not contain enough information to answer, respond with exactly: "{FALLBACK_MESSAGE}" '
        "Do not invent policy terms, rates, or guarantees."
    )


def format_conversation_for_prompt(
    history: list[dict] | None,
    max_messages: int,
    max_chars: int,
) -> str:
    """
    Turn prior user/assistant turns into a compact block for the RAG user prompt.

    The current user message is not included here: it is passed separately as ``Question``.
    """
    if not history or max_messages <= 0:
        return ""
    lines: list[str] = []
    for m in history[-max_messages:]:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    if not lines:
        return ""
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = "…\n" + text[-max_chars:].lstrip()
    return text


class NodeBundle:
    """Node functions bound to :class:`AgentContext`, compiled into the LangGraph."""

    def __init__(self, ctx: AgentContext) -> None:
        self._ctx = ctx

    def router_node(self, state: GraphState) -> dict[str, Any]:
        q = state.get("user_message", "")
        if _is_off_topic(q):
            return {"query_type": "off_topic", "vague_query": False}
        return {
            "query_type": _classify_query_type(q),
            "vague_query": _is_vague_query(q),
        }

    async def retriever_node(self, state: GraphState) -> dict[str, Any]:
        s = self._ctx.settings
        r = self._ctx.retriever
        msg = state.get("user_message", "")
        if state.get("query_type") == "off_topic":
            return {
                "retrieved": [],
                "best_score": 0.0,
                "context_text": "",
                "low_confidence": True,
            }
        vague = bool(state.get("vague_query", False))
        history = state.get("history") or []
        # Standalone vague text (e.g. "ok", "thanks") often gets spuriously high similarity;
        # skip retrieval and avoid an LLM call unless prior turns disambiguate.
        if vague and not history:
            return {
                "retrieved": [],
                "best_score": 0.0,
                "context_text": "",
                "low_confidence": True,
            }
        min_score = s.retrieval_min_score * (
            _VAGUE_RETRIEVAL_SCORE_MULT if vague else 1.0
        )
        chunks, best = r.search(msg, k=s.retriever_top_k)
        low = (not chunks) or (best < min_score) or (not r.is_ready)
        return {
            "retrieved": _ser_chunks(chunks),
            "best_score": best,
            "context_text": r.format_context(chunks) if chunks else "",
            "low_confidence": low,
        }

    def prompt_builder_node(self, state: GraphState) -> dict[str, Any]:
        s = self._ctx.settings
        user_q = state.get("user_message", "")
        low = state.get("low_confidence", False)
        retrieved = state.get("retrieved", [])

        if low or not retrieved:
            context_block = "No relevant information found in the knowledge base."
        else:
            parts = [r.get("text", "") for r in retrieved[:3] if r.get("text")]
            context_block = _trim_context("\n\n".join(parts))

        history_block = format_conversation_for_prompt(
            state.get("history"),
            max_messages=s.memory_max_messages,
            max_chars=s.memory_prompt_max_chars,
        )
        if history_block:
            user_payload = (
                f"Recent conversation:\n{history_block}\n\n"
                f"Question: {user_q}\n\n"
                f"Context:\n{context_block}\n\n"
                f"Answer:"
            )
        else:
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
            _log.warning("LLM request failed (OpenAIError): %s", exc)
            return {
                "raw_llm_response": FALLBACK_MESSAGE,
                "llm_call_failed": True,
            }
        except Exception as exc:
            _log.warning("LLM request failed: %s", exc, exc_info=True)
            return {
                "raw_llm_response": FALLBACK_MESSAGE,
                "llm_call_failed": True,
            }
        return {"raw_llm_response": out or FALLBACK_MESSAGE}

    def validator_node(self, state: GraphState) -> dict[str, Any]:
        s = self._ctx.settings
        low = bool(state.get("low_confidence", False)) or (not self._ctx.retriever.is_ready)
        raw = (state.get("raw_llm_response") or "").strip() or FALLBACK_MESSAGE
        ctx = state.get("context_text", "")
        src = _sources_from_retrieved(state.get("retrieved", []))

        if state.get("llm_call_failed"):
            return {
                "final_response": FALLBACK_MESSAGE,
                "grounded_ok": False,
                "sources": [],
                "low_confidence": True,
            }

        if low or not ctx.strip():
            return {
                "final_response": FALLBACK_MESSAGE,
                "grounded_ok": False,
                "sources": [],
                "low_confidence": True,
            }

        clean = _collapse_repeated_fallback(raw)

        if _should_collapse_to_fallback(clean):
            return {
                "final_response": FALLBACK_MESSAGE,
                "grounded_ok": False,
                "sources": [],
                "low_confidence": True,
            }

        if clean == FALLBACK_MESSAGE:
            return {
                "final_response": FALLBACK_MESSAGE,
                "grounded_ok": True,
                "sources": [],
                "low_confidence": True,
            }

        g = grounding_score(_strip_for_grounding(clean), ctx)
        if g >= s.grounding_min_overlap:
            return {
                "final_response": clean,
                "grounded_ok": True,
                "sources": src,
                "low_confidence": False,
            }

        return {
            "final_response": FALLBACK_MESSAGE,
            "grounded_ok": False,
            "sources": [],
            "low_confidence": True,
        }


__all__ = [
    "NodeBundle",
    "_classify_query_type",
    "_is_off_topic",
    "_is_vague_query",
    "_should_collapse_to_fallback",
    "format_conversation_for_prompt",
]
