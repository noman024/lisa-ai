"""Token-aware text chunking with overlap for RAG."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import tiktoken


# Approximate 300–500 token window (target mid-range) with overlap
_DEFAULT_ENCODING = "cl100k_base"
_TARGET_TOKENS = 400
_OVERLAP_TOKENS = 60


@dataclass
class TextChunk:
    """A slice of source text with optional section hint."""

    text: str
    source_section: str


def _split_by_headers(markdown: str) -> list[tuple[str, str]]:
    """
    Split markdown into (section_title, body) by ``##`` headers.

    With a capturing group, ``re.split`` yields: preamble, *then*
    [title, body, title, body, ...] for each ``##`` line.
    """
    pattern = re.compile(r"^##\s+(.+)$", re.MULTILINE)
    parts = pattern.split(markdown)
    if len(parts) < 2:
        return [("Overview", markdown.strip())] if markdown.strip() else []
    out: list[tuple[str, str]] = []
    # parts[0] = text before the first `##` (e.g. top-level H1). Remainder alternates.
    rest = parts[1:]
    for i in range(0, len(rest) - 1, 2):
        title = rest[i].strip()
        body = rest[i + 1].strip() if i + 1 < len(rest) else ""
        if title and body:
            out.append((title, body))
    if not out and markdown.strip():
        return [("Overview", markdown.strip())]
    return out


def chunk_markdown(
    markdown: str,
    target_tokens: int = _TARGET_TOKENS,
    overlap_tokens: int = _OVERLAP_TOKENS,
) -> list[TextChunk]:
    """
    Chunk markdown into overlapping segments of roughly ``target_tokens`` each.

    Uses tiktoken (cl100k) for a stable token count. Splits on section headers
    first, then sub-chunks long sections.
    """
    enc = tiktoken.get_encoding(_DEFAULT_ENCODING)
    out: list[TextChunk] = []
    for section_title, body in _split_by_headers(markdown):
        # Prefix section for retrieval context
        text = f"## {section_title}\n\n{body}"
        token_ids = enc.encode(text)
        if len(token_ids) <= target_tokens:
            out.append(TextChunk(text=enc.decode(token_ids), source_section=section_title))
            continue
        start = 0
        n = len(token_ids)
        while start < n:
            end = min(start + target_tokens, n)
            window = token_ids[start:end]
            out.append(
                TextChunk(
                    text=enc.decode(window).strip(),
                    source_section=section_title,
                )
            )
            if end >= n:
                break
            start = max(0, end - overlap_tokens)
    return out


__all__ = ["TextChunk", "chunk_markdown", "_split_by_headers"]
