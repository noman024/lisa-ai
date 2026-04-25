"""SQA: markdown chunking and section split — app.utils.chunking."""

from __future__ import annotations

import pytest

from app.utils.chunking import TextChunk, _split_by_headers, chunk_markdown


def test_split_by_headers_empty() -> None:
    assert _split_by_headers("") == []


def test_split_by_headers_whitespace_only() -> None:
    assert _split_by_headers("  \n  ") == []


def test_split_by_headers_preamble_no_h2() -> None:
    md = "Intro line without h2\n\njust text."
    out = _split_by_headers(md)
    assert len(out) == 1
    assert out[0][0] == "Overview"
    assert "Intro line" in out[0][1]


def test_split_by_headers_one_section() -> None:
    md = "## Term Life\n\nThis is the body of term life content."
    out = _split_by_headers(md)
    assert out == [("Term Life", "This is the body of term life content.")]


def test_split_by_headers_two_sections() -> None:
    md = (
        "## Section A\n\nFirst body.\n"
        "## Section B\n\nSecond body with more text."
    )
    out = _split_by_headers(md)
    assert [t for t, _ in out] == ["Section A", "Section B"]
    assert "First body" in out[0][1]
    assert "Second body" in out[1][1]


def test_chunk_markdown_produces_text_chunks() -> None:
    md = "## Small\n\n" + "word " * 50
    chunks = chunk_markdown(md, target_tokens=400, overlap_tokens=60)
    assert all(isinstance(c, TextChunk) for c in chunks)
    assert all(c.source_section == "Small" for c in chunks)
    assert all(len(c.text) > 0 for c in chunks)


def test_chunk_markdown_tiny_section_single_chunk() -> None:
    md = "## Tiny\n\nShort."
    chunks = chunk_markdown(md)
    assert len(chunks) == 1
    assert "Tiny" in chunks[0].text
    assert "Short" in chunks[0].text


def test_chunk_markdown_long_section_creates_sliding_windows() -> None:
    # Far more than 400 cl100k tokens: force multiple windows with overlap
    long_body = ("Term life insurance and premiums. " * 2000) + "\n" * 2
    md = f"## LongSection\n\n{long_body}"
    chunks = chunk_markdown(md, target_tokens=400, overlap_tokens=60)
    assert len(chunks) >= 2
    assert all(c.source_section == "LongSection" for c in chunks)
    combined = " ".join(c.text for c in chunks)
    assert "Term life" in combined
