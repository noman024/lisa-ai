"""SQA: small pure helpers in app.agent.nodes (trim, strip, collapse)."""

from __future__ import annotations

import pytest

from app.agent.nodes import (
    _collapse_repeated_fallback,
    _should_collapse_to_fallback,
    _strip_for_grounding,
    _trim_context,
)
from app.agent.state import FALLBACK_MESSAGE


def test_trim_context_within_budget() -> None:
    t = "short"
    assert _trim_context(t, max_chars=2000) == t


def test_trim_context_truncates_paragraphs() -> None:
    paras = [f"p{n}" for n in range(20)]
    big = "\n\n".join(paras)
    out = _trim_context(big, max_chars=20)
    assert len(out) <= 20
    # fallback when no whole paragraph fits
    single = "x" * 100
    out2 = _trim_context(single, max_chars=20)
    assert len(out2) == 20


def test_strip_for_grounding_removes_labels() -> None:
    out = _strip_for_grounding("Definition:  fixed premiums")
    assert "Definition:" not in out
    assert "fixed" in out.lower() and "premiums" in out.lower()


def test_collapse_repeated_fallback_lines() -> None:
    multi = f"{FALLBACK_MESSAGE}\n{FALLBACK_MESSAGE}\n{FALLBACK_MESSAGE}"
    assert _collapse_repeated_fallback(multi) == FALLBACK_MESSAGE


def test_collapse_does_not_touch_single_line() -> None:
    t = "Term life is affordable."
    assert _collapse_repeated_fallback(t) == t


def test_should_collapse_ido_not_know() -> None:
    assert _should_collapse_to_fallback("I do not know. The sky is blue.") is True


def test_should_not_collapse_factual_continuation() -> None:
    assert _should_collapse_to_fallback("I don't know. Universal life is flexible.") is False
