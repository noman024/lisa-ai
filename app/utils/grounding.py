"""Lightweight checks that an answer is supported by supplied context text."""

from __future__ import annotations

import re
import string
from typing import Set


_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "or",
    "and", "but", "if", "it", "its", "this", "that", "these", "those", "i",
    "you", "we", "they", "he", "she", "may", "can", "could", "should", "would",
    "not", "no", "yes", "per", "any", "all", "each", "per", "also", "use",
    "when", "where", "which", "who", "how", "what", "will", "into", "such",
}


def _tokens(text: str) -> set[str]:
    t = text.lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    return {w for w in t.split() if len(w) > 2 and w not in _STOPWORDS}


def grounding_score(answer: str, context: str) -> float:
    """
    Return the fraction of content words in ``answer`` that appear in ``context``.

    1.0 means all scored words in the answer appear in the context; 0.0 if none.
    If there are no scorable words in the answer, returns 0.0.
    """
    a, c = _tokens(answer), _tokens(context)
    if not a:
        return 0.0
    overlap = len(a & c)
    return overlap / len(a)
