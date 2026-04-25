"""SQA: app.utils.grounding — word-overlap score behaviour and edge cases."""

from __future__ import annotations

import pytest

from app.utils.grounding import grounding_score


def test_grounding_perfect_overlap() -> None:
    assert (
        grounding_score(
            "beneficiaries and the death benefit",
            "death benefit and beneficiaries named in the policy",
        )
        == 1.0
    )


def test_grounding_partial_overlap() -> None:
    s = grounding_score("term life and zebras", "term life insurance policies")
    assert 0.0 < s < 1.0
    # "zebras" is not in context; two words match of three scorable? term, life — zebras
    # term, life, zebras -> term and life in context? "term" "life" in "term life insurance" yes
    # Actually zebras is third - overlap 2/3
    assert abs(s - 2 / 3) < 0.01


def test_grounding_no_content_words() -> None:
    assert grounding_score("a an the is of", "substantive context here with many words") == 0.0


def test_grounding_empty_answer() -> None:
    assert grounding_score("", "any context long enough") == 0.0


def test_grounding_punctuation_stripped() -> None:
    a = grounding_score("premiums, fixed!", "Fixed premiums and rates")
    b = grounding_score("premiums fixed", "Fixed premiums and rates")
    assert a == b == 1.0


@pytest.mark.parametrize(
    "answer,context,expected",
    [
        ("", "x", 0.0),
        ("x", "", 0.0),
    ],
)
def test_grounding_empty_either_side(answer: str, context: str, expected: float) -> None:
    assert grounding_score(answer, context) == expected
