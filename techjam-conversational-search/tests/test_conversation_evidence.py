"""Regression tests for durable conversational evidence and slate novelty."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from starter.agent import Agent, _extract_constraint_evidence  # noqa: E402


def test_extracts_arbitrary_feature_reply() -> None:
    message = "For that, what matters is: nickel free with an adjustable clasp."
    assert _extract_constraint_evidence(message) == "nickel free with an adjustable clasp"


def test_ignores_no_preference_reply() -> None:
    assert _extract_constraint_evidence(
        "I don't have an additional preference for color."
    ) is None


def test_novel_slate_uses_unseen_candidates_first() -> None:
    ranked = ["A", "B", "C", "D", "E"]
    assert Agent._novel_slate(ranked, {"A", "C"}, 3) == ["B", "D", "E"]


def test_novel_slate_repeats_only_to_fill_small_pool() -> None:
    ranked = ["A", "B", "C"]
    assert Agent._novel_slate(ranked, {"A", "B"}, 3) == ["C", "A", "B"]


def test_multi_constraint_reply_is_split_before_card_matching() -> None:
    """A single turn can disclose several constraints joined by '; '.

    Regression: the joined blob was matched against the constraint card as ONE string,
    which can never equal a single card entry, so the exact-match branch silently
    degraded to partial coverage and the strongest ranking signal was lost. Open-ended
    clarification returns two constraints at once, so this fired constantly.
    """
    agent = Agent.__new__(Agent)
    agent._constraint_card_cache = {}
    agent.products = {
        "p": {
            "title": "Cotton crew tee",
            "features": ["cotton", "color: grey", "Machine washable"],
            "details": {},
        }
    }
    joined = agent._constraint_card_score("p", {"free_text_constraints": ["cotton; color: grey"]})
    separate = agent._constraint_card_score("p", {"free_text_constraints": ["cotton", "color: grey"]})
    assert joined == pytest.approx(separate)
    assert joined >= 0.9
