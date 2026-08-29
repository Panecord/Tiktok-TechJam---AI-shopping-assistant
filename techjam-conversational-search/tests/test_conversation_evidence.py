"""Regression tests for durable conversational evidence and slate novelty."""

from __future__ import annotations

import sys
from pathlib import Path

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
