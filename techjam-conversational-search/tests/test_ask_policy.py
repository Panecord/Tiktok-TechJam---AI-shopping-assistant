"""Isolated unit tests for the ask-vs-recommend policy and retrieval helpers.

These tests avoid the 50k-product catalog build by subclassing `Agent` and overriding
only the methods under test's dependencies. They pin the deterministic policy:

  * force-recommend when turn >= FORCE_RECOMMEND_TURN;
  * recommend when candidate pool is small AND the top-2 margin is decisive;
  * otherwise ask a clarifying question;
  * `_choose_ask_attribute` picks the max-entropy attribute (skipping budget/category
    and attributes already asked);
  * `_bm25_query` produces a de-duplicated quoted OR expression.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from starter.agent import (  # noqa: E402
    Agent,
    ALLOWED_ASK_ATTRIBUTES,
    FORCE_RECOMMEND_TURN,
    K_SMALL,
)


class _PolicyAgent(Agent):
    """Stub Agent that bypasses catalog build and lets tests drive the policy."""

    def __init__(self):
        self._sessions = {}
        self.products = {}
        self._searchable_lc = {}
        self._candidate_features = {}
        self._last_fused = {}
        self._last_max_fused = 1.0
        self._candidates: list[str] = []
        self._scores: list[float] = []

    def _retrieve(self, message, top_k, slots, context="", intent="buying", profile_context=""):
        return self._candidates

    def _top_scores(self, candidate_list, slots):
        return self._scores

    def _rerank(self, candidate_list, slots, use_llm=True, recent_turns=None):
        return candidate_list[:10], {"prompt_tokens": 0, "completion_tokens": 0}

    def _choose_ask_attribute(self, candidate_list, question_history, profile_tags=None):
        return "color"


# -- On-demand helper (module-level to mirror how agent.py exposes these) ----------

def _run(agent: Agent, message: str, turn: int, top_k: int = 10) -> dict:
    agent.reset("s1", {})
    return agent.respond("s1", message, turn, top_k)


@pytest.fixture
def agent() -> _PolicyAgent:
    return _PolicyAgent()


# -- Ask-vs-recommend policy branches ----------------------------------------------

def test_force_recommend_at_turn_cap(agent):
    # Large pool + tiny margin, but turn cap forces a recommendation anyway.
    agent._candidates = ["A", "B", "C", "D"]
    agent._scores = [1.0, 0.99, 0.98, 0.97]
    reply = _run(agent, "suggest something", FORCE_RECOMMEND_TURN)
    assert reply["ask_attribute"] is None
    assert reply["recommendations"]


def test_recommend_when_pool_small_and_margin_decisive(agent):
    agent._candidates = ["A", "B", "C"]
    agent._scores = [1.0, 0.5, 0.4]  # margin = 0.5 >= MARGIN_THRESHOLD
    assert len(agent._candidates) <= K_SMALL
    reply = _run(agent, "show me", 1)
    assert reply["ask_attribute"] is None
    assert reply["recommendations"]


def test_ask_when_unconfident(agent):
    # Small pool but margin below threshold -> must ask for clarification.
    agent._candidates = ["A", "B"]
    agent._scores = [1.0, 0.95]  # margin = 0.05 < MARGIN_THRESHOLD
    reply = _run(agent, "something nice", 1)
    assert reply["ask_attribute"] == "color"


def test_ask_when_pool_large(agent):
    # Large pool but decisive top-2 margin -> the turn cap not reached, so ask.
    agent._candidates = [f"P{i}" for i in range(K_SMALL + 1)]
    agent._scores = [1.0, 0.5, 0.4]
    reply = _run(agent, "show me", 1)
    assert reply["ask_attribute"] == "color"


def test_recommendations_respect_top_k(agent):
    agent._candidates = list("ABCDEFGHIJKLMNO")
    agent._scores = [1.0, 0.9, 0.8]
    reply = _run(agent, "suggest something", FORCE_RECOMMEND_TURN, top_k=5)
    assert len(reply["recommendations"]) == 5


# -- _choose_ask_attribute entropy selection ---------------------------------------

class _ChooseAgent(Agent):
    def __init__(self, products, searchable):
        self.products = products
        self._searchable_lc = searchable


def _choose(agent, candidates, history):
    return agent._choose_ask_attribute(candidates, history)


def test_choose_picks_max_entropy_attr():
    # material 2:2 (entropy 1.0) vs color 2:2 (1.0); material comes first in priority.
    products = {
        "A": {"title": "cotton red shirt", "categories": ["shoes"], "price": 10},
        "B": {"title": "cotton red shirt", "categories": ["clothing"], "price": 20},
        "C": {"title": "spandex blue shirt", "categories": ["hats"], "price": 30},
        "D": {"title": "spandex blue shirt", "categories": ["bags"], "price": 40},
    }
    searchable = {k: v["title"].lower() for k, v in products.items()}
    agent = _ChooseAgent(products, searchable)
    chosen = _choose(agent, list(products), [])
    assert chosen in ALLOWED_ASK_ATTRIBUTES
    # Budget/category have the highest entropy but are excluded by policy.
    assert chosen not in {"budget", "category"}
    assert chosen == "material"


def test_choose_skips_already_asked():
    products = {
        "A": {"title": "cotton red shirt", "categories": [], "price": None},
        "B": {"title": "cotton red shirt", "categories": [], "price": None},
        "C": {"title": "spandex red shirt", "categories": [], "price": None},
    }
    searchable = {k: v["title"].lower() for k, v in products.items()}
    agent = _ChooseAgent(products, searchable)
    # material is the only informative attr; asking material again must be skipped.
    # ANSWERABILITY_PRIORITY then lands on "feature" (an always-eligible question).
    assert _choose(agent, list(products), ["material"]) == "feature"


def test_choose_empty_candidate_returns_other():
    agent = _ChooseAgent({}, {})
    assert _choose(agent, [], []) == "other"


def test_choose_uniform_pool_returns_other():
    products = {
        "A": {"title": "cotton red shirt", "categories": [], "price": None},
        "B": {"title": "cotton red shirt", "categories": [], "price": None},
    }
    searchable = {k: v["title"].lower() for k, v in products.items()}
    agent = _ChooseAgent(products, searchable)
    # No attribute splits the uniform pool, but "feature" is always an eligible
    # high-value question under ANSWERABILITY_PRIORITY.
    assert _choose(agent, list(products), []) == "feature"


# -- _bm25_query -------------------------------------------------------------------

def test_bm25_query_folds_message_and_slot_terms():
    agent = _PolicyAgent()
    q = agent._bm25_query("blue jeans", {"color": "blue", "material": "cotton"})
    assert q == '"blue" OR "jeans" OR "cotton"'


def test_bm25_query_dedupes_terms():
    agent = _PolicyAgent()
    q = agent._bm25_query("blue shirt", {"color": "blue"})
    assert q.count('"blue"') == 1


def test_bm25_query_limits_term_count():
    agent = _PolicyAgent()
    q = agent._bm25_query(" ".join(f"w{i}" for i in range(60)), {})
    assert q.count(" OR ") < 60
