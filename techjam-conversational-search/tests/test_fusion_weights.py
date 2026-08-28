"""Tests for learned fusion weights (Iteration 2, Task 4).

The fusion now uses logistic-regression weights learned on the public dev set to combine
[bm25, dense, slot, price] instead of the hand-set RRF_K / slot-boost combination. These
tests pin the learned-weight scoring, the price parsing fix, and the documented fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import starter.agent as mod  # noqa: E402
from starter.agent import Agent, FUSION_WEIGHTS, _parse_money  # noqa: E402


class FakeAgent(Agent):
    """Minimal agent that skips the expensive catalog build."""

    def __init__(self) -> None:
        self.products = {}


def _product(asin: str, price: object = None, description: str = "") -> dict:
    return {
        "parent_asin": asin,
        "title": asin,
        "categories": [],
        "features": [],
        "details": {},
        "store": "",
        "description": description,
        "price": price,
    }


def test_parse_money():
    assert _parse_money("$5.99") == 5.99
    assert _parse_money("from 5.99") == 5.99
    assert _parse_money("5.99 - 9.99") == 5.99
    assert _parse_money(5.99) == 5.99
    assert _parse_money(None) is None
    assert _parse_money("") is None
    assert _parse_money("n/a") is None


def test_fusion_weights_are_finite():
    for key in ("bm25", "dense", "slot", "price", "bias"):
        assert key in FUSION_WEIGHTS
        assert abs(FUSION_WEIGHTS[key]) < 1e6
        assert FUSION_WEIGHTS[key] == FUSION_WEIGHTS[key]  # not NaN


def test_linear_fusion_learned(monkeypatch):
    monkeypatch.setattr(mod, "USE_LEARNED_FUSION", True)
    agent = FakeAgent()
    w = FUSION_WEIGHTS
    expected = w["bm25"] * 1.0 + w["dense"] * 1.0 + w["slot"] * 0.5 + w["price"] * 0.0 + w["bias"]
    assert agent._linear_fusion([1.0, 1.0, 0.5, 0.0]) == pytest.approx(expected)


def test_linear_fusion_fallback(monkeypatch):
    monkeypatch.setattr(mod, "USE_LEARNED_FUSION", False)
    agent = FakeAgent()
    assert agent._linear_fusion([1.0, 1.0, 0.5, 0.0]) == pytest.approx((1.0 + 1.0) / 2 + 0.5 * 0.5)


def test_feature_vector(monkeypatch):
    agent = FakeAgent()
    agent.products = {"A": _product("A", description="cotton shirt")}
    feat = agent._feature_vector("A", {"material": "cotton"}, 0.8, 0.5)
    assert len(feat) == 4
    assert feat[0] == 0.8
    assert feat[1] == 0.5
    assert feat[2] == pytest.approx(1.0)  # material matches
    assert feat[3] == pytest.approx(0.0)  # no budget


def test_price_similarity_robust(monkeypatch):
    agent = FakeAgent()
    agent.products = {"A": _product("A", price="$5.99")}
    assert agent._price_similarity(agent.products["A"], {"budget": 5.99}) == pytest.approx(1.0)
    assert agent._price_similarity(agent.products["A"], {"budget": 11.98}) == pytest.approx(
        max(0.0, 1.0 - abs(5.99 - 11.98) / max(11.98, 1.0))
    )
    # Non-numeric prices / no budget must not crash and should score 0.
    assert agent._price_similarity(_product("B", price="from 5.99"), {"budget": 5.99}) == pytest.approx(1.0)
    assert agent._price_similarity(_product("C", price="n/a"), {"budget": 5.99}) == pytest.approx(0.0)
    assert agent._price_similarity(_product("D", price=5.0), {}) == pytest.approx(0.0)
