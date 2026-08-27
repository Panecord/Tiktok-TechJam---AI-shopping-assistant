"""Tests for the grounded LLM listwise reranker (Iteration 2, Task 2).

The LLM reranker must:
  * only operate on a TRIMMED pool (top `LLM_TOP` by the deterministic score),
  * only ever return ids that are members of that pool (grounding),
  * retry once on a grounding violation / error, then fall back to the deterministic
    reranker,
  * report real token usage (accumulated across all calls).

These tests use a lightweight fake agent (no catalog build) and monkeypatch the LLM
transport so they run quickly and deterministically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from starter.agent import Agent, LLM_KEY_ENV, LLM_TOP, LLM_URL_ENV  # noqa: E402


def _basic_product(asin: str) -> dict:
    return {
        "parent_asin": asin,
        "title": f"product {asin}",
        "categories": ["clothing"],
        "features": [],
        "details": {},
        "store": "s",
        "description": asin,
    }


class FakeAgent(Agent):
    """Minimal agent that skips the expensive catalog build."""

    def __init__(self) -> None:
        self.products = {
            "A": _basic_product("A"),
            "B": _basic_product("B"),
            "C": _basic_product("C"),
        }
        self._last_fused = {"A": 0.3, "B": 0.2, "C": 0.1}
        self._last_max_fused = 0.3
        self._sessions = {}


@pytest.fixture(autouse=True)
def _no_llm_by_default(monkeypatch):
    monkeypatch.delenv(LLM_URL_ENV, raising=False)
    monkeypatch.delenv(LLM_KEY_ENV, raising=False)


def test_no_llm_uses_deterministic_zero_usage(monkeypatch):
    agent = FakeAgent()
    ranked, usage = agent._rerank(["A", "B", "C"], {})
    assert ranked == ["A", "B", "C"]  # deterministic combined-score order
    assert usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_valid_llm_output_is_used(monkeypatch):
    monkeypatch.setenv(LLM_URL_ENV, "http://llm")
    monkeypatch.setenv(LLM_KEY_ENV, "k")
    agent = FakeAgent()

    def fake(candidate_list, slots, model, url, key):
        return (["B", "A", "C"], {"prompt_tokens": 10, "completion_tokens": 2})

    monkeypatch.setattr(agent, "_call_llm_rerank", fake)
    ranked, usage = agent._rerank(["A", "B", "C"], {})
    assert ranked == ["B", "A", "C"]
    assert usage == {"prompt_tokens": 10, "completion_tokens": 2}


def test_grounding_retry_then_valid(monkeypatch):
    monkeypatch.setenv(LLM_URL_ENV, "http://llm")
    monkeypatch.setenv(LLM_KEY_ENV, "k")
    agent = FakeAgent()
    calls = {"n": 0}

    def fake(candidate_list, slots, model, url, key):
        calls["n"] += 1
        if calls["n"] == 1:
            return (["B", "X"], {"prompt_tokens": 10, "completion_tokens": 2})  # X not a candidate
        return (["B", "A", "C"], {"prompt_tokens": 10, "completion_tokens": 2})

    monkeypatch.setattr(agent, "_call_llm_rerank", fake)
    ranked, usage = agent._rerank(["A", "B", "C"], {})
    assert ranked == ["B", "A", "C"]
    assert calls["n"] == 2  # retried once
    assert usage == {"prompt_tokens": 20, "completion_tokens": 4}


def test_grounding_retry_then_fallback(monkeypatch):
    monkeypatch.setenv(LLM_URL_ENV, "http://llm")
    monkeypatch.setenv(LLM_KEY_ENV, "k")
    agent = FakeAgent()
    calls = {"n": 0}

    def fake(candidate_list, slots, model, url, key):
        calls["n"] += 1
        return (["X"], {"prompt_tokens": 10, "completion_tokens": 2})  # invalid both times

    monkeypatch.setattr(agent, "_call_llm_rerank", fake)
    ranked, usage = agent._rerank(["A", "B", "C"], {})
    assert ranked == ["A", "B", "C"]  # deterministic fallback
    assert calls["n"] == 2
    assert usage == {"prompt_tokens": 20, "completion_tokens": 4}


def test_pool_trimmed_before_llm(monkeypatch):
    monkeypatch.setenv(LLM_URL_ENV, "http://llm")
    monkeypatch.setenv(LLM_KEY_ENV, "k")
    agent = FakeAgent()
    products = {}
    fused = {}
    cands = []
    for k in range(30):
        asin = f"i{k}"
        products[asin] = _basic_product(asin)
        fused[asin] = (30 - k) / 100.0  # i0 highest
        cands.append(asin)
    agent.products = products
    agent._last_fused = fused
    agent._last_max_fused = 0.30
    seen = {}

    def fake(candidate_list, slots, model, url, key):
        seen["n"] = len(candidate_list)
        seen["first"] = candidate_list[0]
        return (candidate_list, {"prompt_tokens": 1, "completion_tokens": 1})

    monkeypatch.setattr(agent, "_call_llm_rerank", fake)
    ranked, usage = agent._rerank(cands, {})
    assert seen["n"] == LLM_TOP
    assert seen["first"] == "i0"
    assert ranked == cands  # LLM rank + remaining candidates in deterministic order
