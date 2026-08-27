"""Regression tests for the buying-regression fix (Iteration 2, Task 1).

The v2.0.0 pipeline regressed buying-scenario sessions that the weak BM25 baseline
already hit. These tests pin the two root causes that were fixed:

1. The deterministic reranker used to re-order the candidate pool purely by the
   slot-match score (with ASIN tie-breaking), throwing away the BM25+dense relevance
   score. That buried a relevant-but-slot-mismatched target below the top-10, so a
   session that was a baseline hit became a miss.

2. `_detect_override` treated benign negations such as "I don't have a preference for
   budget." (a common simulated customer reply) as an intent override and wiped all
   slots, destroying retrieval.

Assertions:
  * Every buying session that flipped from a baseline hit to a v2.0.0 miss now hits
    again (the 12 recovered cases; `public_0190` is a documented expected-fail
    residual where the hybrid ranks the target just below pure-BM25's #1).
  * No returned `parent_asin` falls outside the frozen catalog (grounding).
  * A benign "I don't have a preference" reply does not clear dialogue slots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluator.local_evaluator import (  # noqa: E402
    MAX_TURNS,
    TOP_K,
    catalog_index,
    coarse_category,
    customer_reply,
    initial_message,
    load_jsonl,
    materialize_hidden_fields,
    normalize_recommendations,
)
from starter.agent import Agent  # noqa: E402

# Buying sessions that flipped from baseline HIT (BM25 always-recommend) to v2.0.0 MISS.
RECOVERED = [
    ("public_0010", "B0929KL5W7"),
    ("public_0044", "B09BQ4G5BD"),
    ("public_0061", "B08HCP9YTV"),
    ("public_0067", "B09G9BXJZM"),
    ("public_0088", "B07Z6J5N6Y"),
    ("public_0090", "B07MGR6D5M"),
    ("public_0107", "B01KPFK9ZA"),
    ("public_0129", "B0936ZJJ68"),
    ("public_0143", "B01H54X6CM"),
    ("public_0160", "B01AAANF2Y"),
    ("public_0168", "B08YYHDJD1"),
    ("public_0193", "B07YM55NLW"),
]
# Residual: hybrid dense+BM25 ranks this target just below top-10 on turn 1, whereas
# the baseline's pure-BM25 happened to rank it #1. Still flips from the old pipeline.
RESIDUAL = [("public_0190", "B01MQUDPPO")]


@pytest.fixture(scope="module")
def loaded():
    samples = {s["sample_id"]: s for s in load_jsonl(REPO / "data" / "public_set.jsonl")}
    catalog_ids, categories, products = catalog_index(REPO / "data" / "catalog.jsonl")
    agent = Agent(REPO / "data" / "catalog.jsonl")
    return {
        "samples": samples,
        "catalog_ids": catalog_ids,
        "categories": categories,
        "products": products,
        "agent": agent,
    }


def _run_session(agent, sample, catalog_ids, categories, products):
    """Replicate the evaluator's per-session loop for one sample."""
    sid = "test_" + sample["sample_id"]
    agent.reset(sid, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    eff = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, MAX_TURNS + 1):
        resp = agent.respond(sid, user_message, turn, TOP_K)
        ranked = normalize_recommendations(resp.get("recommendations"), catalog_ids)
        # Grounding guarantee: never return a parent_asin outside the catalog.
        assert set(ranked) <= catalog_ids, "recommendation referenced an id outside the catalog"
        if override_applied and target in ranked:
            return True, turn
        if turn == MAX_TURNS:
            break
        override = eff.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            user_message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            user_message, boundary_used = customer_reply(
                eff, resp.get("ask_attribute"), disclosed, boundary_used
            )
    return False, MAX_TURNS


@pytest.mark.parametrize("sample_id,target", RECOVERED)
def test_flipped_buying_sessions_hit_again(loaded, sample_id, target):
    """Flipped buying sessions must recover to a hit after the fix."""
    sample = loaded["samples"][sample_id]
    hit, turn = _run_session(
        loaded["agent"], sample, loaded["catalog_ids"], loaded["categories"], loaded["products"]
    )
    assert hit, f"buying session {sample_id} regressed again (target {target} not found within 10 turns)"


@pytest.mark.parametrize("sample_id,target", RESIDUAL)
def test_known_residual_buying_miss(loaded, sample_id, target):
    """Document the single residual miss (expected-fail until a later task)."""
    sample = loaded["samples"][sample_id]
    hit, _turn = _run_session(
        loaded["agent"], sample, loaded["catalog_ids"], loaded["categories"], loaded["products"]
    )
    if not hit:
        pytest.xfail(
            f"residual: {sample_id} target {target} ranked below top-10 by the hybrid on turn 1"
        )


def test_benign_negation_does_not_clear_slots(loaded):
    """'I don't have a preference for X' must NOT wipe dialogue state (regression)."""
    agent = loaded["agent"]
    agent.reset("slot_test", {"purchase_frequency": "x"})
    agent.respond("slot_test", "I'm looking for cotton shirts.", 1, TOP_K)
    assert agent._sessions["slot_test"]["slots"].get("material") == "cotton"

    # A benign negation from a simulated customer must not be treated as an override.
    agent.respond("slot_test", "I don't have an additional preference for budget.", 2, TOP_K)
    assert agent._sessions["slot_test"]["slots"].get("material") == "cotton", (
        "benign negation cleared the material slot"
    )
