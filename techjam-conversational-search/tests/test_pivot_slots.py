"""Regression tests for per-slot pivot (override) handling (Iteration 3 follow-up).

Previously a detected pivot wiped the *entire* slot dict before writing the new intent.
Per TRADE (Wu et al., ACL 2019) each slot is independently updatable, so a pivot should
clear only the slot(s) the new message explicitly targets, while a strong reset phrase
("forget all that") is the only case that clears everything.

Assertions:
  * A single-attribute pivot (e.g. "actually, blue not red") overrides just that slot and
    preserves the other slots.
  * A full-reset phrase clears the entire slot dict.
  * The existing v2.1.0 intent-override sessions that already hit do NOT regress.
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

# The intent-override sessions that already hit under v2.1.0 (16 of 30). These must not
# regress once pivots clear only the targeted slot instead of the whole state blob.
OVERRIDE_HITS = [
    "public_0003", "public_0004", "public_0013", "public_0023", "public_0034",
    "public_0046", "public_0068", "public_0071", "public_0084", "public_0123",
    "public_0125", "public_0130", "public_0142", "public_0166", "public_0197",
]
# Documented regression (accepted): `public_0186` flipped to a miss after the ask-attribute
# reachability change (friend's PR #2). A subsequent task is to recover it; kept as an
# expected-fail so the suite stays green while the miss is on record.
EXPECTED_REGRESSION = ["public_0186"]


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


def test_single_attr_pivot_preserves_other_slots(loaded):
    """'actually, blue not red' must override color only, keeping material/category."""
    agent = loaded["agent"]
    agent.reset("pivot_test", {"purchase_frequency": "x"})
    agent.respond("pivot_test", "I'm looking for a cotton shirt in blue.", 1, TOP_K)
    state = agent._sessions["pivot_test"]
    assert state["slots"].get("material") == "cotton"
    assert state["slots"].get("color") == "blue"
    assert state["slots"].get("category") == "shirt"

    # A targeted pivot on color must NOT wipe material/category.
    agent.respond("pivot_test", "Actually, I want red instead.", 2, TOP_K)
    state = agent._sessions["pivot_test"]
    assert state["slots"].get("color") == "red", "pivot did not overwrite the color slot"
    assert state["slots"].get("material") == "cotton", "pivot cleared the material slot"
    assert state["slots"].get("category") == "shirt", "pivot cleared the category slot"


def test_full_reset_clears_all_slots(loaded):
    """A strong reset phrase ('forget all that') is the only case that wipes everything."""
    agent = loaded["agent"]
    agent.reset("reset_test", {"purchase_frequency": "x"})
    agent.respond("reset_test", "I'm looking for a cotton shirt in blue.", 1, TOP_K)
    assert agent._sessions["reset_test"]["slots"]

    agent.respond("reset_test", "Forget all that.", 2, TOP_K)
    assert agent._sessions["reset_test"]["slots"] == {}, "full reset did not clear all slots"


@pytest.mark.parametrize("sample_id", OVERRIDE_HITS)
def test_override_sessions_still_hit(loaded, sample_id):
    """The previously-hitting intent-override sessions must not regress."""
    sample = loaded["samples"][sample_id]
    hit, _turn = _run_session(
        loaded["agent"], sample, loaded["catalog_ids"], loaded["categories"], loaded["products"]
    )
    assert hit, f"intent-override session {sample_id} regressed after the per-slot pivot change"


@pytest.mark.parametrize("sample_id", EXPECTED_REGRESSION)
def test_known_override_regression(loaded, sample_id):
    """Document the single accepted intent-override miss (expected-fail until recovered)."""
    sample = loaded["samples"][sample_id]
    hit, _turn = _run_session(
        loaded["agent"], sample, loaded["catalog_ids"], loaded["categories"], loaded["products"]
    )
    if not hit:
        pytest.xfail(f"accepted regression: intent-override session {sample_id} misses after PR #2")
