"""Clarifying-question quality diagnostic (Task 6).

Aliannejadi et al. (SIGIR 2019) argue a well-chosen clarifying question should measurably
reduce ambiguity, not just look sensible. Here the proxy for ambiguity is the **candidate
pool size**: a good attribute to ask about should, once answered, narrow the pool more than
a random attribute would.

For every ask turn in the public set we record:
  * pool_before = fused candidate pool size at the ask turn, and
  * pool_after  = fused candidate pool size on the following turn (after the simulated
                  customer answers the asked attribute).
  * reduction  = pool_before - pool_after (>= 0 when the answer narrows the pool).

We compare the entropy-based policy (`_choose_ask_attribute`, the agent's real selection)
against a random-attribute baseline. If the entropy policy's mean reduction is meaningfully
larger, the selection is doing real work rather than just filling the `ask_attribute` field.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

from evaluator.local_evaluator import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
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

samples = load_jsonl(root / "data" / "public_set.jsonl")
catalog_ids, categories, products = catalog_index(root / "data" / "catalog.jsonl")
agent = Agent(root / "data" / "catalog.jsonl")

_orig_choose_attr = agent._choose_ask_attribute


def run_policy(ask_policy: str) -> dict:
    """Run the full set under an ask policy and collect per-ask pool reductions."""
    if ask_policy == "random":
        def _random_attr(candidate_list, question_history):
            allowed = [a for a in ALLOWED_ATTRIBUTES if a not in question_history] or list(ALLOWED_ATTRIBUTES)
            return random.choice(allowed)
        agent._choose_ask_attribute = _random_attr  # type: ignore[method-assign]

    total_reduction = 0.0
    n_asks = 0
    zero_reduction_asks = 0
    per_scenario = {}
    t0 = time.time()

    for sample in samples:
        sid = "aq_" + sample["sample_id"]
        agent.reset(sid, sample["user_profile"])
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = materialize_hidden_fields(sample, products)
        eff = {**sample, "intent_card": card, "behavior": behavior}
        disclosed: set[str] = set()
        boundary_used = False
        override_applied = sample["scenario_type"] != "intent_override"
        user_message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)
        pool_sizes: list[int] = []
        ask_flags: list[bool] = []
        for turn in range(1, MAX_TURNS + 1):
            resp = agent.respond(sid, user_message, turn, TOP_K)
            pool_sizes.append(len(getattr(agent, "_last_candidates", None) or []))
            ask_flags.append(resp.get("ask_attribute") is not None)
            ranked = normalize_recommendations(resp.get("recommendations"), catalog_ids)
            if override_applied and target in ranked:
                break
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
        # Compute reductions: for each ask turn, compare pool size to the next turn.
        for t in range(len(pool_sizes)):
            if ask_flags[t] and t + 1 < len(pool_sizes):
                before = pool_sizes[t]
                after = pool_sizes[t + 1]
                reduction = max(0.0, before - after)
                total_reduction += reduction
                n_asks += 1
                if reduction <= 0.0:
                    zero_reduction_asks += 1
                sc = sample["scenario_type"]
                d = per_scenario.setdefault(sc, {"n": 0, "reduction": 0.0})
                d["n"] += 1
                d["reduction"] += reduction
    # Restore the real policy.
    agent._choose_ask_attribute = _orig_choose_attr  # type: ignore[method-assign]

    mean_reduction = total_reduction / n_asks if n_asks else 0.0
    return {
        "policy": ask_policy,
        "n_sessions": len(samples),
        "n_asks": n_asks,
        "mean_pool_reduction": round(mean_reduction, 4),
        "mean_reduction_pct": round(mean_reduction / 300.0 * 100.0, 4),
        "zero_reduction_share": round(zero_reduction_asks / n_asks, 4) if n_asks else None,
        "scenario": {
            k: {"n": v["n"], "mean_reduction": round(v["reduction"] / v["n"], 4)}
            for k, v in sorted(per_scenario.items())
        },
        "seconds": round(time.time() - t0, 1),
    }


# Reproducible random baseline.
random.seed(42)
entropy = run_policy("entropy")
random.seed(42)
random_baseline = run_policy("random")

summary = {
    "entropy_policy": entropy,
    "random_baseline": random_baseline,
    "reduction_gain": round(entropy["mean_pool_reduction"] - random_baseline["mean_pool_reduction"], 4),
    "reduction_gain_pct": round(
        (entropy["mean_pool_reduction"] / random_baseline["mean_pool_reduction"] - 1.0) * 100.0, 4
    ) if random_baseline["mean_pool_reduction"] else None,
}
with open(root / "validation_ask_quality.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("VALIDATION_DONE", json.dumps(summary))
