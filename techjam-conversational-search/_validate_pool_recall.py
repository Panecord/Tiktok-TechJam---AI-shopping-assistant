"""Pool-recall diagnostic vs. Hit Rate@10 (Task 1.5).

For every session in the public set, this records whether the target `parent_asin` is
present anywhere in the fused candidate pool built by `_retrieve()` (independent of what
is actually returned in the top-10). It reports **pool recall** alongside the standard
HR@10 / MRR / MTTC / TechnicalScore so the retrieval-vs-ranking bottleneck is visible
without rebuilding this diagnostic each iteration.

Interpretation:
  * pool recall >> HR@10  -> the bottleneck is ranking/selection, NOT retrieval. Enlarging
    BM25_TOP / DENSE_TOP would add cost/noise without fixing the real miss.
  * pool recall ~ HR@10  -> retrieval genuinely is the ceiling; widening the pool has real
    room to help.
  * Neither can reach 1.0: `top_k` is fixed, some sessions are deliberately ambiguous, and
    the (simulated) user must reveal enough disambiguating information within the turn budget.
"""
import json
import sys
import time
from collections import defaultdict

root = r"c:\Users\ImanKasni\OneDrive - Kuok (Singapore) Limited\Desktop\Work Documents\02 - Personal\07 - TT TechJam T4\techjam-conversational-search"
import os
os.chdir(root)
sys.path.insert(0, root)

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl  # noqa: E402
from starter.agent import Agent  # noqa: E402

samples = load_jsonl("data/public_set.jsonl")
catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
agent = Agent("data/catalog.jsonl")

_orig_respond = agent.respond
captured = {"pools": [], "target": None}


def _wrapped_respond(session_id, user_message, turn, top_k):
    resp = _orig_respond(session_id, user_message, turn, top_k)
    # The fused candidate pool built by _retrieve() this turn (exposed for diagnostics).
    captured["pools"].append(list(getattr(agent, "_last_candidates", None) or []))
    return resp


agent.respond = _wrapped_respond

n = len(samples)
hits = 0
pool_recall = 0
rr_sum = 0.0
mttc_sum = 0.0
scen: dict[str, dict] = defaultdict(lambda: {"n": 0, "hits": 0, "recall": 0})
t0 = time.time()
for i, sample in enumerate(samples):
    target = str(sample["ground_truth"]["parent_asin"])
    captured["pools"] = []
    res = evaluate(agent, [sample], catalog_ids, categories, products)
    s = res["sessions"][0]
    present = any(target in pool for pool in captured["pools"])
    if present:
        pool_recall += 1
    if s["hit"]:
        hits += 1
        rr_sum += s["reciprocal_rank"]
        mttc_sum += s["first_hit_turn"]
    else:
        mttc_sum += 11
    sc = sample["scenario_type"]
    scen[sc]["n"] += 1
    scen[sc]["hits"] += 1 if s["hit"] else 0
    scen[sc]["recall"] += 1 if present else 0
    if (i + 1) % 20 == 0 or i + 1 == n:
        el = time.time() - t0
        print(
            f"{i+1}/{n} | elapsed={el:.0f}s | pool_recall={pool_recall/max(1,i+1):.3f} "
            f"| hr={hits/max(1,i+1):.3f}",
            flush=True,
        )

agent.respond = _orig_respond

hr = hits / n
pool_recall_rate = pool_recall / n
mrr = rr_sum / n
mttc = mttc_sum / n
eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
score = 0.5 * hr + 0.3 * mrr + 0.2 * eff

summary = {
    "sample_count": n,
    "pool_recall": round(pool_recall_rate, 6),
    "hit_rate_at_10": round(hr, 6),
    "mrr": round(mrr, 6),
    "mttc": round(mttc, 6),
    "efficiency": round(eff, 6),
    "technical_score": round(score, 6),
    "rank_gap": round(pool_recall_rate - hr, 6),
    "scenario_metrics": {
        k: {
            "n": v["n"],
            "pool_recall": round(v["recall"] / v["n"], 6),
            "hit_rate_at_10": round(v["hits"] / v["n"], 6),
        }
        for k, v in sorted(scen.items())
    },
    "seconds": round(time.time() - t0, 1),
}
with open("validation_pool_recall.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("VALIDATION_DONE", json.dumps(summary))
