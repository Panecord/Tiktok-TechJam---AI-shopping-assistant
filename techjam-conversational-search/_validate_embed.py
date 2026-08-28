"""Full-200-session validation of sentence-embedding dense retrieval (COPILOT_DENSE=embed).

Runs with USE_LEARNED_FUSION=False (hand-tuned) so the dense-retrieval change is isolated;
compare browsing HR against the TF-IDF + hand-tuned baseline (v2.1.0: browsing HR 0.525).
"""
import json
import os
import sys
import time

root = r"c:\Users\ImanKasni\OneDrive - Kuok (Singapore) Limited\Desktop\Work Documents\02 - Personal\07 - TT TechJam T4\techjam-conversational-search"
os.chdir(root)
sys.path.insert(0, root)

import starter.agent as mod
from starter.agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

os.environ["COPILOT_DENSE"] = "embed"
mod.USE_LEARNED_FUSION = False
samples = load_jsonl("data/public_set.jsonl")
catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
t0 = time.time()
print("embedding catalog...", flush=True)
agent = Agent("data/catalog.jsonl")
print(f"catalog embedded in {time.time()-t0:.0f}s (mode={agent.dense_mode})", flush=True)

n = len(samples)
hits = 0
rr_sum = 0.0
mttc_sum = 0.0
scen = {}
for i, sample in enumerate(samples):
    res = evaluate(agent, [sample], catalog_ids, categories, products)
    s = res["sessions"][0]
    if s["hit"]:
        hits += 1
        rr_sum += s["reciprocal_rank"]
        mttc_sum += s["first_hit_turn"]
    else:
        mttc_sum += 11
    sc = sample["scenario_type"]
    scen.setdefault(sc, {"n": 0, "hits": 0})
    scen[sc]["n"] += 1
    scen[sc]["hits"] += 1 if s["hit"] else 0
    if (i + 1) % 10 == 0 or i + 1 == n:
        el = time.time() - t0
        per = el / (i + 1)
        eta = per * (n - i - 1)
        print(f"{i+1}/{n} done | elapsed={el:.0f}s | eta={eta:.0f}s", flush=True)

hr = hits / n
mrr = rr_sum / n
mttc = mttc_sum / n
eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
score = 0.5 * hr + 0.3 * mrr + 0.2 * eff
summary = {
    "HR": hr, "MRR": mrr, "MTTC": mttc, "efficiency": eff, "score": score,
    "scenario_metrics": {k: {"HR": v["hits"] / v["n"], "n": v["n"]} for k, v in scen.items()},
    "seconds": round(time.time() - t0, 1),
}
with open("validation_embed.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("VALIDATION_DONE", json.dumps(summary))
