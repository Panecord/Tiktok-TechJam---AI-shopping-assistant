"""Full-200-session validation of the LLM listwise reranker with real token cost.

Loads credentials from .env, forces the valid lowercase model, uses the hand-tuned base
fusion (USE_LEARNED_FUSION=False) so the LLM rerank effect is isolated, and reports
per-session token usage plus LLM call count.
"""
import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
sys.path.insert(0, str(root))

# Load .env (strip surrounding quotes) WITHOUT printing secrets.
env_path = root / ".env"
if os.path.exists(env_path):
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
                    v = v[1:-1]
                os.environ.setdefault(k.strip(), v.strip())
os.environ["COPILOT_LLM_MODEL"] = "deepseek-v4-flash"

import starter.agent as mod
from starter.agent import Agent
from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl

mod.USE_LEARNED_FUSION = False
samples = load_jsonl(root / "data" / "public_set.jsonl")
catalog_ids, categories, products = catalog_index(root / "data" / "catalog.jsonl")
t0 = time.time()
with open(root / "validation_llm_progress.txt", "w", encoding="utf-8") as pf:
    pf.write("building agent (TF-IDF dense, LLM enabled)...\n")
agent = Agent(root / "data" / "catalog.jsonl")
with open(root / "validation_llm_progress.txt", "a", encoding="utf-8") as pf:
    pf.write(f"agent ready in {time.time()-t0:.0f}s, starting sessions...\n")

_real_call = agent._call_llm_rerank
llm_calls = {"n": 0}


def _wrap(cl, slots, model, url, key, recent_turns=None):
    llm_calls["n"] += 1
    return _real_call(cl, slots, model, url, key, recent_turns)


agent._call_llm_rerank = _wrap

n = len(samples)
hits = 0
rr_sum = 0.0
mttc_sum = 0.0
scen = {}
tp = 0
tc = 0
for i, sample in enumerate(samples):
    res = evaluate(agent, [sample], catalog_ids, categories, products)
    s = res["sessions"][0]
    if s["hit"]:
        hits += 1
        rr_sum += s["reciprocal_rank"]
        mttc_sum += s["first_hit_turn"]
    else:
        mttc_sum += 11
    tp += res["reported_token_usage"]["prompt_tokens"]
    tc += res["reported_token_usage"]["completion_tokens"]
    sc = sample["scenario_type"]
    scen.setdefault(sc, {"n": 0, "hits": 0})
    scen[sc]["n"] += 1
    scen[sc]["hits"] += 1 if s["hit"] else 0
    if (i + 1) % 5 == 0 or i + 1 == n:
        el = time.time() - t0
        per = el / (i + 1)
        eta = per * (n - i - 1)
        line = (
            f"{i+1}/{n} done | elapsed={el:.0f}s | eta={eta:.0f}s | llm_calls={llm_calls['n']} "
            f"| hits={hits} | hr={hits / max(1, i + 1):.3f}"
        )
        print(line, flush=True)
        with open(root / "validation_llm_progress.txt", "w", encoding="utf-8") as pf:
            pf.write(line + "\n")

hr = hits / n
mrr = rr_sum / n
mttc = mttc_sum / n
eff = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
score = 0.5 * hr + 0.3 * mrr + 0.2 * eff
summary = {
    "HR": hr, "MRR": mrr, "MTTC": mttc, "efficiency": eff, "score": score,
    "scenario_metrics": {k: {"HR": v["hits"] / v["n"], "n": v["n"]} for k, v in scen.items()},
    "llm_calls": llm_calls["n"],
    "prompt_tokens": tp, "completion_tokens": tc, "total_tokens": tp + tc,
    "avg_tokens_per_session": round((tp + tc) / n, 1),
    "seconds": round(time.time() - t0, 1),
}
with open(root / "validation_llm.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("VALIDATION_DONE", json.dumps(summary))
