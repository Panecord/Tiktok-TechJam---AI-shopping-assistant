"""5-fold cross-validation for the learned fusion weights (Task 5).

Replaces the single 70/30 holdout (which gave Train/val AUC 0.873/0.821, a noisy estimate
on 200 sessions) with k-fold CV, reporting the mean and spread of validation AUC across
folds. A large spread signals the learned weights may not generalise to the 800 hidden
sessions, in which case the hand-tuned fallback (`USE_LEARNED_FUSION = False`) is the safer
submission default.

Feature pipeline mirrors `_feature_vector`: [bm25, dense, slot, price] for every candidate
in the fused pool, labelled 1 for the target `parent_asin` and 0 otherwise. Features are
captured at every turn of a replayed session (the same dialogue the evaluator runs).
"""
import json
import os
import sys
import time
from collections import defaultdict

root = r"c:\Users\ImanKasni\OneDrive - Kuok (Singapore) Limited\Desktop\Work Documents\02 - Personal\07 - TT TechJam T4\techjam-conversational-search"
os.chdir(root)
sys.path.insert(0, root)

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

samples = load_jsonl("data/public_set.jsonl")
catalog_ids, categories, products = catalog_index("data/catalog.jsonl")
agent = Agent("data/catalog.jsonl")


def _feature_rows_for_session(sample) -> tuple[list[list[float]], list[int]]:
    """Replay a session, returning (feature rows, labels) for every pool candidate."""
    sid = "cv_" + sample["sample_id"]
    agent.reset(sid, sample["user_profile"])
    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = materialize_hidden_fields(sample, products)
    eff = {**sample, "intent_card": card, "behavior": behavior}
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    user_message = initial_message(eff, coarse_category(categories.get(target, [])), disclosed)
    X: list[list[float]] = []
    y: list[int] = []
    for turn in range(1, MAX_TURNS + 1):
        resp = agent.respond(sid, user_message, turn, TOP_K)
        features = getattr(agent, "_candidate_features", {}) or {}
        for asin, feat in features.items():
            if asin in agent.products:
                X.append([float(v) for v in feat])
                y.append(1 if asin == target else 0)
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
    return X, y


# Build per-session feature rows so folds hold out whole *sessions* (the unit we must
# generalise to: the 800 hidden sessions).
sessions_data = []  # (features, labels, scenario)
t0 = time.time()
for i, sample in enumerate(samples):
    X, y = _feature_rows_for_session(sample)
    sessions_data.append((X, y, sample["scenario_type"]))
    if (i + 1) % 40 == 0 or i + 1 == len(samples):
        rows = sum(len(x) for x, _, _ in sessions_data)
        pos = sum(sum(yy) for _, yy, _ in sessions_data)
        print(f"{i+1}/{len(samples)} sessions | rows={rows} | positives={pos} | elapsed={time.time()-t0:.0f}s", flush=True)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402


def _stack(records):
    """Stack a list of (features, labels, scenario) into X, y arrays."""
    xs = [np.asarray(x, dtype=np.float64) for x, _, _ in records]
    ys = [np.asarray(yy, dtype=np.int64) for _, yy, _ in records]
    X = np.vstack(xs) if xs else np.empty((0, 4))
    y = np.concatenate(ys) if ys else np.empty((0,), dtype=np.int64)
    return X, y


# Reference single 70/30 split by session (mirrors the earlier holdout).
train_idx, test_idx = train_test_split(
    range(len(sessions_data)), test_size=0.30, random_state=42,
    stratify=[sc for _, _, sc in sessions_data],
)
Xtr, ytr = _stack([sessions_data[i] for i in train_idx])
Xte, yte = _stack([sessions_data[i] for i in test_idx])
lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(Xtr, ytr)
single_auc = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])

# Session-level 5-fold CV, stratified by scenario so each fold keeps the mix.
strategies = [sc for _, _, sc in sessions_data]
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_aucs = []
fold_weights = []
for fold, (tr_idx, va_idx) in enumerate(kf.split(sessions_data, strategies)):
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(*_stack([sessions_data[i] for i in tr_idx]))
    Xva, yva = _stack([sessions_data[i] for i in va_idx])
    auc = roc_auc_score(yva, lr.predict_proba(Xva)[:, 1])
    fold_aucs.append(auc)
    fold_weights.append({
        "bm25": round(float(lr.coef_[0][0]), 4),
        "dense": round(float(lr.coef_[0][1]), 4),
        "slot": round(float(lr.coef_[0][2]), 4),
        "price": round(float(lr.coef_[0][3]), 4),
        "bias": round(float(lr.intercept_[0]), 4),
    })
    print(f"fold {fold}: AUC={auc:.4f} weights={fold_weights[-1]}", flush=True)

mean_auc = float(np.mean(fold_aucs))
std_auc = float(np.std(fold_aucs))
n_rows = sum(len(x) for x, _, _ in sessions_data)
n_pos = int(sum(sum(yy) for _, yy, _ in sessions_data))
summary = {
    "n_sessions": len(sessions_data),
    "n_rows": n_rows,
    "n_positives": n_pos,
    "single_70_30_val_auc": round(single_auc, 4),
    "kfold_5_session_level": {
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "min_auc": round(float(np.min(fold_aucs)), 4),
        "max_auc": round(float(np.max(fold_aucs)), 4),
        "spread": round(float(np.max(fold_aucs) - np.min(fold_aucs)), 4),
    },
    "mean_weights": {
        "bm25": round(sum(w["bm25"] for w in fold_weights) / 5, 4),
        "dense": round(sum(w["dense"] for w in fold_weights) / 5, 4),
        "slot": round(sum(w["slot"] for w in fold_weights) / 5, 4),
        "price": round(sum(w["price"] for w in fold_weights) / 5, 4),
        "bias": round(sum(w["bias"] for w in fold_weights) / 5, 4),
    },
    "per_fold_weights": fold_weights,
    "seconds": round(time.time() - t0, 1),
}
with open("validation_fusion_cv.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("VALIDATION_DONE", json.dumps(summary))
