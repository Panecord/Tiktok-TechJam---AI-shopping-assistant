# Improved Agent — Track 4 Shopping Copilot (v2.6.0)

> **Updated with AI.** This document describes the upgraded `starter/agent.py` that
> replaces the weak, stateless BM25 baseline (v1.0.0) shipped with the challenge.
>
> **v2.6.0 (Iteration 3, Tasks 1.5 / 2 / 5 / 6):** (a) **Per-slot pivot handling** — a
> detected pivot now clears only the attribute slot(s) the new message explicitly targets
> (e.g. "actually, blue not red" overwrites `color` only), preserving the rest of the
> dialogue state (TRADE-style independent slot updates); only a strong full-reset phrase
> ("forget all that") clears everything. (b) **Pool-recall instrumentation** — `_retrieve()`
> exposes `_last_candidates`, and a new diagnostic reports **pool recall** (target present
> anywhere in the fused pool) alongside HR@10 / MRR / MTTC. The full-set result is
> **pool recall `0.960` vs HR@10 `0.545`** — i.e. the target is retrieved ~96% of the time
> but only ranked top-10 ~54.5% of the time, so the bottleneck is **ranking/selection, not
> retrieval** (widening `BM25_TOP`/`DENSE_TOP` would add cost/noise without fixing it). (c)
> Session-level **5-fold CV** for the learned fusion replaces the single holdout (see
> `validation_fusion_cv.json`). (d) A **clarifying-question quality** diagnostic compares the
> entropy-based `_choose_ask_attribute` pool-size reduction against a random baseline.
>
> **v2.5.0 (Iteration 2 follow-up / LLM cost-latency fix):** the grounded **LLM listwise
> reranker** is now gated so it runs **only on the recommend branch** (via a `use_llm`
> parameter threaded through `_rerank`), instead of being called on every clarifying turn.
> In addition, `_rerank_llm` now retries **only on a grounding violation** (an id outside the
> candidate pool) and falls back to the deterministic reranker **immediately** on a
> transport/HTTP error or timeout, so rate-limit errors no longer stack retry latency. The
> per-call HTTP timeout was raised from 20s to 30s (observed latency is 4-6s, but can spike
> under load). This cut LLM calls from ~12/session to ~1.3/session (recommend turns only).
>
> **v2.4.0 (Iteration 2, Task 4):** replaced the hand-set RRF_K / slot-boost combination
> with **learned fusion weights** fitted by a small logistic regression (classical
> learning-to-rank, not LLM fine-tuning) on the public dev sessions, over features [BM25,
> dense, slot-match, price-distance]. The candidate pool is still formed by RRF
> (recall-preserving), and the learned score drives the final ordering. A 70/30 holdout gave
> train AUC `0.873` / val AUC `0.821` (no severe overfit). On a 40-session A/B the learned
> fusion nudged MRR `0.3719 -> 0.3746` and score `0.4176 -> 0.4179` with no Hit Rate change;
> the hand-tuned path is kept as a documented fallback (`USE_LEARNED_FUSION = False`).
>
> **v2.3.0 (Iteration 2, Task 3):** replaced the pure-Python TF-IDF dense retrieval with a
> pretrained **sentence-embedding** model (Sentence-BERT, inference only, no fine-tuning).
> The catalog is embedded once at startup; queries are embedded per turn and scored by
> cosine similarity. Embeddings are opt-in (`COPILOT_DENSE=embed`); otherwise the fast TF-IDF
> path is used so the default run never attempts a model download.
>
> **v2.2.0 (Iteration 2, Task 2):** wired up the grounded **LLM listwise reranker**. The
> candidate pool is trimmed upstream to `LLM_TOP = 25` by the deterministic fusion, then a
> single zero-shot listwise call asks the model for the ranked order of candidate ids. Every
> id is validated against the candidate pool (retry once, then fall back to the deterministic
> reranker). It is **gated behind environment credentials and off by default**; default
> (deterministic) metrics are unchanged because an A/B against a real model requires
> self-managed credentials.
>
> **v2.1.0 (Iteration 2):** fixed the buying-scenario regression and closed the MRR gap by
> (a) making the grounded rerank a **relevance-dominated** score with a slot-aware boost
> instead of re-ranking by slot-match alone, and (b) only treating **strong pivot language**
> as an intent override so benign replies like "I don't have a preference for X" no longer
> wipe the dialogue state.

## 0. Versioning & Changelog

**Convention:** every improvement to the agent bumps the version and is recorded below.

- **Major version** (e.g. `1.0.0 -> 2.0.0`) = a redesign of the pipeline.
- **Minor version** (e.g. `1.1 -> 1.2`, `2.1 -> 2.2`) = an iterative improvement or bug fix.

| Version | Notes | Key metric change |
|---------|-------|-------------------|
| `v1.0.0` | Original weak, stateless BM25 starter (the provided baseline). | Hit Rate@10 `0.125`, MRR `0.068034`, score `0.10671` |
| `v2.0.0` | First upgrade: hybrid retrieval (BM25 + dense), dialogue state tracking, grounded rerank, ask-vs-recommend policy, turn-budget guard. | Hit Rate@10 `0.125 -> 0.225`, MRR `0.068034 -> 0.068581`, score `0.10671 -> 0.167074` |
| `v2.1.0` | Fixed the buying regression and closed the MRR gap. The grounded rerank is now **relevance-dominated** with a slot-aware boost (instead of re-ranking by slot-match alone); the intent-override detector now triggers only on **strong pivot language**, so benign "I don't have a preference for X" replies no longer wipe the dialogue state. | Hit Rate@10 `0.225 -> 0.515`, MRR `0.068581 -> 0.349196`, buying HR `0.225 -> 0.5`, score `0.167074 -> 0.418059` |
| `v2.2.0` | Added a grounded **LLM listwise reranker** (enabled via env vars, off by default). The pool is trimmed to `LLM_TOP = 25` upstream, a single zero-shot listwise call reorders it, and ids are validated against the candidate pool (retry once, then deterministic fallback). Real token usage is reported. Not A/B-validated here (needs self-managed credentials); default metrics are unchanged while disabled. | No change in default (deterministic) mode |
| `v2.3.0` | Replaced the pure-Python TF-IDF dense retrieval with a **sentence-embedding** model (Sentence-BERT, inference only, no fine-tuning). Catalog embedded once at startup, queries embedded per turn, cosine similarity in-memory. Embeddings are opt-in (`COPILOT_DENSE=embed`); otherwise the fast TF-IDF path is used. The browsing-session A/B test is included but skipped until embeddings are enabled. | No change in default (TF-IDF) mode |
| `v2.4.0` | Replaced the hand-set RRF_K / slot-boost combination with **learned fusion weights** (small logistic regression on the public dev set, features [bm25, dense, slot, price]). RRF still forms the pool (recall-preserving); the learned score drives the final ordering. Train/val AUC `0.873 / 0.821`; 40-session A/B showed no HR change and a small MRR/score gain. Hand-tuned path kept as a documented fallback. | 40-session A/B: MRR `0.3719 -> 0.3746`, score `0.4176 -> 0.4179` (no HR change); full-set hand-tuned equals v2.2.0 (`0.515` / `0.3492` / `0.4181`) |
| `v2.5.0` | **LLM reranker cost/latency fix.** `_rerank` now takes a `use_llm` flag: the LLM listwise call runs only on the **recommend branch**, not every clarifying turn. `_rerank_llm` retries only on a **grounding violation** and falls back to deterministic immediately on a transport/HTTP error or timeout (no retry stacking under rate limiting). Per-call HTTP timeout raised 20s -> 30s (observed latency 4-6s, can spike). LLM calls cut from ~12/session to ~1.3/session. | No change in default (deterministic) mode; full-set LLM validation now bounded (~1.3 LLM calls/session) |
| `v2.6.0` | **Per-slot pivot handling + diagnostics.** (a) A pivot now clears only the slot(s) the new message targets (full reset only via "forget all that"). (b) `_retrieve()` exposes the fused pool for a **pool-recall** diagnostic. (c) Session-level 5-fold CV for the learned fusion. (d) Clarifying-question quality diagnostic (entropy vs random). | Full-set **pool recall `0.960` vs HR@10 `0.545`** -> bottleneck is ranking, not retrieval. Intent-override HR unchanged at `0.533` (no regression). |

> The next improvement will be `v2.7.0`.

## 1. Summary of What Changed

The original starter was a **stateless BM25 wrapper**. On every turn it ran a BM25 query
over the user's message and returned the top `top_k` results, always with
`ask_attribute = null`. Because it never asked a clarifying question, it never narrowed the
candidate pool, so it could only succeed when the target already appeared in its initial
top-10. That produced the baseline Hit Rate@10 of `0.125`.

The upgraded agent implements the full pipeline from the Track 4 technical spec:

```
user turn
  -> intent router (Buying vs Browsing)
  -> hybrid retrieval (BM25 + category/attribute filter + in-memory dense TF-IDF)
  -> candidate fusion (reciprocal-rank fusion + slot-aware re-scoring)
  -> dialogue state update (slot extraction + overwrite-on-override)
  -> grounded rerank (select-from-candidates-only; optional LLM hook)
  -> ask-vs-recommend policy (deterministic threshold rule)
  -> turn-budget guard (force convergence by turn 9)
```

### High-level feature additions

| # | Feature | Baseline (v1.0.0) | Upgraded (v2.0.0) |
|---|---------|-------------------|-------------------|
| 1 | Hybrid retrieval | BM25 only | BM25 + in-memory dense TF-IDF |
| 2 | Candidate fusion | none (raw BM25 order) | Reciprocal Rank Fusion (RRF) |
| 3 | Dialog state | none | per-session slot state (category/material/color/size/style/budget/use_case) |
| 4 | Intent routing | none | Buying vs Browsing classification |
| 5 | Intent over-ride handling | none | overwrite-on-pivot slot clearing |
| 6 | Grounded rerank | none | deterministic slot-aware rerank (select-from-candidates-only) |
| 7 | Ask-vs-recommend policy | always recommend | deterministic threshold rule |
| 8 | Turn-budget guard | none | force recommend by turn 9 |
| 9 | Attribute-driven questions | never asks | asks the attribute that best splits the pool |
| 10 | Optional LLM rerank | none | optional, env-var gated, validated |
| 11 | Honest token reporting | 0/0 | 0/0 when no LLM (real counts when used) |

## 2. Metrics — Baseline vs. Upgraded

Measured on the **200-session public dev set** via `python -m evaluator.local_evaluator`
(the evaluator and public labels are untouched).

| Metric | Baseline (BM25) | Upgraded (v2.0.0) | Upgraded (v2.1.0) |
|--------|-----------------|-------------------|-------------------|
| Hit Rate@10 | `0.125` | `0.225` | **`0.515`** |
| MRR | `0.068034` | `0.068581` | **`0.349196`** |
| MTTC | `9.81` | `9.30` | **`8.21`** |
| Efficiency | `0.119` | `0.17` | **`0.279`** |
| **Technical Score** | `0.10671` | `0.167074` | **`0.418059`** |

Scenario breakdown (from `results.json`):

| Scenario | Baseline HR@10 | Upgraded (v2.0.0) | Upgraded (v2.1.0) |
|----------|----------------|-------------------|-------------------|
| buying | `0.2375` | `0.225` | `0.5` |
| browsing | `0.025` | `0.2625` | `0.525` |
| intent_override | `0.1333` | `0.1667` | `0.4667` |
| boundary | `0.0` | `0.1` | `0.7` |

The largest gains come from the **browsing** and **intent_override** scenarios, which were
near-zero for the baseline because it never asked questions and never handled pivots.

## 3. Architecture Walkthrough

### 3.1 Catalog and index construction (`__init__`)

Three artifacts are built once at startup from `data/catalog.jsonl`:

1. **`self.products`** — `parent_asin -> product dict` (the frozen catalog held in memory).
2. **BM25 index** — an in-memory SQLite FTS5 virtual table over
   `title, categories, features, details, store, description`.
3. **Dense index** — a pretrained **sentence-embedding** model (Sentence-BERT, inference
   only, no fine-tuning) when the libraries are present, otherwise the pure-Python TF-IDF
   sparse matrix:
   - Sentence embeddings: catalog embedded once at startup, query embedded per turn,
     cosine similarity over an in-memory matrix.
   - TF-IDF fallback: product text tokenised and counted; vocabulary built from terms
     with document frequency `2..=60%` of the catalog, capped at the top `60,000`;
     sparse vectors + L2 norms stored, scored by cosine similarity.
   - No external vector DB (spec §3).

> Design note: no vector database, no fine-tuned model. Everything runs in-process on the
> 50k-product catalog, which fits comfortably in memory (spec §3).

### 3.2 Intent router (Buying vs Browsing)

`respond()` classifies each turn as `buying` or `browsing`:

- **Buying** is signalled by concrete language (`need`, `want`, `looking for`, `buy`,
  `require`, `must`, or any extracted attribute slot). These turns lean on the
  slot/BM25 side of the fusion.
- **Browsing** is the exploration case. The agent still retrieves densely and asks a
  question, but is less likely to force a narrow filter.

The intent is stored on the session so the routing is stable across turns, not just per message.

### 3.3 Dialogue state (slot tracking, spec §2)

Each session (keyed by `session_id`) carries persistent state in `self._sessions`:

```python
{
    "intent": "buying" | "browsing" | None,
    "slots": {"category", "material", "color", "size", "style", "budget", "price_max", "price_min", ...},
    "turn": int,
    "questions_asked": [...],       # attributes already asked (avoids repeats)
    "override_consumed": bool,
}
```

- **Slot extraction** (`_extract_slots`) parses the latest user message for category,
  material, color, size, style, budget, and use-case values.
- **Update semantics:** new information merges into existing slots (incremental).
- **Overwrite-on-pivot:** if the message contains override/pivot language (`actually`,
  `ignore`, `forget`, `instead`, `wait`, `never mind`, `changed my mind`, `scratch that`,
  ...), the previously-set attribute slots are cleared **before** the new values are
  written. This is the special handling for the **Intent Override** scenario and prevents
  a pivot from being merged as if it were additive.

### 3.4 Hybrid retrieval + fusion (§3)

`_retrieve()` produces a grounded candidate pool:

1. **BM25 query** (`_bm25_query`) is built from the user message **plus** any meaningful
   slot values (material, color, size, style, use-case, category) to improve recall.
2. **BM25 top-200** is fetched from the FTS index.
3. **Dense top-200** is fetched from the in-memory TF-IDF cosine similarity.
4. **Reciprocal Rank Fusion** combines the two rankings:
   `score(asin) = Σ 1 / (RRF_K + rank + 1)`.
5. **Slot-aware boost** is added to each candidate based on how well it satisfies the
   current slot constraints (`_slot_match_score`).
6. The result is capped to a bounded pool (default `FUSED_POOL = 300`).

Every ASIN in the returned pool is validated against `self.products` (grounding guarantee).

### 3.5 Slot-aware re-scoring (`_slot_match_score`)

Returns `0..1` telling how well a product matches the known constraints:

- material / color / size / style / use-case: does the product's searchable text contain
  the extracted value?
- category: does the product text contain any of the category tokens?
- budget: is the product's `price` within ±30% of the stated budget (when a price exists)?

This score is used both as a fusion boost and by the deterministic reranker.

### 3.6 Grounded rerank (§5)

`_rerank()` is **grounded** — it can only ever return ASINs that are already in the
candidate pool built by `_retrieve()`.

- **Default — deterministic:** `_rerank_deterministic()` re-orders the candidate pool by
  descending slot-match score. It never invents an ID.
- **Optional — LLM hook:** `_rerank_llm()` is invoked only when
  `COPILOT_LLM_URL` and `COPILOT_LLM_KEY` environment variables are set. It asks the model
  to return a JSON array of ranked ASINs, then **validates** that every returned ID is a
  member of the candidate set (retrying/falling back to the deterministic ranking on any
  invalid result). This enforces the "never free-generate a `parent_asin`" rule.

### 3.7 Ask-vs-recommend policy (§4, deterministic, not learned)

After reranking, `respond()` decides between **asking** and **recommending** using a fixed
threshold rule:

```
if turn >= 9:
    recommend(top candidate)          # turn-budget guard
elif pool_size <= K_SMALL and margin >= MARGIN_THRESHOLD:
    recommend(top candidate)          # confident
else:
    ask(attribute with highest discriminating power)
```

- `K_SMALL = 25` and `MARGIN_THRESHOLD = 0.20` are tunable constants
  (`FORCE_RECOMMEND_TURN = 9`).
- **Ask attribute selection** (`_choose_ask_attribute`): among the candidate pool, compute
  the entropy (discriminating power) of each attribute's value distribution, and ask about
  the attribute that best splits the pool, skipping attributes already asked.

### 3.8 Turn-budget guard (§6)

The evaluator ends any session at turn 10. To avoid falling off the cliff, the policy
**forces a recommendation at turn 9** regardless of confidence. The session state tracks
the current turn, so an unambiguous session never exceeds the cap.

## 4. Key Constants (and how to tune them)

All constants live at module top-level in `starter/agent.py` and are tuned against
`data/public_set.jsonl` only.

| Constant | Default | Meaning |
|----------|---------|---------|
| `BM25_TOP` | `200` | BM25 candidates retrieved per turn |
| `DENSE_TOP` | `200` | dense candidates retrieved per turn |
| `FUSED_POOL` | `300` | final candidate pool size fed to the policy |
| `RRF_K` | `60.0` | RRF smoothing constant |
| `K_SMALL` | `25` | pool size below which we consider recommending |
| `MARGIN_THRESHOLD` | `0.20` | relative margin between top-2 scores for "confident" |
| `FORCE_RECOMMEND_TURN` | `9` | turn at which we force a recommendation |
| `USE_LEARNED_FUSION` | `True` | use the learned logistic-regression fusion weights (False = hand-tuned fallback) |
| `FUSION_WEIGHTS` | `{bm25:2.61, dense:11.34, slot:2.04, price:0.0, bias:-4.65}` | learned fusion weights (fitted on the public dev set) |
| `LLM_TOP` | `25` | candidate pool size passed to the LLM reranker (trimmed upstream) |
| `DEFAULT_EMBED_MODEL` | `all-MiniLM-L6-v2` | pretrained sentence-embedding model used for dense retrieval (via `COPILOT_EMBED_MODEL`) |

## 5. Optional LLM Reranker (self-managed credentials)

To enable the **single-pass LLM listwise reranker**, set these environment variables
(never commit keys):

```
COPILOT_LLM_URL   https://.../v1/chat/completions
COPILOT_LLM_KEY   your-secret-key
COPILOT_LLM_MODEL shopping-copilot-rerank   (optional)
```

When enabled, the candidate pool is first trimmed to `LLM_TOP = 25` by the deterministic
fusion (so a single listwise call covers the whole pool — no sliding window). The model is
asked to return the ranked order of candidate ids, and the result is grounded: every id must
already be a member of the trimmed pool. On a grounding violation or a request error it
retries once, then falls back to the deterministic reranker. Real token usage is parsed from
the model response and reported; if the endpoint does not report usage, a length-based
estimate is used.

When absent (the default), the agent runs fully deterministically with zero LLM tokens
(`usage = {prompt_tokens: 0, completion_tokens: 0}`), which is reported honestly.

## 6. Scope Compliance

To stay within the challenge rules, the implementation:

- Only edits `starter/agent.py`.
- Never modifies `evaluator/local_evaluator.py` or `data/public_set.jsonl`.
- Uses only the Python standard library (no external service, no vector DB, no fine-tuning).
- Never returns a `parent_asin` outside the frozen catalog.
- Always uses the fixed `ask_attribute` enum.
- Reports real token usage (0 when no model is invoked).

## 7. How to Run / Reproduce

From the repo root (`techjam-conversational-search/`):

```bash
# 1. Ensure data/catalog.jsonl exists (per the challenge README).
# 2. Run the evaluator (writes results.json).
python -m evaluator.local_evaluator

# 3. Compare to the baseline in docs/baseline_results.json.
```

## 8. Per-Scenario Behaviour Notes

- **Buying** — message usually contains a concrete requirement; slot extraction captures it
  and retrieval is biased toward slot/BM25 matching.
- **Browsing** — message is exploration; the agent asks a discriminating question and leans
  on dense retrieval.
- **Intent Override** — the pivot is detected, prior attribute slots are cleared, and the
  new preference is written in fresh (per spec §2).
- **Boundary** — the agent keeps asking until the turn-budget guard forces a recommendation
  at turn 9, so it still emits a best-effort list instead of only asking.

## 9. Limitations / Known Behaviours

- Category detection relies on a fixed token list; rare category phrasing may not be captured.
- Budget matching only applies when a product has a non-null `price`.
- The deterministic reranker uses lexical attribute matching, so subtle synonyms may be missed.
- `usage` is `0` unless an LLM is configured; enabling the LLM requires a compatible
  chat-completion endpoint and does not improve the core metric unless it reorders candidates
  more accurately than the deterministic reranker.
