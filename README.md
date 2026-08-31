# Shopping Copilot — TikTok TechJam 2026, Track 4

A conversational e-commerce search agent that finds a shopper's hidden target product
within 10 turns by asking useful clarifying questions, not by matching keywords. Built
against a frozen 50,000-product Amazon catalog and the organizer's headless Agent API
(`reset` / `respond`) — evaluated purely on backend behavior, no UI.

On the 200-session public development set, the current agent (**v2.12.1**) reaches
Hit Rate@10 `1.0`, MRR `0.948458`, MTTC `3.315`, and Technical Score `0.938237`, entirely
deterministically — zero model calls, zero LLM tokens. The provided weak BM25 starter
scores Hit Rate@10 `0.125`, MRR `0.068034` for comparison. These are public-set results,
not a guarantee for the private evaluation set.

## Repository layout

```
techjam-conversational-search/
  starter/agent.py                the entire solution (only file edited)
  evaluator/local_evaluator.py    organizer's evaluator — untouched
  data/catalog.jsonl               frozen 50k-product catalog (included, no download step)
  data/public_set.jsonl            200 labeled development sessions
  demo.py                          interactive terminal demo
  demo-assets/app.py               Streamlit metrics dashboard (judging-time live demo)
  requirements.txt                 deps for tests, embeddings, and the dashboard
  tests/                           regression suite (96 tests)
  docs/TRACK4_COMPLIANCE.md        requirement-by-requirement audit
  docs/MRR_MTTC_RESEARCH.md        metric feasibility math + ablations
  README.md                        organizer's competition brief, updated with our own reproduced results
```

## Setup and installation

Requires Python 3.10+. The core agent runs on the standard library alone; the catalog
is already included in the repo, so no separate download step is needed.

```bash
git clone https://github.com/Panecord/Tiktok-TechJam---AI-shopping-assistant.git
cd Tiktok-TechJam---AI-shopping-assistant/techjam-conversational-search
pip install -r requirements.txt   # optional: only needed for tests, embeddings, and dev tooling
```

## Demo video

[PASTE THE PUBLIC YOUTUBE URL — pending recording, see docs/demo_script.md]

## Metrics dashboard

For the live judging demo, a Streamlit dashboard (`techjam-conversational-search/demo-assets/app.py`)
reads the same `results.json` the evaluator writes and is styled after
[`demo-assets/results_dashboard.html`](techjam-conversational-search/demo-assets/results_dashboard.html):
a dark "receipt"-themed hero with the headline metrics, a **Baseline → Final** before/after
comparison (the weak BM25 starter v1.0.0 vs. the current run), per-scenario cards, an
interactive Plotly breakdown, and a filterable session explorer. It also ships a built-in
**baseline → current version-history table** so the progression chart always shows the
improvement without an upload, and can trigger the evaluator run from the sidebar.

```bash
cd techjam-conversational-search
pip install -r requirements.txt   # includes streamlit, pandas, plotly
streamlit run demo-assets/app.py
```

In the sidebar you can:

- **Evaluator mode** — choose **Local (deterministic)** (`python -m evaluator.local_evaluator`
  → `results.json`) or **LLM (with model)** (`_validate_llm.py` → `validation_llm.json`, using
  the `COPILOT_LLM_*` env credentials).
- **▶ Run Evaluator** — runs the chosen evaluator from the repo root as a subprocess, streams
  its stdout into an expandable log, shows a **live progress bar with an ETA** (the LLM mode
  prints `N/M done … eta=…`, so the bar advances per session; the local mode shows an
  elapsed-time indicator), and auto-loads the fresh output on completion.
- **Load existing results** — upload a `results.json` / `validation_llm.json`, or point to a
  path, if you'd rather not re-run.
- **Version history** — upload/paste a CSV or JSON table (`version`, `description`,
  `hit_rate_at_10`, `mrr`, `mttc`, `efficiency`, `technical_score`, token usage) to plot
  the Technical Score progression with changelog tooltips.
- **Session filters** — scenario and hit/miss filters for spot-checking misses during Q&A.

Each headline metric card shows a hover **tooltip** explaining the metric, and the
**"📖 Metric definitions"** expander lists them all.

The **controls sidebar is collapsible** and starts collapsed to maximise chart space for a
live demo; reopen it with the top-left arrow. Each control group is icon-labelled and
independent collapsible (`⚙️ Run evaluator`, `📂 Load existing results`, `📈 Version
history`, `🔍 Session filters`), with a chevron showing its open/closed state. All evaluator
paths are resolved dynamically relative to the repo root, so the dashboard works from
anywhere (no hardcoded absolute paths).


The headline row uses `st.metric` deltas to compare the current run against the previous
version, and shows the Technical Score formula as a caption:

    Technical Score = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency

---

# Architecture, Version History & Full Implementation Notes

*(This section was previously a separate file, `IMPROVED_AGENT_README.md` — merged here
so the whole project is documented in one place.)*

> **Updated with AI.** This document describes the upgraded `starter/agent.py` that
> replaces the weak, stateless BM25 baseline (v1.0.0) shipped with the challenge.
>
> **v2.12.1 (Pareto refinement):** strengthens generic evidence, refines the browsing
> expansion schedule, and immediately restores Top 10 only after explicit slate rejection.
> It improves every public aggregate over v2.12.0: MRR `0.948458`, MTTC `3.315`,
> Efficiency `0.7685`, and Technical Score **`0.938237`**, with Hit Rate@10 `1.0` and zero
> model tokens.
>
> **v2.12.0 (rank-first precision):** adds catalog-derived constraint-source consistency,
> durable category context, and route/pivot-aware expanding slates. The agent starts with a
> small high-confidence hero slate, then expands toward Top 10 as evidence accumulates. On
> all 200 public sessions this produces **Hit Rate@10 `1.0`**, MRR **`0.939048`**, MTTC
> `3.325`, Efficiency `0.7675`, and Technical Score **`0.935214`**, with zero model tokens.
> See `docs/MRR_MTTC_RESEARCH.md` for feasibility math, ablations, and primary research.
>
> **v2.11.0 (restored recall layer):** applies the previously validated candidate-memory
> recovery design on top of the current v2.10 synonym-aware dual-route agent. A bounded
> historical beam is re-scored after long-evidence query drift, while an exact-substring
> evidence route restores catalog products dropped by tokenized BM25/TF-IDF retrieval.
> The two-to-one live/recall blend preserves ranking quality. Result: **Hit Rate@10 `1.0`
> (200/200)**, MRR `0.583518`, MTTC `2.70`, and Technical Score **`0.841055`**, with zero
> model tokens.
>
> **v2.10.0 (Iteration 7, dual-route + evidence):** the largest single jump yet — HR@10
> `0.715 → 0.995`, score `0.6095 → 0.8382`. Built on top of v2.9.0 (synonyms + re-fit
> weights retained). (a) **Dual-route intent routing** — `_route_intent` + `ROUTE_RRF_WEIGHTS`
> give Buying a BM25/constraint-precision bias and Browsing a dense/diversity bias. (b)
> **Free-text constraint evidence** — `_extract_constraint_evidence` preserves durable
> catalogly constraints ("what matters is: arch support") and `_evidence_match_score`
> (weight `EVIDENCE_BOOST_WEIGHT=3.0`) matches them against grounded catalog text, which is
> hugely discriminative because the simulator's answers are catalog-derived. (c)
> **Anonymized profile context** — `_profile_terms` distils `preference_tags` into an
> allow-listed vocabulary that biases browsing dense retrieval. (d) **Answerability-aware
> questions** — `ANSWERABILITY_PRIORITY` + profile-tag tie-breakers in `_choose_ask_attribute`.
> (e) **Scoped replies** — a `feature`/`other` answer no longer clobbers structured slots,
> keeping confirmed material/color. (f) **`_novel_slate`** avoids recommending already-shown
> ids. v2.9.0 synonyms (`_COLOR_SYNONYMS`/`_MATERIAL_SYNONYMS`), the re-fit
> `FUSION_WEIGHTS`, and `_bm25_query` de-dup are preserved.
>
> **v2.9.0 (Iteration 6, Tasks 3 / 7 / 8 / 9 / 11 / 12):** the strongest single-iteration
> gain so far. (a) **Static synonym-aware color/material matching** — the slot
> vocabularies only cover canonical words, so common catalog phrasings ("emerald",
> "wine", "elastane", "faux leather", …) were previously missed. Added `_COLOR_SYNONYMS` /
> `_MATERIAL_SYNONYMS`: extraction now checks synonyms *before* the base vocab (so
> "faux leather" → polyurethane, not leather; "rose gold" → pink, not gold), and
> `_slot_match_score` resolves a canonical value against a product that uses a synonym.
> (b) **Re-fitted learned fusion weights** via 5-fold, session-stratified CV
> (mean AUC `0.8054`, spread `0.057`): `bm25` becomes the dominant relevance signal
> (`2.61 → 4.24`) and `dense` is down-weighted (`11.34 → 0.71`), which lifted HR@10
> `0.650 → 0.715`, MRR `0.4787 → 0.4938`, MTTC `6.470 → 5.805`, score `0.5592 → 0.6095`.
> (c) **`_bm25_query` de-duplication** so a slot value already in the message is not
> emitted twice in the FTS expression. (d) **`public_0190` resolved** — the last
> dedicated residual miss now hits after the re-fit, and was promoted to a recovered
> regression case. (e) **Isolated unit tests** for the ask-vs-recommend policy,
> `_choose_ask_attribute`, `_bm25_query`, and synonym matching. (f) **Parallelized
> `_validate_llm.py`** (ThreadPoolExecutor, per-worker Agent, session slices) — code only.
> (g) **Pool-recall re-check**: pool recall `0.965` vs HR@10 `0.715` → the bottleneck is
> **ranking/selection, not retrieval**, so sentence-embedding retrieval (Task 12.2) was
> investigated and deferred (limited upside, heavy `torch` dependency).
>
> **v2.8.0 (Iteration 5):** **ask-attribute reachability** — `_choose_ask_attribute` now
> skips `budget` and `category` (the simulator's constraint classifier never labels a
> disclosure `budget` or `category`, so asking is a guaranteed dead end) and registers
> non-recognized materials (denim, linen, jewelry metals, …) under `feature` so the
> entropy-based selector can actually reach them (`EVALUATOR_RECOGNIZED_MATERIALS`). The
> LLM rerank prompt also gained `recent_turns` (last 3 user turns), matching the spec's
> rerank request shape.
>
> **v2.7.0 (Iteration 4):** expanded the slot vocabularies — `CATEGORY_TOKENS` gained the
> plural forms and data-driven product types found in the public-set audit (tees, bras,
> socks, jeans, slippers, loafers, …), and `MATERIALS` gained jewelry/accessory materials
> (alloy, gold, silver, gemstone, …) plus explicit **"Label: value"** constraint parsing
> (`"Material: alloy"`, `"Color: rose gold"`). Two performance fixes: the lowercased
> searchable text is now cached once per product (`_searchable_lc`), and dense TF-IDF
> scoring uses a **posting-list inverted index** instead of scanning all 50k docs each turn
> (~4x faster per turn). Null-price budget handling is now an explicit, documented
> deterministic choice.
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
| `v2.6.0` | **Per-slot pivot handling + diagnostics.** (a) A pivot now clears only the slot(s) the new message targets (full reset only via "forget all that"). (b) `_retrieve()` exposes the fused pool for a **pool-recall** diagnostic. (c) Session-level 5-fold CV for the learned fusion. (d) Clarifying-question quality diagnostic (entropy vs random). | Full-set **pool recall `0.960` vs HR@10 `0.545`** -> bottleneck is ranking, not retrieval. Full-set: HR@10 `0.545`, MRR `0.422623`, MTTC `8.255`, score `0.454187`. Intent-override HR `0.5333` (no regression). |
| `v2.6.1` | Dense retrieval now folds the accumulated slot values into its query (via `_dense_query`, token-de-duplicated) instead of scoring the raw message only — the dominant learned-fusion (dense) signal was previously blind to earlier-turn constraints. | HR@10 `0.545 -> 0.575`, MRR `0.4226 -> 0.4310`, MTTC `8.255 -> 8.20`, score `0.4542 -> 0.4728` |
| `v2.7.0` | **Slot-vocabulary expansion + performance.** (a) `CATEGORY_TOKENS` extended with plural forms and data-driven additions from the public-set audit (tees, bras, socks, jeans, slippers, loafers, …). (b) `MATERIALS` extended with jewelry/accessory materials, plus explicit `"Label: value"` constraint parsing. (c) Perf: cached lowercased searchable text (`_searchable_lc`) and a posting-list inverted index for dense scoring (~4x faster per turn). (d) Null-price budget behavior made an explicit, documented deterministic choice. | HR@10 `0.575 -> 0.630`, MRR `0.4310 -> 0.4556`, MTTC `8.20 -> 7.555`, score `0.4728 -> 0.5206` |
| `v2.8.0` | **Ask-attribute reachability + LLM context.** `_choose_ask_attribute` now skips `budget` and `category` (guaranteed dead-end asks: the simulator never discloses a constraint classed `budget`/`category`), and registers non-recognized materials (denim, linen, jewelry metals, …) under `feature` via `EVALUATOR_RECOGNIZED_MATERIALS` so the entropy selector can reach them. `recent_turns` (last 3 user turns) added to the LLM rerank prompt per spec §2. | HR@10 `0.630 -> 0.650`, MRR `0.4556 -> 0.4787`, MTTC `7.555 -> 6.470`, score `0.5206 -> 0.5592` |
| `v2.9.0` | **Synonym-aware matching + re-fitted fusion weights + test coverage.** (a) Static `_COLOR_SYNONYMS` / `_MATERIAL_SYNONYMS` checked before base vocab (extraction + slot-match). (b) Learned fusion weights re-fitted by 5-fold session CV (mean AUC `0.8054`): bm25 `2.61 -> 4.24`, dense `11.34 -> 0.71`. (c) `_bm25_query` de-duplication. (d) `public_0190` residual resolved. (e) Isolated unit tests for policy / `_choose_ask_attribute` / `_bm25_query` / synonyms. (f) `_validate_llm.py` parallelized (code only). (g) Pool-recall re-check: `0.965` vs HR@10 `0.715` (ranking bottleneck). | HR@10 `0.650 -> 0.715`, MRR `0.4787 -> 0.4938`, MTTC `6.470 -> 5.805`, Efficiency `0.453 -> 0.5195`, score `0.5592 -> 0.6095` |
| `v2.10.0` | **Dual-route intent + free-text evidence.** `_route_intent` + `ROUTE_RRF_WEIGHTS` (Buying = bm25/slot precision, Browsing = dense/profile diversity); `_extract_constraint_evidence` + `_evidence_match_score` with `EVIDENCE_BOOST_WEIGHT=3.0`; `_profile_terms` anonymized profile context; `ANSWERABILITY_PRIORITY` + profile tie-breakers in `_choose_ask_attribute`; scoped-reply slot protection; `_novel_slate`. v2.9.0 synonyms + re-fit weights retained. | HR@10 `0.715 -> 0.995`, MRR `0.4938 -> 0.58419`, MTTC `5.805 -> 2.73`, Efficiency `0.5195 -> 0.827`, score `0.6095 -> 0.8382` |
| `v2.11.0` | **Candidate-memory + exact-evidence recall.** Keeps a bounded prior beam, re-scores it with current evidence, clears it on pivots/resets, and adds a grounded exact-substring recall route with a two-live-to-one-recall blend. | HR@10 `0.995 -> 1.0` (200/200), MRR `0.583518`, MTTC `2.70`, Efficiency `0.83`, score `0.841055`; every scenario HR `1.0` |
| `v2.12.0` | **Constraint-source reranking + durable category + precision-first slates.** Reconstructs candidate constraint cards from catalog fields, keeps the original category active after long replies, and expands route/pivot-aware slates only as confidence grows. | HR@10 `1.0`, MRR `0.583518 -> 0.939048`, MTTC `2.70 -> 3.325`, Efficiency `0.7675`, score `0.841055 -> 0.935214`; zero tokens |
| `v2.12.1` | **Pareto refinement.** Evidence weight `5 -> 6`, Browsing expands `1,1,1,2,4,10`, and explicit rejection restores Top 10 immediately. | HR@10 `1.0`, MRR `0.939048 -> 0.948458`, MTTC `3.325 -> 3.315`, Efficiency `0.7675 -> 0.7685`, score `0.935214 -> 0.938237` |

> The next improvement will be `v2.13.0`.

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
  -> candidate-memory + exact-evidence recall routes
  -> constraint-source + durable-category scoring
  -> ask-vs-recommend policy (deterministic threshold rule)
  -> precision-first expanding slate
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
| 12 | Multi-turn recall recovery | none | bounded prior beam + grounded exact-evidence route |
| 13 | Rank-first slate control | fixed Top 10 | route/pivot-aware `1 -> 2 -> 5 -> 10` expansion |

## 2. Metrics — Baseline vs. Upgraded

Measured on the **200-session public dev set** via `python -m evaluator.local_evaluator`
(the evaluator and public labels are untouched).

| Metric | Baseline | v2.7.0 | v2.8.0 | v2.9.0 | v2.11.0 | Current v2.12.1 |
|--------|----------|--------|--------|--------|---------|-----------------|
| Hit Rate@10 | `0.125` | `0.630` | `0.650` | `0.715` | `1.0` | **`1.0`** |
| MRR | `0.068034` | `0.455567` | `0.478685` | `0.493812` | `0.583518` | **`0.948458`** |
| MTTC | `9.81` | `7.555` | `6.470` | `5.805` | `2.70` | **`3.315`** |
| Efficiency | `0.119` | `0.3445` | `0.453` | `0.5195` | `0.83` | **`0.7685`** |
| **Technical Score** | `0.10671` | `0.520570` | `0.559206` | `0.609544` | `0.841055` | **`0.938237`** |

Scenario breakdown (from `results.json`):

| Scenario | Baseline HR@10 | v2.11.0 | Current v2.12.1 | Current MRR | Current MTTC |
|----------|----------------|---------|-----------------|-------------|--------------|
| buying | `0.2375` | `1.0` | **`1.0`** | `0.974375` | `2.7375` |
| browsing | `0.025` | `1.0` | **`1.0`** | `0.930729` | `3.25` |
| intent_override | `0.1333` | `1.0` | **`1.0`** | `0.934444` | `4.633333` |
| boundary | `0.0` | `1.0` | **`1.0`** | `0.925` | `4.5` |

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
    "recent_turns": [...],          # bounded optional-LLM context
    "evidence": [...],              # durable arbitrary constraints
    "shown_ids": set(...),          # avoids repeated slates
    "profile_terms": [...],         # bounded anonymized preferences
    "candidate_memory": [...],      # bounded prior retrieval beam
    "candidate_memory_features": {},
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
- **Recall-memory reset:** intent pivots and explicit full resets also clear the historical
  candidate beam, preventing products from an obsolete intent entering the new slate.

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

For long catalog-specific evidence, v2.11 adds two grounded recovery paths:

1. **Candidate memory:** up to 900 earlier retrieved products retain their strongest
   lexical/dense features and are re-scored using current slots, price, and evidence.
2. **Exact-evidence recall:** the complete normalized phrase is matched against cached
   catalog text, preserving punctuation-heavy feature structure lost by tokenization.

The live+memory ranking is interleaved, then the exact route receives one position for every
two primary results. Every recovery candidate still comes from `self.products`.

Every ASIN in the returned pool is validated against `self.products` (grounding guarantee).

### 3.5 Slot-aware re-scoring (`_slot_match_score`)

Returns `0..1` telling how well a product matches the known constraints:

- material / color / size / style / use-case: does the product's searchable text contain
  the extracted value?
- category: does the product text contain any of the category tokens?
- budget: is the product's `price` within ±30% of the stated budget (when a price exists)?

This score is used both as a fusion boost and by the deterministic reranker.

### 3.6 Grounded rerank (§5)

`_rerank()` is **grounded**: deterministic and optional-LLM results are restricted to the
live pool, while the memory and exact-evidence routes are restricted to prior/catalog IDs.

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
| `FUSION_WEIGHTS` | `{bm25:4.2429, dense:0.7117, slot:2.6878, price:0.0, bias:-8.9584}` | five-fold mean weights fitted on the public development set |
| `MEMORY_POOL_CAP` | `900` | maximum grounded candidates retained across turns |
| `MEMORY_MIN_EVIDENCE_CHARS` | `40` | minimum evidence length that activates memory/exact recall |
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
- Budget matching deliberately gives no budget credit to null-price products (treated as
  out-of-budget); ~79% of the catalog has no price, so only verifiably in-budget products
  earn the budget signal.
- The deterministic reranker uses lexical attribute matching. v2.9.0 added a static
  color/material synonym table (`_COLOR_SYNONYMS`, `_MATERIAL_SYNONYMS`), so common
  phrasings like "emerald", "wine", "elastane", "faux leather" resolve to their canonical
  value; synonyms outside that table (and non color/material attributes) may still be missed.
- `usage` is `0` unless an LLM is configured; enabling the LLM requires a compatible
  chat-completion endpoint and does not improve the core metric unless it reorders candidates
  more accurately than the deterministic reranker.
- The public result is 200/200, but this is a development-set result, not a private-set
  guarantee. Indistinguishable groups larger than the ten-turn Top-10 budget remain an
  information limit when the shopper never discloses a unique title, saying, or brand.


## Team contributions

- Iman D: Set up the repo and built the core architecture from scratch — the
  system that tells apart someone who's just browsing from someone who knows
  what they want, and routes each differently. Also built the part that
  remembers what a shopper actually said earlier in the conversation and
  matches it against real catalog text, plus taught the matcher to understand
  that "faux leather" means the same thing as "polyurethane." Most of the
  early heavy lifting on retrieval quality came from this work.

- Rahul Sivakumar: Spent a while debugging why the agent kept asking
  unnecessary questions instead of just recommending — turned out it was
  checking the wrong number under the hood, so the "confident enough"
  threshold could basically never trigger. Fixed that, plus tuned how much
  weight a BM25 rank should carry so a huge gap between rank 1 and rank 30
  doesn't drown out other signals. Also caught a bug where certain material
  types weren't being picked up from what shoppers typed, and cleaned up a
  couple of spots where the agent would ask about things it could never
  actually use.

- Tan Ah Kow: Tackled the trickiest remaining failure mode — cases where the
  right product had already been seen earlier in the conversation but got
  lost once the shopper said something that changed the query. Built a memory
  layer that holds onto recently seen candidates and can pull them back when
  that happens. Followed up with a few rounds of scoring refinements — making
  sure the system's confidence in a product actually lines up with what the
  shopper confirmed — and a final tuning pass to squeeze out more ranking
  quality without breaking anything that already worked.

## Further documentation

- `techjam-conversational-search/docs/TRACK4_COMPLIANCE.md` — official Track 4
  requirement-by-requirement audit.
- `techjam-conversational-search/docs/v2.12.1_release_notes.md` — current release
  results and validation steps.
- `techjam-conversational-search/docs/MRR_MTTC_RESEARCH.md` — MRR/MTTC feasibility
  math and the theoretical Technical Score ceiling (`0.9922`).
- `techjam-conversational-search/docs/devpost_project_description.md` and
  `docs/demo_script.md` — submission draft and demo recording outline.
- `techjam-conversational-search/DATA_ATTRIBUTION.md` — Amazon Reviews 2023 dataset
  attribution and usage terms.