# Improved Agent — Track 4 Shopping Copilot (v2.9.0)

> **Updated with AI.** This document describes the upgraded `starter/agent.py` that
> replaces the weak, stateless BM25 baseline (v1.0.0) shipped with the challenge.
>
> **v2.9.0 (candidate-memory route):** preserves a bounded earlier retrieval beam and
> re-scores it when long catalog-specific evidence would otherwise cause query drift.
> Live and memory rankings are interleaved, already-shown products remain suppressed, and
> pivot/full-reset operations clear the memory. This closes the final public miss without
> using sample IDs or labels: **Hit Rate@10 `1.0` (200/200)**, MRR `0.572823`, MTTC `2.71`,
> and Technical Score **`0.837647`**, with zero model tokens.
>
> **v2.8.1 (Track 4 compliance pass):** makes the Buying and Browsing routes
> operationally distinct, correctly treats “still exploring” as Browsing even when a
> category is present, transitions to Buying when concrete evidence arrives, and distils
> anonymized preference tags into bounded browsing context. The optional LLM candidate
> cards now include category, features, and price rather than title alone. The deterministic
> result remains **Hit Rate@10 `0.995`**, while MRR improves to `0.574935`, MTTC to `2.74`,
> and Technical Score to **`0.835180`**, with zero model tokens.
>
> **v2.8.0 (Iteration 5):** made multi-turn evidence durable, rotates unseen grounded
> recommendations across turns, uses answerability-aware clarification questions, preserves
> independently confirmed evidence across per-slot pivots, expands category parsing beyond a
> fixed vocabulary, adds exact constraint-coverage reranking, scopes replies to the attribute
> asked, forwards recent turns to the optional LLM, and replaces stale fusion coefficients
> with the session-level five-fold values already recorded in `validation_fusion_cv.json`.
> The deterministic public-set result is **Hit Rate@10 `0.995`**, MRR `0.573546`, MTTC
> `2.745`, and Technical Score **`0.834664`**, with zero model tokens.
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
| `v2.8.0` | Durable free-text constraints, non-repeating multi-turn slates, answerability-aware questions, override-safe evidence, vocabulary-independent categories, exact evidence coverage, scoped reply updates, LLM history forwarding, and corrected five-fold fusion weights. | HR@10 `0.650 -> 0.995`, MRR `0.478685 -> 0.573546`, MTTC `6.47 -> 2.745`, score `0.559206 -> 0.834664`; boundary/browsing/intent-override HR `1.0`, buying HR `0.9875` |
| `v2.8.1` | Official Track 4 compliance pass: distinct precision/discovery route weights, correct exploration routing and state-driven route transitions, bounded profile-context distillation, and richer grounded optional-LLM candidate cards. Added an interactive demo and submission/compliance documents. | HR@10 `0.995`; MRR `0.574935`; MTTC `2.74`; score `0.835180`; browsing/boundary/intent-override HR `1.0`, buying HR `0.9875` |
| `v2.9.0` | Added bounded session candidate memory. Long catalog-specific evidence activates a separately re-scored historical beam, interleaved with live retrieval; intent pivots clear it. | HR@10 `0.995 -> 1.0` (200/200); MRR `0.572823`; MTTC `2.71`; score `0.837647`; every scenario HR `1.0` |

> The next improvement will be `v2.10.0`.

## 1. Summary of What Changed

The original starter was a **stateless BM25 wrapper**. On every turn it ran a BM25 query
over the user's message and returned the top `top_k` results, always with
`ask_attribute = null`. Because it never asked a clarifying question, it never narrowed the
candidate pool, so it could only succeed when the target already appeared in its initial
top-10. That produced the baseline Hit Rate@10 of `0.125`.

The upgraded agent implements the full pipeline from the Track 4 technical spec:

```
user turn
  -> dialogue update (structured slots + durable free-text evidence)
  -> per-slot pivot/full-reset handling
  -> intent router (Buying vs Browsing)
  -> hybrid retrieval (BM25 + TF-IDF or opt-in sentence embeddings)
  -> candidate fusion (RRF pool + five-fold learned scoring)
  -> grounded rerank (slot + exact-evidence coverage; optional LLM)
  -> long-evidence candidate-memory route (bounded prior beam + live interleave)
  -> ask-vs-recommend policy (deterministic threshold rule)
  -> unseen-product slate selection (avoid wasting later turns on repeats)
  -> turn-budget guard (force convergence by turn 9)
```

### High-level feature additions

| # | Feature | Baseline (v1.0.0) | Current (v2.9.0) |
|---|---------|-------------------|-------------------|
| 1 | Hybrid retrieval | BM25 only | BM25 + in-memory dense TF-IDF |
| 2 | Candidate fusion | none (raw BM25 order) | RRF recall pool + session-level five-fold weights |
| 3 | Dialog state | none | structured slots + durable free-text constraints + recent turns |
| 4 | Intent routing | none | Buying vs Browsing classification |
| 5 | Intent override handling | none | per-slot updates; only a full reset clears everything |
| 6 | Grounded rerank | none | learned relevance + slot + exact-evidence coverage |
| 7 | Ask-vs-recommend policy | always recommend | deterministic threshold rule |
| 8 | Turn-budget guard | none | force recommend by turn 9 |
| 9 | Attribute-driven questions | never asks | answerability first, entropy as supporting evidence |
| 10 | Optional LLM rerank | none | optional, env-var gated, validated |
| 11 | Honest token reporting | 0/0 | 0/0 when no LLM (real counts when used) |
| 12 | Cumulative slate coverage | repeats the same top results | prioritizes high-ranked unseen products each turn |
| 13 | Candidate memory | none | bounded historical beam re-scored after long-evidence query drift |

## 2. Metrics — Baseline vs. Upgraded

Measured on the **200-session public dev set** via `python -m evaluator.local_evaluator`
(the evaluator and public labels are untouched).

| Metric | Baseline (BM25) | v2.1.0 | v2.6.0 | v2.7.0 | Current v2.9.0 |
|--------|-----------------|--------|--------|--------|----------------|
| Hit Rate@10 | `0.125` | `0.515` | `0.545` | `0.630` | **`1.0`** |
| Successful sessions | `25/200` | `103/200` | `109/200` | `126/200` | **`200/200`** |
| MRR | `0.068034` | `0.349196` | `0.422623` | `0.455567` | **`0.572823`** |
| MTTC | `9.81` | `8.21` | `8.255` | `7.555` | **`2.71`** |
| Efficiency | `0.119` | `0.279` | `0.2745` | `0.3445` | **`0.829`** |
| **Technical Score** | `0.10671` | `0.418059` | `0.454187` | `0.520570` | **`0.837647`** |

Scenario breakdown (from `results.json`):

| Scenario | Samples | Baseline HR@10 | v2.7.0 | Current v2.9.0 | Current MRR | Current MTTC |
|----------|---------|----------------|--------|----------------|-------------|--------------|
| buying | `80` | `0.2375` | `0.6375` | **`1.0`** | `0.541295` | `2.0875` |
| browsing | `80` | `0.025` | `0.6125` | **`1.0`** | `0.544797` | `2.575` |
| intent_override | `30` | `0.1333` | `0.6` | **`1.0`** | `0.660913` | `4.266667` |
| boundary | `10` | `0.0` | `0.8` | **`1.0`** | `0.785` | `4.10` |

The current deterministic run uses no external model and reports zero tokens. These are
public development-set results rather than a guarantee about the organizer's private set.

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

- **Browsing** is selected by explicit exploration language such as “still exploring,”
  “just browsing,” “not sure,” or “show me ideas.” A broad category does not incorrectly
  turn that request into Buying.
- **Buying** is selected when a shopper gives a material, color, size, style, use-case,
  budget, durable free-text requirement, or concrete purchase language.
- A vague session remains Browsing until evidence arrives, then transitions to Buying.

Both routes keep lexical and semantic retrieval for recall, but their orchestration differs.
Buying gives BM25 and verified slot matches a precision bias. Browsing gives dense retrieval
a discovery bias and appends bounded anonymized profile terms to the dense query. The active
route and weights are stored in `_last_route` for demo/debug observability.

### 3.3 Dialogue state (slot tracking, spec §2)

Each session (keyed by `session_id`) carries persistent state in `self._sessions`:

```python
{
    "intent": "buying" | "browsing" | None,
    "slots": {"category", "material", "color", "size", "style", "budget", "price_max", "price_min", ...},
    "turn": int,
    "questions_asked": [...],       # attributes already asked (avoids repeats)
    "override_consumed": bool,
    "recent_turns": [...],          # last three messages for optional LLM grounding
    "evidence": [...],              # durable arbitrary feature/constraint text
    "shown_ids": set(...),          # cumulative recommendation coverage
    "user_profile": {...},          # anonymized soft preference metadata
    "profile_terms": [...],         # allow-listed long-term preference context
    "distilled_context": "...",    # bounded profile + current session terms
    "candidate_memory": [...],      # bounded earlier retrieval beam
    "candidate_memory_features": {},# strongest prior lexical/dense signals
}
```

- **Structured extraction** (`_extract_slots`) parses category, material, color, size,
  style, budget, use-case, and explicit `Label: value` fragments. Category phrases are
  no longer restricted to a fixed token list.
- **Durable evidence** (`_extract_constraint_evidence`) retains arbitrary requirements
  such as “nickel free” or “arch support” and folds them into subsequent retrieval turns.
- **Scoped answers:** a reply to a feature question cannot accidentally overwrite a
  confirmed material merely because it contains text such as “rubber sole.”
- **Per-slot pivot:** a pivot clears only explicitly replaced structured slots. The initial
  preference being overridden is removed, while independently confirmed later evidence is
  preserved. Product exposure and question history restart for the new intent.
- **Full reset:** strong phrases such as “forget all that” clear slots, evidence, questions,
  and the shown-product slate.
- **Context distillation:** only allow-listed `preference_tags` are expanded into long-term
  retrieval terms; review-rating metadata and free-form profile summaries are not treated as
  product requirements. Current slots/evidence are merged separately into a 32-term runtime
  context on each turn.
- **Candidate memory:** up to 900 earlier grounded candidates and their strongest lexical/
  dense features are retained. A pivot or full reset clears this route along with exposure
  state so old-intent products cannot leak into the new search.

### 3.4 Hybrid retrieval + fusion (§3)

`_retrieve()` produces a grounded candidate pool:

1. **BM25 query** (`_bm25_query`) is built from the user message **plus** any meaningful
   slot values (material, color, size, style, use-case, category) to improve recall.
2. **BM25 top-200** is fetched from the FTS index.
3. **Dense top-200** is fetched from the in-memory TF-IDF cosine similarity.
4. **Route-aware Reciprocal Rank Fusion** combines the two rankings. Buying uses
   BM25/dense weights `1.10/0.90`; Browsing uses `0.90/1.10`. The Buying pool places
   candidates with verified slot coverage first while retaining unmatched backfill for
   recall. Browsing includes bounded profile context only in its dense query.
5. The RRF pool is represented by `[BM25, dense, slot, price]` features and scored with
   the session-level five-fold mean logistic-regression coefficients.
6. Exact phrase/token coverage of durable free-text evidence is added at rerank time.
7. The result is capped to a bounded pool (default `FUSED_POOL = 300`).

When a newly disclosed catalog-specific constraint is at least 40 characters, the agent
also re-scores candidates that were plausible earlier but fell out of the live pool. Live
and memory rankings are interleaved before unseen-slate selection. This addresses a common
retrieval failure in which copied manufacturing boilerplate makes a later query less useful
than the earlier category query; every memory candidate still originated in grounded catalog
retrieval.

Every ASIN in the returned pool is validated against `self.products` (grounding guarantee).

### 3.5 Slot-aware re-scoring (`_slot_match_score`)

Returns `0..1` telling how well a product matches the known constraints:

- material / color / size / style / use-case: does the product's searchable text contain
  the extracted value?
- budget: is the product's `price` within ±30% of the stated budget (when a price exists)?

Category is intentionally excluded from this average because it already anchors BM25 and
dense retrieval; counting it again gave nearly every candidate equal credit and compressed
the more useful material/color/style signal.

### 3.6 Grounded rerank (§5)

`_rerank()` is **grounded** — it can only ever return ASINs that are already in the
candidate pool built by `_retrieve()`.

- **Default — deterministic:** `_rerank_deterministic()` orders the grounded pool by the
  corrected five-fold fusion score plus exact free-text evidence coverage. It never
  invents an ID.
- **Optional — LLM hook:** `_rerank_llm()` is invoked only when
  `COPILOT_LLM_URL` and `COPILOT_LLM_KEY` environment variables are set. It asks the model
  to return a JSON array of ranked ASINs, then **validates** that every returned ID is a
  member of the candidate set. It retries once only for a grounding violation; transport,
  HTTP, and timeout failures fall back immediately. Recent shopper turns are forwarded to
  the model. This enforces the "never free-generate a `parent_asin`" rule.

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
- **Ask attribute selection** (`_choose_ask_attribute`): prioritize attributes that the
  customer is likely able to answer (`material`, `feature`, `color`, `style`, `size`,
  `use_case`), then use candidate-pool entropy and profile tags as supporting signals.
  Budget/category are not actively asked because the released simulator cannot answer
  them informatively; unprompted values are still extracted.

### 3.8 Cumulative slate coverage

Every response remains grounded in the current reranked pool, but `_novel_slate()` prefers
products that have not already been displayed in the session. Repeats are used only when
needed to fill a small pool. On an intent pivot, exposure history resets so a product shown
under the old intent may be considered again. This uses the full ten-turn recommendation
budget instead of repeatedly returning an unchanged top 10.

### 3.9 Turn-budget guard (§6)

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
| `FUSION_WEIGHTS` | `{bm25:4.5882, dense:1.4963, slot:2.5234, price:0.0303, bias:-9.3415}` | session-level five-fold mean weights from `validation_fusion_cv.json` |
| `EVIDENCE_BOOST_WEIGHT` | `3.0` | exact/free-text constraint-coverage contribution to reranking |
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
already be a member of the trimmed pool. A grounding violation is retried once. A request,
HTTP, or timeout failure falls back immediately to deterministic ranking, avoiding stacked
latency. The prompt includes the last three shopper turns. Real token usage is parsed from
the model response and reported; if the endpoint does not report usage, a length-based
estimate is used.

When absent (the default), the agent runs fully deterministically with zero LLM tokens
(`usage = {prompt_tokens: 0, completion_tokens: 0}`), which is reported honestly.

## 6. Scope Compliance

To stay within the challenge rules, the implementation:

- Keeps the competition evaluator and public labels unchanged.
- Never modifies `evaluator/local_evaluator.py` or `data/public_set.jsonl`.
- Uses only the Python standard library in default mode (no external service, vector DB,
  or fine-tuning). Optional embeddings/LLM integrations remain explicitly gated.
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

# 4. With the optional development dependencies installed, run regressions.
python -m pytest -q

# 5. Run a live multi-turn terminal demo.
python demo.py --profile-tags comfort,fit,durability
```

The checked-in `results.json` contains the v2.9.0 deterministic public-set run. `pytest`
is a development dependency; the evaluator itself needs only Python 3.10+ standard library.
The compliance matrix, Devpost draft, and recording outline are in
`docs/TRACK4_COMPLIANCE.md`, `docs/devpost_project_description.md`, and
`docs/demo_script.md`.

## 8. Per-Scenario Behaviour Notes

- **Buying** — concrete requirements are accumulated, used in both retrieval channels, and
  reranked with exact evidence coverage. Long evidence may also activate the bounded memory
  route to prevent the earlier precision beam from being lost to query drift.
- **Browsing** — answerable clarification questions reveal constraints while unseen slates
  progressively explore the grounded pool.
- **Intent Override** — only replaced slots/initial preference evidence are invalidated;
  independent constraints remain, and question/exposure state restarts.
- **Boundary** — no-preference replies do not wipe state; the agent continues returning
  grounded unseen candidates and forces convergence at turn 9.

## 9. Limitations / Known Behaviours

- The public result is `200/200`, but this is a development-set measurement rather than a
  guarantee of private-set performance. Some novelty products remain information-theoretically
  indistinguishable from the disclosed constraints; candidate memory improves bounded
  exploration but cannot guarantee arbitrary hidden targets in groups larger than the total
  recommendation budget.
- Budget matching deliberately gives no budget credit to null-price products (treated as
  out-of-budget); ~79% of the catalog has no price, so only verifiably in-budget products
  earn the budget signal.
- The default TF-IDF path remains lexical; subtle synonyms may still benefit from the
  optional sentence-embedding mode, which must be validated separately before submission.
- `usage` is `0` unless an LLM is configured; enabling the LLM requires a compatible
  chat-completion endpoint and does not improve the core metric unless it reorders candidates
  more accurately than the deterministic reranker.
