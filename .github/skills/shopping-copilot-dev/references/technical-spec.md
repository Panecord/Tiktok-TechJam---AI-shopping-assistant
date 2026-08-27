# Track 4 — Shopping Copilot: Technical Spec (fixed; do not redesign)

## 1. Pipeline overview (single forward pass per turn — no lookahead/tree search)
```
user turn
  → intent router (Buying vs Browsing)
  → hybrid retrieval (BM25 + category filter + in-memory dense similarity)
  → merge candidates (score fusion / reciprocal rank fusion)
  → dialogue state update (slot extraction + overwrite-on-override)
  → grounded LLM rerank (select id from candidates only, structured output)
  → ask-vs-recommend policy (deterministic threshold)
  → emit: clarifying question OR recommendation
  → turn budget check (force convergence by turn 9)
```

## 2. Core data model

**Candidate** (from retrieval): `{ id, title, category, attrs: {...}, bm25_score, dense_score, fused_score }`

**Slot state** (persistent per session): `{ category, price_max, price_min, attributes: {k: v, ...}, last_updated_turn }`
- New info merges into existing slots (incremental).
- Explicit contradiction/pivot language clears the conflicting slot(s) before writing new
  ones — do not merge a pivot as if it were additive.

**Rerank request** (to the LLM): `{ candidate_ids: [...], slot_state, recent_turns: [...] }`
**Rerank response** (from the LLM, structured/JSON-schema-enforced): `{ ranked_ids: [...] }`
  where every id in `ranked_ids` MUST be a member of `candidate_ids`. Validate before use.

**Turn record** (logged every turn): `{ turn_number, intent, candidate_pool_size, action: "ask"|"recommend", target_attribute_asked (if ask), recommended_id (if recommend) }`

## 3. Retrieval (deterministic, testable in isolation before reranking is built)
- BM25 over product title/description text (in-memory inverted index).
- Category/attribute filter for hard constraints already in slot state.
- Dense similarity: embed catalog once at startup, embed the query per turn, cosine similarity
  over an in-memory array (no external vector DB — 50k products fits trivially in memory).
- Fuse the three signals into one ranked candidate list (weighted sum or RRF). Buying-intent
  turns weight the filter/BM25 side higher; Browsing-intent turns weight dense similarity higher.

## 4. Ask-vs-recommend policy (fixed rule, not learned)
```
if candidate_pool_size <= K_small AND top_score_margin >= MARGIN_THRESHOLD:
    action = recommend(top candidate)
elif turn_number >= 9:
    action = recommend(top candidate)   # force convergence before the turn-budget cliff
else:
    action = ask(attribute with highest discriminating power over remaining pool)
```
K_small, MARGIN_THRESHOLD are tunable constants — tune them empirically against the 200
public dev sessions, but the rule structure itself doesn't change.

## 5. Grounding requirement (non-negotiable)
The LLM never sees or produces a product outside `candidate_ids`. Enforce via:
- Structured output / function-calling schema that only accepts ids from the passed-in list.
- A post-call validator that rejects any id not in the candidate set (retry once, then fall
  back deterministically to the top fused-score candidate).

## 6. Turn budget
- Max 10 turns per session; exceeding it is a hard zero.
- Explicit turn counter, checked before deciding to ask vs. recommend.
- By turn 9, force a recommendation regardless of confidence.

## 7. Explicitly out of scope (do not build)
- RL-trained ask/recommend policy or any trained classifier for this decision.
- External vector database or hosted DB cluster.
- Fine-tuning of any base model.
- Any UI beyond what's needed to run the headless evaluator.
- Catalog mutation or synthetic product injection.
- Multi-user concurrency handling (sessions are single-user by assumption).

## 8. Definition of done
- Runs end-to-end against the official local evaluator on the 200 public dev sessions.
- No session in a test run exceeds 10 turns.
- No recommendation or rerank output ever references an id outside the retrieved candidate set.
- Hit Rate@10 / MRR / MTTC recorded and compared against the provided BM25 baseline after
  every pipeline stage is added (see day-by-day task list).