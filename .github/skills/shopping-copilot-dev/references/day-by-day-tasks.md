# Track 4 — Day-by-Day Tasks (2 builders + 1 director)

Feed the coding assistant ONE numbered task at a time, not the whole file. The director's job
is to hand out tasks, review output against the spec, and run the evaluator — not to write
code, but to keep both builders from drifting off their lane.

Two independent lanes so the two builders aren't blocked on each other:
- **Builder A**: retrieval + dialogue state (spec §2 slot state, §3)
- **Builder B**: reranking + policy + turn guard (spec §4, §5, §6)
They integrate at the end of Day 1 and again at the end of Day 2.

## Day 0 (setup — director can do this alone before builders start)
0. Clone the participant repo and pull the participant-kit release (catalog + 200 dev
   sessions + evaluator + weak BM25 starter agent). Verify the catalog against the provided
   SHA256 checksum before using it. Run the starter agent through the local evaluator
   unmodified and record the baseline Hit Rate@10 / MRR / MTTC. This is the number everything
   else is measured against.

## Day 1 — build both lanes in parallel

**Builder A**
1. Stand up the in-memory BM25 index over the catalog (title/description).
2. Add the in-memory dense retrieval: embed the catalog once at startup, embed each turn's
   query, cosine similarity over a plain array. Confirm this alone improves Hit Rate@10 over
   BM25-only when tested against the dev sessions.
3. Implement fusion (weighted sum or RRF) combining BM25 + category filter + dense score into
   one ranked candidate list.
4. Implement the slot-state object and the intent router (Buying vs Browsing) that shifts
   fusion weights. Implement overwrite-on-pivot logic for intent-override language. Unit test
   the override case explicitly (a scripted "actually, forget that" turn).

**Builder B**
5. Implement the turn counter and the hard turn-budget guard (force recommend by turn 9) as
   a standalone unit, testable without retrieval or reranking wired up yet.
6. Define the structured-output schema for the rerank call (candidate ids in, ranked ids out)
   and the post-call validator that rejects any id outside the candidate set. Test the reject
   → retry → deterministic-fallback path explicitly, with a forced bad-output test case.
7. Implement the ask-vs-recommend threshold rule (spec §4) as a pure function taking pool size
   and score margin, independent of the rest of the pipeline. Unit test the three branches
   (recommend-confident, ask, force-recommend-at-turn-9).

## Day 1 end — integration checkpoint
8. Wire Builder A's fused candidate list into Builder B's rerank call and policy function.
   Run the full pipeline against a handful of dev sessions manually and sanity-check.

## Day 2 — full pipeline, tuning, evidence
9. Run the complete pipeline against all 200 public dev sessions through the official
   evaluator. Record Hit Rate@10 / MRR / MTTC.
10. Tune the fixed constants (K_small, MARGIN_THRESHOLD, fusion weights) against the dev
    sessions only — never against anything resembling the hidden set. Re-run after each
    tuning pass; keep a log of what changed and the resulting metric shift.
11. Add the discriminating-attribute selection for the "ask" branch (pick whichever attribute
    best splits the remaining candidate pool) if not already done in task 4/7.
12. Stress-test the turn-budget guard and the grounding validator directly (not just via full
    sessions) — force edge cases: an ambiguous session that would naturally run long, and a
    rerank call that returns garbage.

## Day 2 end / Day 3 morning — polish and package
13. Error handling and cleanup pass on the integration boundary only — not a rewrite of
    working retrieval or policy code.
14. Write the README (setup, how to reproduce, limitations) and the Devpost project
    description (tools, APIs, libraries, datasets used).
15. Record the demo video: solution overview, a full session walkthrough (ask + recommend),
    and the metrics table (baseline vs. final) from task 9/10.

## Stop conditions
- If either builder proposes work outside their lane before the Day 1 integration
  checkpoint, redirect them back to their numbered task.
- If a tuning pass in task 10 doesn't clearly improve a metric, revert it — don't stack
  unverified changes.