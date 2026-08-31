# Track 4 Compliance Matrix

This matrix maps the official Track 4 brief (pages 23–25 of the supplied event PDF) to
the repository. It separates implemented technical requirements from submission items
that still require a human-provided link or name.

## Core pillars

| Official requirement | Repository evidence | Status |
|---|---|---|
| Dual-track Buying/Browsing routing | `_route_intent()` identifies exploration and concrete constraints. `_retrieve()` uses route-specific BM25/dense weights; Buying prioritizes verified slot matches, while Browsing adds anonymized profile context to semantic retrieval. | Implemented |
| Multi-route retrieval and semantic ranking | In-memory SQLite FTS5 BM25 and TF-IDF cosine retrieval feed reciprocal-rank fusion and a learned deterministic reranker. Sentence embeddings and a grounded listwise LLM are opt-in. | Implemented |
| Dynamic state and intent override | Per-session slots, durable free-text evidence, recent turns, shown products, question history, per-slot overwrite, and explicit full reset live in `starter/agent.py`. | Implemented |
| Proactive clarification | The deterministic ask/recommend policy asks answerability-aware, non-repeating attribute questions when confidence is low. Precision-first slates expand from one hero result toward Top 10 as evidence accumulates. | Implemented |
| Dynamic context programming | Long-term profile terms, short-term constraints, and a bounded prior candidate beam are stored separately and re-orchestrated every turn. Long exact evidence can recover products lost to query drift. | Implemented |

## Constraints and evaluation

| Requirement | Evidence / behavior | Status |
|---|---|---|
| Maximum 10 turns | Evaluator enforces 10; the agent forces recommendation by turn 9. | Implemented |
| Catalog is strictly read-only | The agent opens `data/catalog.jsonl` for reading and builds only in-memory indexes. It never modifies catalog rows. | Implemented |
| No invented product identifiers | Every recommendation comes from the retrieved catalog candidate set; optional LLM output is membership-validated. | Implemented |
| Static 50k catalog and isolated sessions | Indexes are built from the frozen catalog; dialogue state is keyed by evaluator-provided `session_id`. | Implemented |
| Heterogeneous routes, weights, truncation, state update | BM25 and dense channels have route weights, top-200 retrieval, a bounded 300-item fused pool, short recent history, and capped profile/session context. | Implemented |
| In-scope local scoring / optional prompt reranking | The default path is entirely offline. Optional embeddings and OpenAI-compatible chat-completions reranking are declared and gated by environment variables. | Implemented |
| Hit Rate@10, MRR, MTTC | `results.json` records all official public metrics and per-scenario metrics. | Implemented |
| Reproducibility and service disclosure | `README.md`, `IMPROVED_AGENT_README.md`, `requirements.txt`, and `docs/devpost_project_description.md` cover setup, commands, dependencies, optional services, cost, and limitations. | Implemented |

## Submission deliverables

| Required deliverable | Current repository state | Status |
|---|---|---|
| Public source repository with setup and limitations | Source and documentation are ready; repository visibility must be confirmed on the hosting service before submission. | Verify externally |
| Written Devpost description | A paste-ready draft is in `docs/devpost_project_description.md`. Replace bracketed team details before submission. | Draft ready |
| Working interface and demonstrated multi-turn session | `python demo.py` provides an interactive terminal demo; `docs/demo_script.md` provides a recording flow. | Implemented |
| Public YouTube demo video linked in submission | Recording/upload cannot be completed from this repository. Add the final URL to the Devpost draft. | Pending team action |
| Team contribution statement | A fill-in section exists in the Devpost draft. Actual names and responsibilities are not known. | Pending team input |
| Trademark/copyright hygiene | The project uses the challenge name descriptively and preserves the supplied dataset attribution in `DATA_ATTRIBUTION.md`. The team must check all video music, images, and logos before publishing. | Final human review required |

## Validated public result

The deterministic v2.13.0 run on all 200 released development sessions produces:

- Hit Rate@10: `0.995` (199/200)
- MRR: `0.976806`
- MTTC: `2.81`
- Efficiency: `0.819`
- Technical Score: `0.954342`
- Reported model tokens and API cost: `0`

Browsing, boundary, and intent-override Hit Rate@10 are `1.0`; Buying is `0.9875`. The
single miss (`public_0020`) is a retrieval-recall limit, not a ranking or policy one: the
target never enters the BM25/dense candidate pool, so no reranking can recover it.

These public development results do not guarantee private-set performance. As a proxy for
the private set, the same agent was evaluated on 400 sessions generated by the same
harness from catalog targets the public set never uses: Technical Score `0.933532`
(v2.12.1 scores `0.910664` on the identical held-out set).
