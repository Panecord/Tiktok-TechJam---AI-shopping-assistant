# Devpost Project Description Draft

Replace every bracketed field before submission.

## Project name

Shopping Copilot — Track 4 Conversational E-Commerce Search

## Inspiration and problem

Product search often fails when a shopper starts with a vague category, discovers what
matters through conversation, or changes their mind. Our Shopping Copilot turns a static
50,000-product catalog into a grounded multi-turn assistant. It asks useful clarification
questions, remembers confirmed constraints, handles intent pivots, and ranks only products
that actually exist in the supplied catalog.

## What it does

The agent maintains separate short-term dialogue evidence and anonymized long-term
preference tags. A router selects one of two retrieval modes: a precision-oriented Buying
route for concrete requirements and a discovery-oriented Browsing route for exploration.
Both combine lexical BM25 and in-memory semantic retrieval. Reciprocal-rank fusion forms a
bounded candidate pool; a learned deterministic scorer then combines retrieval, slot,
price, constraint-source consistency, durable category, and exact evidence-coverage signals.
A confidence policy decides whether to ask a question or recommend. Precision-first slates
show a high-confidence hero item early, then expand as evidence accumulates; unseen-product
rotation preserves cumulative recall. A bounded candidate-memory beam and exact-evidence
route recover products when later boilerplate-heavy queries drift. Intent overrides rewrite
the affected state without discarding unrelated confirmed preferences.

## How we built it

- Language: Python 3.10+
- Core libraries: Python standard library and SQLite FTS5
- Default semantic retrieval: in-memory TF-IDF cosine similarity
- Optional local model: `sentence-transformers` with `all-MiniLM-L6-v2`
- Optional API: an OpenAI-compatible chat-completions endpoint for grounded listwise
  reranking (`COPILOT_LLM_URL`, `COPILOT_LLM_KEY`, `COPILOT_LLM_MODEL`)
- Validation tools: organizer local evaluator; optional pytest and scikit-learn diagnostics
- Dataset: the frozen 50,000-item Amazon Reviews 2023 Clothing, Shoes and Jewelry catalog
  supplied for the challenge; attribution is documented in `DATA_ATTRIBUTION.md`
- Development tools: [LIST THE EDITOR, AI ASSISTANTS, AND OTHER TOOLS ACTUALLY USED]

The validated submission uses the offline TF-IDF route and no external API. It therefore
works when network access and credentials are unavailable.

## Public development results

On the untouched 200-session public set:

- Hit Rate@10: 1.0 (200/200)
- MRR: 0.939048
- MTTC: 3.325 turns
- Efficiency: 0.7675
- Recommended Technical Score: 0.935214
- Model tokens: 0
- Estimated API cost: $0 for the validated run

Buying, Browsing, Intent Override, and Boundary all reach 100% public Hit Rate@10.
Public-set results are development measurements and may differ on the 800-session private
set.

## Challenges and accomplishments

The main challenge was not just first-turn retrieval—it was preserving useful evidence
across ten turns without repeating the same products or allowing a pivot to erase unrelated
constraints. Durable free-text evidence, scoped slot updates, route-aware fusion, candidate
memory, exact-evidence recall, constraint-source consistency, durable category context, and
precision-first expanding slates improved public Hit Rate@10 from 0.65 to 1.0 and MRR to
0.939048 while keeping the system offline and grounded. Every returned ASIN is validated
against the frozen catalog.

## Limitations

Some novelty-shirt groups contain hundreds of products with identical disclosed manufacturing
text and no title, saying, brand, or unique identifier. The new recovery routes cover the
released cases but cannot manufacture undisclosed information, so the perfect public score is
not a private-set guarantee. The default TF-IDF channel is lexical, prices are missing for
many rows, and optional model quality/cost depends on the selected external endpoint.

## Reproduction

From `techjam-conversational-search/`:

```bash
python -m evaluator.local_evaluator
python demo.py
```

Python 3.10+ and the standard library are sufficient for both default commands. Optional
development/model dependencies are listed in `requirements.txt`.

## Demo video

[PASTE THE PUBLIC YOUTUBE URL]

## Team contributions

- [NAME]: [SPECIFIC DESIGN, IMPLEMENTATION, TESTING, DOCUMENTATION, OR DEMO WORK]
- [NAME]: [SPECIFIC CONTRIBUTION]

## Source repository

[PASTE THE PUBLIC REPOSITORY URL]
