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
price, and exact evidence-coverage signals. A confidence policy decides whether to ask a
question or recommend, and unseen-product slate rotation increases recall over multiple
turns. Intent overrides rewrite the affected state without discarding unrelated confirmed
preferences.

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

- Hit Rate@10: 0.995 (199/200)
- MRR: 0.574935
- MTTC: 2.74 turns
- Efficiency: 0.826
- Recommended Technical Score: 0.835180
- Model tokens: 0
- Estimated API cost: $0 for the validated run

The three scenario groups Browsing, Intent Override, and Boundary reach 100% public Hit
Rate@10; Buying reaches 98.75%. Public-set results are development measurements and may
differ on the 800-session private set.

## Challenges and accomplishments

The main challenge was not just first-turn retrieval—it was preserving useful evidence
across ten turns without repeating the same products or allowing a pivot to erase unrelated
constraints. Durable free-text evidence, scoped slot updates, route-aware fusion, and novel
slates improved public Hit Rate@10 from 0.65 to 0.995 while keeping the system offline and
grounded. Every returned ASIN is validated against the frozen catalog.

## Limitations

The remaining public miss is an underdetermined novelty-shirt case where hundreds of items
share the disclosed generic manufacturing text and the conversation never reveals a title,
saying, brand, or other unique identifier. Adding a public sample-to-target lookup would be
label leakage, so it is intentionally excluded. The default TF-IDF channel is lexical rather
than a neural embedding model, prices are missing for many catalog rows, and optional model
quality/cost depends on the external endpoint selected by the team.

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
