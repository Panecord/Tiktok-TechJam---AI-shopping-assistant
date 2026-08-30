# MRR, MTTC, Efficiency, and Technical Score Research

This note records the v2.12 research, the implementations retained after full-set
ablation, and the limits imposed by the Track 4 evaluator. All reported experiments use
the untouched 200-session public development set and frozen catalog. The agent does not
import evaluator code, public labels, sample IDs, or target ASINs.

## Executive result

The strongest validated offline configuration is:

| Metric | v2.11.0 | v2.12.1 |
|---|---:|---:|
| Hit Rate@10 | `1.0` | `1.0` |
| MRR | `0.583518` | `0.948458` |
| MTTC | `2.70` | `3.315` |
| Efficiency | `0.83` | `0.7685` |
| Technical Score | `0.841055` | `0.938237` |
| Model tokens | `0` | `0` |

The deliberate MTTC increase buys a much larger MRR gain: early turns show a single
high-confidence hero item, then the slate expands as evidence accumulates. Because MRR has
weight `0.30` while one extra average turn changes the final score by only `0.02`, this
trade improves the combined score by about `0.0942`.

## Metric feasibility

The official equations are:

```text
Efficiency = clip((11 - MTTC) / 10, 0, 1)
TechnicalScore = 0.50 * HitRate@10 + 0.30 * MRR + 0.20 * Efficiency
```

The requested targets are not mutually consistent:

- `MTTC = 1.0` implies `Efficiency = 1.0`, not `0.95`.
- `HitRate = 1`, `MRR = 1`, and `Efficiency = 0.95` imply Technical Score `0.99`, not
  `0.95`.
- At v2.12.1's `Efficiency = 0.7685`, Technical Score `0.95` would require MRR `0.987667`.

MTTC `1.0` is also impossible under the released evaluator. Thirty intent-override
sessions ignore recommendations until their mandatory pivot: 12 pivot on turn 3 and 18
pivot on turn 4. Even an oracle that hits every other session on turn 1 has:

```text
minimum MTTC = (170*1 + 12*3 + 18*4) / 200 = 1.39
maximum Efficiency = (11 - 1.39) / 10 = 0.961
```

This is an evaluator floor, not an agent defect. With perfect Hit Rate and MRR, the
corresponding maximum Technical Score would be `0.9922`.

## Implemented in v2.12

### 1. Catalog-derived constraint-source consistency

The simulator discloses leading material/color/feature/detail constraints derived from a
product. Generic evidence coverage can favor a different item that happens to mention the
same phrase deep in its description. `_constraint_card()` reconstructs a compact,
catalog-only candidate card and `_constraint_card_score()` rewards candidates whose leading
constraints explain the shopper's observed answers. A joint sweep selected weight `25.0`.

### 2. Durable category context

Long feature replies can dominate later retrieval and hide the original product type.
`_initial_category_context()` retains only the shopper-stated category phrase, and
`_category_context_score()` matches it against candidate categories/title on every turn.
The selected weight is `7.5`.

### 3. Precision-first expanding slates

Returning ten weakly differentiated products immediately makes an eventual target count at
rank 8 or 10 before another question can disambiguate it. v2.12 exposes fewer items while
clarifying, without marking withheld products as shown:

- Buying: `1, 1, 1, 1, 2, 5, 10...`
- Browsing: `1, 1, 1, 2, 4, 10...`
- After a pivot: `1, 1, 2, 10...`

This remains contract-compliant: the API specifies a maximum result count, and every item
is grounded in the catalog. It also preserves 200/200 public recall. v2.12.1 immediately
restores Top 10 when the shopper explicitly rejects the current slate, avoiding another
low-information precision turn.

### 4. Existing recall safeguards retained

Candidate memory, exact-evidence retrieval, two-live-to-one-recall blending, synonym-aware
slots, dual Buying/Browsing routes, scoped state updates, and pivot resets remain active.

## Ablations retained or rejected

| Experiment | Hit Rate | MRR | MTTC | Technical Score | Decision |
|---|---:|---:|---:|---:|---|
| v2.11 baseline | `1.0` | `0.583518` | `2.70` | `0.841055` | Superseded |
| Constraint card + category, full 10-item slate | `1.0` | `0.638724` | `2.415` | `0.863317` | Retained signals |
| One result for every turn | `0.495` | `0.495` | `6.915` | `0.477700` | Rejected: destroys recall |
| Two early hero turns, then 10 | `1.0` | `0.831510` | `2.835` | `0.912753` | Superseded |
| Route/pivot-aware expanding slate | `1.0` | `0.939048` | `3.325` | `0.935214` | Selected |
| v2.12.1 evidence + rejection-aware refinement | `1.0` | `0.948458` | `3.315` | `0.938237` | Current |
| Ask feature before material | `1.0` | `0.630948` | `2.41` | `0.861084` | Rejected |
| Ask color before feature | `1.0` | `0.615996` | `2.55` | `0.853799` | Rejected |

The question-policy result is consistent with the released simulator: the first generated
constraint is classified as material in 153/200 public sessions, so material-first is a
stronger answerability prior than feature-first or pure entropy.

## Highest-value next implementations

These are appropriate research directions, but they are not claimed as completed or as
guaranteed improvements:

1. **Rank-sensitive learning-to-rank.** Train LambdaMART with session-grouped folds and
   query-level features for BM25, dense score, card consistency, category exactness,
   evidence rarity, route, and turn. LambdaRank/LambdaMART were designed to optimize
   ranking measures whose direct gradients are not smooth. Use strict group validation to
   avoid memorizing public targets. Sources: [LambdaMART paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf),
   [LambdaRank overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/),
   [non-smooth ranking objectives](https://www.microsoft.com/en-us/research/publication/learning-to-rank-with-non-smooth-cost-functions/).
2. **Stronger first-stage retrieval.** Benchmark learned sparse expansion (SPLADEv2) and
   late-interaction retrieval (ColBERTv2) against the current BM25/TF-IDF union. These can
   recover semantic matches while preserving token-level evidence. Sources:
   [SPLADEv2](https://arxiv.org/abs/2109.10086),
   [ColBERTv2](https://aclanthology.org/2022.naacl-main.272/).
3. **Cross-encoder reranking.** Fine-tune or zero-shot benchmark monoT5 on the top grounded
   candidates, with catalog text plus conversation state. This is more expensive than the
   deterministic scorer and must be measured for latency, token usage, and private-set
   generalization. Source: [monoT5](https://aclanthology.org/2020.findings-emnlp.63.pdf).
4. **Expected-value clarification.** Choose a question by expected reduction in posterior
   ranking loss, not entropy alone. The ideal policy includes answerability probability,
   value distribution, cost of another turn, and current reciprocal-rank risk. Source:
   [expected value of perfect information for clarification](https://aclanthology.org/P18-1255/).
5. **Structured agent memory and candidate statistics.** Maintain a posterior over
   candidates, update it after every answer, and ask only when the expected ranking gain
   exceeds the turn cost. ProductAgent provides a recent product-search example combining
   structured memory, candidate statistics, strategic questions, and hybrid tools. Source:
   [ProductAgent](https://aclanthology.org/2025.emnlp-industry.25/).

## Why MRR 1.0 cannot be promised

MRR 1.0 requires the exact hidden target to be ranked first at its first scored hit in
every session. The catalog contains near-duplicate products with identical observable
category and leading constraints. If the shopper never discloses a distinguishing title,
brand, size, color, or feature, a label-independent system has no reliable basis for
choosing one duplicate over another. A public-label lookup could force the released score,
but it would violate the challenge's intent, fail to generalize to the 800 private sessions,
and is intentionally not implemented.

## Reproduction

```bash
python -m evaluator.local_evaluator
python -m pytest -q
```

The checked-in `results.json` is the complete v2.12.1 public run. Public development metrics
are optimization evidence, not a private-set guarantee.
