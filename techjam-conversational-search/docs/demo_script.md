# Track 4 Demo Recording Script

Use this as a concise video outline, not as a claim that the video has already been made.

## Before recording

1. Confirm the repository is public and contains no `.env`, API keys, or private data.
2. Run `python -m evaluator.local_evaluator` and keep the final metrics visible.
3. Start `python demo.py --profile-tags comfort,fit,durability`.
4. Use only challenge/catalog assets and music you are licensed to publish.

## Suggested three-minute flow

1. **Problem (20 seconds):** explain that vague, evolving shopping intent is poorly served
   by one-shot keyword search.
2. **Architecture (35 seconds):** show the dual Buying/Browsing router, BM25 plus in-memory
   semantic retrieval, grounded reranking, state, and clarification policy in the README.
3. **Browsing-to-buying session (60 seconds):** start with “I'm looking for women's shoes,
   but I'm still exploring.” Point out `route=browsing`. Answer the next material or feature
   question with a concrete requirement and point out the transition to `route=buying`.
4. **Intent override (30 seconds):** say “Actually, blue instead of red” or another clear
   pivot and show that the affected preference changes while other evidence remains.
5. **Grounding and metrics (25 seconds):** show that results contain catalog ASINs and show
   the checked-in 200/200 result, MRR `0.939048`, Technical Score `0.935214`, and zero
   model tokens/API cost.
6. **Limitations and close (10 seconds):** state that public results do not guarantee private
   performance and mention the underdetermined near-duplicate case.

After upload, paste the public YouTube URL into `docs/devpost_project_description.md` and
the Devpost submission.
