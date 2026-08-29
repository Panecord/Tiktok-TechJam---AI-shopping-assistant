# TechJam Track 4 - Shopping Copilot

This repository contains one subproject: `techjam-conversational-search/`

Inside that folder:

- `README.md` - the competition brief (treat as a read-only spec).
- `IMPROVED_AGENT_README.md` - the current implementation doc (architecture, changelog,
  and metrics). This is the source of truth for how the agent works.
- `docs/TRACK4_COMPLIANCE.md` - official Track 4 requirement-by-requirement audit.
- `docs/v2.8.1_release_notes.md` - concise release results, validation steps, and the
  remaining known limitation.
- `docs/devpost_project_description.md` and `docs/demo_script.md` - submission draft and
  recording outline; YouTube URL and team names still require team input.

Current deterministic implementation: **v2.8.1**. On the released 200-session public
set it reaches Hit Rate@10 `0.995` (199/200), MRR `0.574935`, MTTC `2.74`, and
Technical Score `0.835180`, with zero LLM tokens. These are development-set results;
private-set performance may differ.

Run the evaluator from `techjam-conversational-search/`

    python -m evaluator.local_evaluator

Run the terminal demo:

    python demo.py

Run regression tests after installing the development dependencies:

    python -m pytest -q
