# CLAUDE.md

Read `docs/PROJECT_BRIEF.md` at the start of every session. It is the source of truth for scope, design and working rules.
Agreed decisions and deviations from the brief live in `docs/DECISIONS.md`; they take precedence over the brief where they conflict.

## Non-negotiables (summary; the brief has the detail)
- LLM agents never fit, tune or alter the model. The modelling core is deterministic: fixed seeds, versioned YAML config, same inputs give the same numbers.
- Public data only (Home Credit Default Risk). No employer or proprietary data, names or code.
- `data/`, `outputs/` and `.env` are gitignored. Never commit raw data or secrets.
- Split on `SK_ID_CURR` before any data-learned step; binning, WoE, selection and calibration are fitted on train only.
- The OOT holdout is pseudo-OOT (random, not temporal). Never describe it as temporal validation.
- The ECL layer is an illustrative simulation. Every regulatory parameter goes in config with a source note, verified against the RBI text.

## Working rules
- Build in phases (see brief) and stop for review after each one. Commit after each phase; never push until the author says so.
- LLM mode is `dry-run` for all development and tests. Call Gemini only when explicitly asked, and give the cost estimate first.
- Flag anything technically unsound or in conflict with regulation before building it.

## Local environment
- Raw Kaggle CSVs are already at `data/raw/` (about 2.5 GB).
- Virtualenv at `.venv` (Python 3.13). Install with `.venv/bin/pip install -e ".[dev]"`. Run `.venv/bin/acr run --sample`, `.venv/bin/pytest`, `.venv/bin/ruff check .`.
- Phase status: Phases 1-3 done (skeleton; DQ, split, legs; selection and modelling). Validation, ECL, docs and grounding are stubs until their phase.
- Tests use a synthetic dataset (`tests/synthetic.py`), never the Kaggle files.
