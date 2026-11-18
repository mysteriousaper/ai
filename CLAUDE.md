# Emotion Engine: working rules

Brief: `PROJECT_BRIEF.md`. Product/demo name: **Wiymood AI** (use it in the UI, API title and reports). State: Phases 0–2 done + preprocessing (`engine/preprocessing/pipeline.py`, tested). Next: see `docs/ROADMAP.md` (task list per session).

## Zero money
- No paid APIs, licences or compute. Engine E (appraisal) is **local only**: a zero-shot NLI model (`MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`, MIT) plus a DistilRoBERTa classifier on EmpatheticDialogues. Keep the Claude API path as an optional, off-by-default stub.
- Research-only data is fine for R&D (it's free); commercial swaps are listed in docs/DATA_SOURCES.md.

## Token budget (≤ 20% of a session per phase)
- One phase (or one engine) per session; start fresh sessions instead of growing one long chat. Read this file + only the files you will edit.
- No web research, no subagents, and don't load skill docs unless a task is blocked.
- Batch work: write whole files in one call, run commands chained in one call, pipe output through `tail`/`grep`, run pytest with `-q`.
- Never print datasets or large files; inspect with `head`, `shape`, or summaries.
- Train on subsets first (e.g. 5k GoEmotions rows, 1 epoch) and run full training once, in the background, logging to a file.
- Prefer pretrained models over training (`SamLowe/roberta-base-go_emotions` is Engine B v0).
- Reports: short tables, no long prose. Stop at the end of each engine with a 5-line summary.

## Commands
The environment setup script already runs `uv sync` + `fetch_data.py` in new containers (log: /tmp/fetch_data.log); only re-run if data/ is missing.
`make test` · `uv run python scripts/export_schema.py`
