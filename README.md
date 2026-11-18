# Emotion Inference Engine

A text-first engine that estimates how someone feels (discrete emotions and
valence/arousal/dominance), what probably happened, what they need, and how their state moves
over time. It works from what they write and how they type it, and every output carries a
calibrated confidence. Full brief: [PROJECT_BRIEF.md](PROJECT_BRIEF.md).

**Status:** Phase 0 (environment and skeleton) done. See `docs/` for architecture, decisions,
data sources and tools.

## Quick start

```bash
make setup       # uv sync: CPU-only torch + all dependencies
make test-fast   # offline unit tests
make test        # full suite (downloads a pretrained GoEmotions model, ~480 MB)
make check-env   # hardware / package / smoke-test report
```

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/).

## Privacy

- `data/` (datasets, personal study data), `mlruns/`, `.env` and model weights are git-ignored.
- Third-party telemetry (Hugging Face, MLflow, Gradio) is switched off on `import engine`.
- The only external call with user text is the Claude API appraisal engine (Phase 3), and it
  is optional.

## Branches

One branch per phase (`phase-0-setup`, `phase-1-research`, ...). `main` is updated only at a
checkpoint you approve.
