# Phase 0 Implementation Plan

**Goal:** working environment, tools table and repo skeleton (brief §2).
**Spec:** `docs/superpowers/specs/2026-11-18-phase-0-environment-design.md`

- [x] Branch `phase-0-setup`
- [x] Cloud environment `emotion-engine` with the hosts in ADR-0008 (plus `download-r2.pytorch.org` and `us.aws.cdn.hf.co`, which turned out to be needed)
- [x] `pyproject.toml` (CPU torch index, spaCy model URL); `uv sync` succeeds
- [x] Tests first: `tests/unit/test_environment.py` (imports, spaCy lemmas, en/ru/hy language ID, VADER sign, git-ignored sensitive paths, telemetry off, GoEmotions download marked `network`)
- [x] `engine/privacy.py` + `.gitignore` until all 35 tests pass
- [x] `scripts/check_env.py` report; numbers copied into `docs/TOOLS.md`
- [x] Tools inventory and vetting table (`docs/TOOLS.md`)
- [x] Skeleton per brief §8, docs stubs, ADRs, Makefile, `.env.example`
- [x] Commit and push `phase-0-setup`
- [x] Save the install as the environment's setup script
