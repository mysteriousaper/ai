# Architecture Decision Records

## ADR-001 — Python 3.11 managed by uv
- **Decision:** Pin Python 3.11 (`.python-version`, `requires-python = ">=3.11,<3.12"`), manage deps with `uv`.
- **Alternatives:** System Python 3.13; venv + pip.
- **Why:** 3.13 is newer than some ML wheels (hdbscan, umap/numba) reliably support; uv gives a lockfile and fast reproducible installs.

## ADR-002 — CPU-only, distilled models
- **Decision:** Target CPU inference; prefer distilled encoders (DistilRoBERTa / MiniLM class).
- **Why:** No GPU available; also matches the < 300 ms/message CPU target in Phase 4.

## ADR-003 — torch from PyPI instead of the CPU wheel index
- **Decision:** Install torch from PyPI.
- **Alternatives:** `download.pytorch.org/whl/cpu` (smaller).
- **Why:** That host is blocked by the network policy. Cost: larger venv. Revisit if the host is allowed.

## ADR-004 — trackio over MLflow
- **Decision:** Use trackio for experiment logging.
- **Why:** Local-first, no server, small footprint; MLflow remains a drop-in option.

## ADR-005 — Build in `mysteriousaper/ai`
- **Decision:** Develop in this repo instead of a new `Exishe/emotion-engine`.
- **Why:** GitHub connector unavailable in the session; history can be moved later with `git push` to a new remote.
