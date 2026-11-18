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

## ADR-006 — Separate cloud environment with an exact host allowlist
- **Decision:** This repo runs in a personal environment "mysteriousaper-ai (emotion-engine)" that allows only the dataset/model hosts we use (list in TOOLS.md). Hugging Face downloads set `HF_HUB_DISABLE_XET=1`.
- **Alternatives:** Unrestricted network on the Default environment.
- **Why:** The Default environment is shared by all repos; exact hosts keep the blast radius small.

## ADR-007 — Canonical labels and weighted coarse mappings
- **Decision:** Canonical = GoEmotions 28. Ekman mapping = GoEmotions' published one. Plutchik mapping = ours, using Plutchik's dyads with 0.5/0.5 splits (remorse, disappointment, optimism, love, curiosity, embarrassment...). Projection takes the *max* weighted contribution per coarse label.
- **Alternatives:** Sum (noisy-OR) projection; single-target Plutchik map.
- **Why:** Multi-label probabilities are not a distribution; summing correlated labels (anger + annoyance) would overstate the coarse emotion. Weighted rows keep mass honest. Known approximations: SemEval "pessimism"→sadness and "trust"→approval.

## ADR-008 — Event taxonomy v0.1 (16 types) from OCC + EmpatheticDialogues
- **Decision:** 16 `EventType`s. ED's 32 labels map to 13 of them; unfairness, routine/boredom and health/body exist only through the LLM path.
- **Why:** ED labels name feelings, not causes, so a direct 32-way "situation" label would just duplicate emotion classification. The taxonomy follows OCC's event/agent/object split so `agent` and `goal_congruence` stay meaningful.

## ADR-009 — Safety screen lives in the core and gates adapters
- **Decision:** `engine/safety.py` sets `safety_flag`; `run_adapter` refuses to call any adapter on `urgent`.
- **Why:** Brief §10 hard rule; putting it in the adapter path would rely on every future product remembering it.

## ADR-010 — `uncertain` field added to the ESO
- **Decision:** Explicit boolean, separate from `mixed_or_ambiguous`.
- **Why:** "Signals disagree" (mixed) and "signals are too weak" (uncertain) call for different product behaviour.
