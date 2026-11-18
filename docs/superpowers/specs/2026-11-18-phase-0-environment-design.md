# Phase 0: environment, tools, skeleton (short design)

Approved in chat on 2026-11-18 ("Use this repo, phase branches are fine, go").

- **Intent:** a reproducible, privacy-safe workspace where Phases 1 to 5 can run without
  setup surprises.
- **Goals:** working Python environment with the brief's §2.1 packages; proof that models
  download and run on this hardware; installed-tools table with vetting; repo skeleton per §8.
- **Non-goals:** any engine logic, dataset downloads (Phase 1), schemas (Phase 2).
- **Constraints:** no GPU (4 CPU cores, about 16 GB RAM); cloud network allowlist; ephemeral
  container; brief §1.8 privacy.
- **Approaches considered:** (a) uv + CPU-only torch on Python 3.13 (chosen); (b) pip/venv
  with default torch (2.5 GB of unusable CUDA libraries); (c) conda (heavier; not in your
  stack).
- **Decided by you:** use `mysteriousaper/ai`; one branch per phase.
- **My calls (see DECISIONS.md):** ADR-0002 to ADR-0008.
