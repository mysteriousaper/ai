# Decisions (ADR log)

Each entry: decision, alternatives considered, why. Newest last.

## ADR-0001: Repo and branching
- **Decision:** Code lives in `mysteriousaper/ai`. One branch per phase (`phase-0-setup`,
  `phase-1-research`, ...), each branched from the previous phase. Nothing merges into
  `main` until you say so at a checkpoint.
- **Alternatives:** a new `Exishe/emotion-engine` repo (can't be created from the cloud session).
- **Why:** your choice at the Phase 0 kick-off.

## ADR-0002: Python 3.13 with uv
- **Decision:** `uv` project, `.python-version` = 3.13, `requires-python >= 3.11`.
- **Alternatives:** Python 3.12 (needs a uv interpreter download, which the session's network
  blocks); 3.11 (the brief's floor).
- **Why:** 3.13 is pre-installed in the cloud container and every dependency ships 3.13 wheels.
  The `>= 3.11` floor keeps it installable on your own machine.

## ADR-0003: CPU-only PyTorch; small models by default
- **Decision:** torch comes from the PyTorch CPU wheel index. Default to distilled or base-size
  encoders, and train light heads on frozen embeddings before attempting full fine-tunes.
- **Alternatives:** the default PyPI torch (around 2.5 GB of CUDA libraries that a no-GPU box
  can't use); GPU fine-tuning (no GPU here).
- **Why:** the container has 4 CPU cores, about 16 GB RAM and no GPU. RoBERTa-base inference
  measured 29 ms per message, so inference is fine. A full RoBERTa fine-tune on GoEmotions
  would take hours on CPU. If Phase 3 shows a fine-tune is worth it, the options are paid
  Hugging Face Jobs or your own GPU machine; that decision is deferred to Phase 3.

## ADR-0004: scikit-learn's HDBSCAN
- **Decision:** `sklearn.cluster.HDBSCAN` (scikit-learn >= 1.3).
- **Alternatives:** the standalone `hdbscan` package.
- **Why:** same algorithm with no C build step. Swap only if we need `hdbscan`-only features
  (soft clustering via `all_points_membership_vectors`); revisit in Phase 3 (engine F).

## ADR-0005: lingua for language identification
- **Decision:** `lingua-language-detector`, restricted at runtime to the languages we handle.
- **Alternatives:** `langdetect` (no Armenian); fastText `lid.176` (extra model download, and
  its Python binding is unreliable on new Python versions).
- **Why:** it supports English, Russian and Armenian, runs offline and is accurate on short
  texts. Verified by `test_language_id_covers_english_russian_armenian`.

## ADR-0006: MLflow (local only) for experiment tracking
- **Decision:** MLflow with a local backend (`mlflow.db` / `mlruns/`, git-ignored).
- **Alternatives:** trackio (lighter, but built around syncing to Hugging Face Spaces).
- **Why:** mature, offline by default, standard UI. Nothing leaves the machine.

## ADR-0007: Third-party telemetry off by default
- **Decision:** `import engine` sets `HF_HUB_DISABLE_TELEMETRY`, `MLFLOW_DISABLE_TELEMETRY`,
  `DO_NOT_TRACK` and `GRADIO_ANALYTICS_ENABLED` with `setdefault`, so an explicit user choice
  wins. The Makefile exports the same values.
- **Why:** privacy by design (brief §1.8). Library usage pings are a data flow you didn't approve.

## ADR-0008: Cloud environment `emotion-engine` with an explicit host allowlist
- **Decision:** a personal cloud environment (not the Default one) that adds only these hosts:
  Hugging Face (`huggingface.co`, `cdn-lfs[-us-1].huggingface.co`, `cdn-lfs[-us-1].hf.co`,
  `cas-bridge/cas-server/transfer.xethub.hf.co`, `us.aws.cdn.hf.co`), PyTorch
  (`download.pytorch.org`, `download-r2.pytorch.org`) and the dataset hosts `saifmohammad.com`,
  `dl.fbaipublicfiles.com`, `crr.ugent.be` and `www.unige.ch`.
- **Why:** smallest network surface that lets the brief's models and datasets download. The
  Default environment is shared by all your repos, so it stays untouched.
- **Note:** `www.unige.ch` (ISEAR) answers 403 to scripts through its own bot protection, so
  ISEAR may need a manual download in Phase 1.
