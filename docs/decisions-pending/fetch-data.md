# Pending decisions: fetch-data

Decisions taken while building `scripts/fetch_data.py` and `config/data_sources.toml`, for the
orchestrator to merge into `docs/DECISIONS.md`.

## Manifest in TOML, checksums in a separate JSON lock
- **Decision:** people edit `config/data_sources.toml`. Checksums live in
  `config/data_sources.lock.json` as `{"version": 1, "files": {id: {dest: sha256}}}`. Only
  `--record-checksums` writes the lock, and it only adds entries that are missing. It never
  overwrites an existing entry. To accept a changed upstream file, delete its lock entry by hand.
- **Alternatives:** sha256 inline in the TOML (it is hand-edited and noisy to rewrite); letting
  `--record-checksums` overwrite mismatches (one flag would then silently accept tampered data).
- **Why:** reviewable diffs, and a mismatch always needs a deliberate human step.

## Strict schema; `extract` is required
- **Decision:** unknown keys at any level are errors. Keys that belong to another access kind
  (e.g. `repo_id` on an `auto` source) get an error that says so. `extract` must be written out
  on every file. Optional keys are `version`, `licence_url` and `notes`.
- **Why:** a typo such as `sha265` or `extarct` should fail loudly, not be ignored.

## Manual `expected` entries are glob patterns
- **Decision:** `expected` takes paths or glob patterns relative to `data/raw/<id>/`, matched
  with `Path.glob`. A pattern counts as present when it matches at least one real data file.
  `SOURCE.json`, `*.part`, `*.bad` and `.cache/` do not count. Seeded sources use `**/*.csv` or
  `**/*`, because nobody knows the exact file names yet.
- **Why:** the exact names of the manual downloads (ISEAR, EmoSurv, ...) are unknown until
  someone downloads them. Tighten the patterns once they are known.

## Source choices differing from R1's JSON
- `goemotions` is an `hf_dataset` pinned to R1's revision `add4922...`. A separate `auto` source,
  `goemotions_meta`, fetches `emotions.txt` and `ekman_mapping.json` from GitHub. The raw
  rater-level CSVs on `storage.googleapis.com` are left out: that host is not on the ADR-0008
  allowlist, and the files contain Reddit usernames.
- `dailydialog` is an `hf_dataset` (`roskoN/dailydialog`). R1 listed only `train.zip`, and
  guessing the other split names was riskier. Its revision is `main` (unpinned) because R1 gave
  no sha. The summary reports it as unpinned.
- `cedr` and `ru_go_emotions` are `hf_dataset` sources at R1's shas.
- `armeniannlp` is `manual`, not `auto`. R1's only URL is a GitHub repo page, not a file.
- `warriner2013` uses only the first mirror (JULIELab/XANEW). The second mirror is in `notes`.
- Every source has `verified = false`. `licence_class` is R1's strictest reading.

## Extraction
- **Decision:** archives are extracted next to themselves, inside `data/raw/<id>/`, and the
  archive is kept. `<archive>.extracted` stores the archive's sha256, so a rerun skips
  extraction. Every member is validated before anything is written: absolute paths, `..`,
  backslashes, drive letters, zip symlinks, tar devices, and tar links that escape are all
  rejected. Tar then also extracts with `filter="data"`.
- **Why:** validation that only happens during extraction can leave half an archive on disk.

## hf_dataset sources
- **Decision:** `snapshot_download(repo_type="dataset", revision=..., local_dir=data/raw/<id>,
  etag_timeout=timeout)`. After the snapshot, every file is hashed (`.cache/` excluded) and
  checked against the lock. `huggingface_hub` is imported lazily, with telemetry off
  (ADR-0007). It arrives as a transitive dependency of `transformers`/`datasets` and is not
  declared directly in `pyproject.toml`.

## User-Agent
- **Decision:** `emotion-engine-fetch-data/0.1`. It contains no contact details, and no
  `urllib` token: saifmohammad.com returns 406 to any User-Agent containing "urllib" (found by
  the network HEAD test on 2026-11-18).

## Exit codes
- 0: success, including when manual sources are missing (reported loudly). 1: an auto or
  hf_dataset source failed. 2: a usage, manifest or lock-file error.
