# Installed tools (Phase 0)

Environment: cloud container, Linux, **CPU only** (4 threads, ~16 GB RAM, no CUDA).
Python 3.11.15 via `uv`, Node 22. Licences below are read from installed package metadata, not from memory.

## Python packages

| Package | Version | Licence | Commercial | Why |
|---|---|---|---|---|
| torch | 2.15.0 | BSD-3 + bundled permissive (Apache/MIT/BSL/Zlib/ISC) | OK | Model inference / fine-tuning |
| transformers | 5.23.0 | Apache-2.0 | OK | Engine B (encoder models) |
| datasets | 5.2.0 | Apache-2.0 | OK | Loading GoEmotions, EmoBank, etc. |
| accelerate | 1.16.0 | Apache-2.0 | OK | Training loop helper |
| sentence-transformers | 6.3.0 | Apache-2.0 | OK | Engine F embeddings |
| scikit-learn | 1.9.1 | BSD-3 | OK | Stacking, calibration, metrics, agglomerative clustering |
| scipy / numpy / pandas | 1.17 / 2.4 / 3.1 | BSD-3 | OK | Maths, bootstrap CIs |
| spacy + en_core_web_sm | 3.8.18 / 3.8.0 | MIT | OK | Tokenise, lemmatise, dependency parse (negation scope) |
| hdbscan | 0.8.44 | BSD-3 | OK | Engine F clustering |
| umap-learn | 0.6.0 | BSD-3 | OK | Engine F dimensionality reduction |
| pydantic | 2.14.3 | MIT | OK | Schemas, JSON Schema export |
| fastapi / uvicorn | 0.164 / 0.54 | MIT / BSD-3 | OK | Phase 4 API |
| vaderSentiment | 3.3.2 | MIT | OK | Baseline + heuristic reference for Engine A |
| emoji | 2.16.0 | BSD-3 | OK | Emoji → text mapping |
| langdetect | 1.0.9 | Apache-2.0 (metadata also says MIT — both permissive) | OK | Language ID |
| trackio | 0.45.1 | MIT | OK | Local experiment tracking (chosen over MLflow: lighter, no server) |
| pytest / hypothesis / ruff | 9.2 / 6.170 / 0.17 | MIT / **MPL-2.0** / MIT | OK (dev only) | Tests, property tests, lint |

## Claude Code skills available in this session

| Skill | Source | Use in project |
|---|---|---|
| code-review, security-review, simplify | built-in | End of each phase, before any deploy |
| skill-creator | Anthropic official | Build `emotion-eval`, `add-new-engine` (Phase 2–3) |
| xlsx, pdf | Anthropic official | Optional report exports |
| dataviz | built-in | Reliability diagrams, eval charts |
| claude-api | built-in | Engine E (appraisal LLM call) |

## Not installed (and why)

| Item | Reason |
|---|---|
| Hugging Face MCP | Not needed: HF Hub reached directly via `huggingface_hub` + the vetted `hf-cli` skill. |
| superpowers `brainstorming` visual server | Copied but not used (spawns a local web server; markdown workflow is enough). |
| HF `community-evals`, `llm-trainer`, cloud/SageMaker skills | Target GPU/vLLM/HF Jobs (paid); not applicable on CPU. |
| GitHub MCP | Connector failed to connect (proxy 403). Plain `git push` to `mysteriousaper/ai` works. |
| Context7 MCP | Not configured; low value vs. cost. Ask if you want it. |
| Supabase MCP | Deferred to Phase 5 per brief. |
| External community skills (awesome-* lists) | Skipped to save credits; no gap that built-ins don't cover yet. |

## External skills (vetted, installed in `.claude/skills/`)

Copied into the repo so they travel with it. Each folder carries its upstream LICENSE.

| Skill | Source (commit) | Licence | Stars / activity | Code review | Why |
|---|---|---|---|---|---|
| brainstorming, writing-plans, test-driven-development, systematic-debugging, verification-before-completion | [obra/superpowers](https://github.com/obra/superpowers) @ 5291c2b | MIT | ~320k ★, commit 2026-11-03 | Markdown instructions; only script is brainstorming's localhost server (own random token, timing-safe compare, loads a logo image). No credential access or obfuscation. | Plan → TDD → verify workflow from the brief |
| hf-cli, huggingface-datasets, huggingface-trackio, train-sentence-transformers, huggingface-gradio | [huggingface/skills](https://github.com/huggingface/skills) @ 22f0b46 | Apache-2.0 | ~11k ★, commit 2026-11-17 | Markdown + example training scripts; no subprocess/network calls beyond the HF libraries; no credential handling. | Data fetch, experiment tracking, engine F embedding fine-tune, Phase 5 Gradio UI option |

## Network

This repo runs in its own cloud environment **"mysteriousaper-ai (emotion-engine)"**, which allows: Hugging Face (+ `us/eu.aws.cdn.hf.co` weight CDN), saifmohammad.com (NRC), dl.fbaipublicfiles.com (EmpatheticDialogues), zenodo, ieee-dataport, springer static content, storage.googleapis.com, github.com, download.pytorch.org.
`scripts/fetch_data.py` sets `HF_HUB_DISABLE_XET=1` because the Xet CAS hosts are not reliable through the proxy.
