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
| superpowers plugin, Hugging Face skills | Plugins can't be installed from a cloud session (`/plugin` is local CLI only). Install on your machine if you run Claude Code locally. |
| Hugging Face MCP, HF Hub access | `huggingface.co` blocked by environment network policy. **Skipped by your decision**; needed before Phase 1 dataset/model fetch and Phase 3 Engine B. |
| GitHub MCP | Connector failed to connect (proxy 403). Plain `git push` to `mysteriousaper/ai` works. |
| Context7 MCP | Not configured; low value vs. cost. Ask if you want it. |
| Supabase MCP | Deferred to Phase 5 per brief. |
| External community skills (awesome-* lists) | Skipped to save credits; no gap that built-ins don't cover yet. |

## Network blockers found

- `huggingface.co` — blocked (needed later).
- `download.pytorch.org` — blocked, so torch comes from PyPI (CUDA build, ~6 GB venv; runs on CPU fine).
