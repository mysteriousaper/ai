# Tools, skills and connectors (Phase 0)

Snapshot of 2026-11-18. "Vetted" means: licence checked, activity checked, and for anything
whose code runs locally, code read before use (brief §2.2.4).

## Claude Code skills and plugins

| Name | Source | Licence | Stars / last push | Status | Why |
|---|---|---|---|---|---|
| superpowers | obra/superpowers | MIT | ~320k / 2026-11 | Pre-installed (your claude.ai org) | Brainstorming, plans, TDD, debugging, verification, code review |
| huggingface-skills | huggingface/skills | Apache-2.0 | ~11k / 2026-11 | Pre-installed. Skills work; its MCP server needs your Hugging Face authorisation | hf-cli, datasets, trackio, Gradio/Spaces, training, community evals |
| Anthropic skills (skill-creator, xlsx, pdf, docx) | anthropics/skills | Mixed: example skills Apache-2.0; document skills source-available under Anthropic's terms | ~197k / 2026-11 | Pre-installed | skill-creator for our own skills; xlsx/pdf for reports |
| K-Dense-AI/claude-scientific-skills | GitHub | MIT | popular, active | **Not installed** | 125+ skills, mostly bio/chem. The ML/statistics parts duplicate what we need, and the whole collection would bloat every session's context. Revisit only if evaluation work needs a specific one. |
| bradautomates/claude-video (`watch` skill) | GitHub, commit 03ceb42 | MIT | ~21k / 2026-09 | **Vendored** in `.claude/skills/watch/` (code read in full; standard-library Python; network only to Gemini/Groq/OpenAI when configured). Startup hook not used. Run via `scripts/watch.sh` (Gemini engine) | Researchers watch public talks/tutorials |
| affaan-m/ECC | GitHub | MIT | ~306k / 2026-11 | **Not installed.** 56 hooks are minified inline Node scripts that would run on every shell command and can't be audited properly; also overlaps with superpowers. Five agent/skill prompts adapted into `.claude/agents/` (read in full; plain markdown) | Checklists for research, verification, ML review, silent failures, test coverage |
| Awesome lists (ComposioHQ, travisvn, BehiSecc awesome-claude-skills; punkpeye/awesome-mcp-servers) | GitHub | lists only | active | Scanned, nothing installed | No emotion/NLP-specific skill beyond Hugging Face and Context7 |

## Project agents (`.claude/agents/`, ADR-0009)
`researcher` (smaller model), `verifier`, `builder`, `bug-hunter`, `tester`.

## Our own project skills (planned)
`emotion-eval`, `add-new-engine` and `dataset-license-check`, created with skill-creator at the
end of Phase 2, once the engine interface and evaluation suite exist for them to wrap.

## MCP connectors

| Connector | Status | What leaves the machine | Action needed |
|---|---|---|---|
| GitHub | Connected, scoped to `mysteriousaper/ai` | repo contents you push | none |
| Hugging Face | Not connected (needs authorisation) | model/dataset search queries | Connect in claude.ai connector settings (optional: model downloads already work without it) |
| Context7 (upstash/context7, MIT, ~65k stars) | Not installed | library names and doc queries only, never user text | Connect in claude.ai connector settings (optional) |
| Supabase | Connected | nothing until Phase 5 | Phase 5: pick the project for study data |

## Python environment

See `scripts/check_env.py` (`make check-env`) for a live report. Snapshot from the cloud
container:

| Item | Value |
|---|---|
| Python | 3.13.18 |
| Hardware | 4 CPU cores, 16.9 GB RAM, no GPU |
| torch | 2.15.0+cpu |
| transformers / datasets / sentence-transformers | 5.23.0 / 5.2.0 / 6.3.0 |
| scikit-learn / scipy / numpy / pandas | 1.9.1 / 1.18.1 / 2.5.5 / 3.1.1 |
| spaCy + en_core_web_sm | 3.8.18 + 3.8.0 |
| umap-learn, lingua, emoji, vaderSentiment | 0.6.0, 2.2.0, 2.16.0, 3.3.2 |
| pydantic / fastapi / uvicorn | 2.14.3 / 0.164.0 / 0.54.0 |
| mlflow, pytest, hypothesis | 3.18.0, 9.2.0, 6.170.15 |

Smoke tests (CPU, warm):

| Check | Result | ms |
|---|---|---|
| spaCy parse "I wasn't happy at all yesterday." | lemmas `I be not happy at all yesterday .` | 5 |
| lingua on an Armenian sentence | ARMENIAN | 1 |
| VADER "I am not happy" | compound −0.459 | <1 |
| GoEmotions RoBERTa on "I miss my dad so much, it hurts" | sadness 0.92, disappointment 0.08, grief 0.04 | 29 |

Observation for Phase 3: spaCy lemmatises "n't" to "not", so negation rules can key on the
lemma `not` and catch both forms.
