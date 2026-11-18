# MASTER PROMPT — Human Emotion Inference Engine (Text-First MVP)

> Paste this whole file into Claude Code as the first message, or save it in the project root as `PROJECT_BRIEF.md` and tell Claude Code: "Read PROJECT_BRIEF.md and start Phase 0."

---

## 0. Who you are and what we are building

You are the lead engineer and research scientist for a new project: an **Emotion Inference Engine**, called **"the Engine"** below.

**The goal:** a person writes to the system (chats, types a message), and the Engine works out, from **what** they write and **how** they write it:

1. **How they feel right now.** Discrete emotions (sadness, joy, anger, fear, etc.) plus continuous dimensions: **valence** (pleasant to unpleasant), **arousal** (calm to activated) and **dominance** (in control to overwhelmed).
2. **What probably happened.** The situation or event behind the feeling (e.g. "lost something", "conflict with someone", "achieved a goal", "uncertain about the future") and who or what the person thinks caused it.
3. **What they need.** The underlying need or problem (support, a solution, a change of scene, information, rest, celebration).
4. **Their emotional pattern over time.** How their state moves across messages and sessions compared with their own personal baseline.

The output is a **universal, domain-neutral "Emotional State Object"** (a JSON schema, defined in §4). Later I will plug it into different products and industries: travel recommendations first, then tutoring, wellbeing, customer support and others. **Right now we build only the core logic, prove it works mathematically, and test it on me.** Voice and camera modalities come later, so the architecture must allow new modalities without a rewrite.

**My background:** I am a web developer (Next.js, Supabase, Stripe, Vercel). I can follow Python, but explain ML decisions in plain language when you make them. I work in English, Russian and Armenian. The MVP is **English-first**, but design for multilingual from day one (see §7).

---

## 1. Ground rules (read before doing anything)

1. **Honesty over hype.** Emotion inference from text is *probabilistic*. Every output must carry a calibrated confidence. Never present a guess as fact. When signals are weak or conflicting, the Engine must say "uncertain" rather than invent an emotion.
2. **Plan before code.** Use plan mode (or the `superpowers:brainstorming` → `superpowers:writing-plans` skills if installed) at the start of every phase. Show me the plan, then build.
3. **Checkpoints.** At the end of each phase, STOP and give me: what was built, test results, open questions and the next phase plan. Wait for my "go".
4. **Test-driven.** Write the tests and the evaluation script *before or alongside* each engine (`superpowers:test-driven-development` if available). No engine counts as done without metrics.
5. **Document decisions.** Keep `docs/ARCHITECTURE.md`, `docs/DECISIONS.md` (ADR-style: decision, alternatives, why) and `docs/DATA_SOURCES.md` (every dataset, its licence and its URL) up to date.
6. **Git discipline.** Initialise a repo. Commit after every meaningful step with clear messages. One branch per phase.
7. **Ask me when blocked.** If a download, licence form, API key or install fails, don't loop. Tell me exactly what to fetch and where to put it, and I'll get it for you.
8. **Privacy by design.** Typing and emotion data is highly sensitive. Store raw text and keystrokes only locally or in my own Supabase project, with an easy "delete all my data" path. Never send user data to third-party services except the LLM call I approve in §3.E.

---

## 2. PHASE 0 — Environment, skills, plugins, connectors

### 2.1 Base environment
- Python 3.11+ with `uv` (or `venv` + pip), Node 20+ for the later front end.
- Core Python packages: `torch`, `transformers`, `datasets`, `accelerate`, `sentence-transformers`, `scikit-learn`, `scipy`, `numpy`, `pandas`, `spacy` (+ `en_core_web_sm`), `hdbscan` or `scikit-learn` HDBSCAN, `umap-learn`, `pydantic`, `fastapi`, `uvicorn`, `pytest`, `hypothesis`, `mlflow` or `trackio` (experiment tracking), `vaderSentiment` (baseline), `emoji`, `langdetect` or `fasttext` LID.
- Verify GPU/CPU availability and pick model sizes to match. If there is no GPU, use distilled models and say so.

### 2.2 Claude Code skills and plugins
1. First list what is already installed (`/plugin`, the skills directory).
2. Install or enable, if available:
   - **superpowers** (brainstorming, writing-plans, TDD, systematic-debugging, verification-before-completion, code review).
   - **Hugging Face skills** (hf-cli, datasets, model training/TRL, community evals, trackio, Gradio/Spaces for a quick demo UI).
   - Anthropic's official skills (skill-creator, plus xlsx/pdf for reports if useful).
3. **Search beyond the built-in library** for skills, MCP servers and repos that help with: NLP/emotion analysis, ML evaluation, dataset handling, Python testing, FastAPI, data visualisation. Look at GitHub (e.g. "awesome-claude-skills" and "awesome-mcp-servers" lists), the official Anthropic skills repo and well-known community collections. YouTube and blog tutorials are fine as *pointers*, but install only from the source repo.
4. **Vetting rule. Before installing ANY external skill or repo, check and report to me in a table:**
   - licence (MIT, Apache-2.0 or BSD preferred; flag anything else);
   - stars, recent commits, maintainer reputation;
   - **read the SKILL.md / scripts yourself** and confirm no suspicious network calls, credential access or obfuscated code.
   Install automatically only if it passes all three. Otherwise ask me.
5. If you think a project-specific skill is missing, **create our own** with skill-creator. Candidates: `emotion-eval` (runs the full evaluation suite and prints a report), `dataset-license-check`, `add-new-engine` (scaffold for a new modality/engine following our interfaces).

### 2.3 MCP connectors (propose and set up, ask me for keys)
- **GitHub** (repo: my account `Exishe`, new repo `emotion-engine`).
- **Hugging Face MCP** (model and dataset search).
- **Context7** or similar for up-to-date library docs.
- **Supabase** (my own stack, for storing test sessions later). Use Phase 5, not now.

**Phase 0 deliverable:** a working environment, an installed-tools table (name, source, licence, why), and an empty repo with the structure in §8. → CHECKPOINT.

---

## 3. PHASE 1 — Research and data acquisition

Do a focused literature and resource scan (web search) to confirm and extend what I already found, and write it up in `docs/RESEARCH.md`. I already know about these, so verify the details and licences:

| Resource | What it is | Use in the Engine |
|---|---|---|
| **ANEW** (Bradley & Lang, 1999) | ~1,034 words rated for valence/arousal/dominance | Historical baseline; licence is request-only, so probably skip |
| **Warriner et al. 2013** | VAD norms for **13,915 English lemmas** (likely the "~15,000 words, three scales" I remember) | Core lexicon prior |
| **NRC VAD Lexicon v2** (Mohammad, 2025) | VAD for **55k+ words and multi-word phrases**, scores in [-1, 1] | Main VAD lexicon (check the commercial licence!) |
| **NRC Emotion Lexicon (EmoLex)** | Word → 8 Plutchik emotions + positive/negative | Discrete-emotion lexicon prior |
| **GoEmotions** (Google, 2020) | ~58k Reddit comments, 27 emotions + neutral, multi-label | Train/evaluate the transformer engine |
| **EmoBank** | ~10k sentences with VAD labels | Evaluate sentence-level VAD regression |
| **EmpatheticDialogues** (Meta) | ~25k conversations grounded in 32 emotion *situations* | Train/evaluate "what happened" (situation inference) |
| **ISEAR**, **DailyDialog**, **SemEval-2018 Task 1 (Affect in Tweets)** | Event descriptions + emotion; dialogue emotion; emotion intensity | Extra evaluation sets, intensity |
| **EmoSurv** (IEEE DataPort) | Keystroke timings with emotion labels | Keystroke engine prior and evaluation |
| Keystroke-emotion literature (Zimmermann et al.; Epp et al. 2011; Kołakowska & Landowska 2021) | Typing speed, latency, pauses, backspace rate vs emotion/stress | Feature design |

Also find and evaluate:
- **Pretrained emotion models** on Hugging Face (e.g. RoBERTa fine-tuned on GoEmotions; DistilRoBERTa "emotion-english"; multilingual XLM-R emotion models). Record their reported metrics and licences.
- **Appraisal theory** (Scherer's component process model, the OCC model) for structuring "what happened → what is felt".
- Open alternatives to LIWC for psycholinguistic categories (e.g. **Empath**, open word lists for absolutist words and first-person pronouns).
- My half-remembered "clustering" method: common words go into big clusters which split into sub-clusters. That is **hierarchical clustering of word/sentence embeddings** (agglomerative or HDBSCAN on top of UMAP). Find the best current practice for building an emotion-concept hierarchy this way.

**Licensing is critical.** I may sell this later. In `DATA_SOURCES.md`, mark every resource as `commercial-OK`, `research-only` or `needs-licence`. Design so research-only resources can be swapped out without breaking the architecture.

Download everything allowed into `data/raw/` (git-ignored) with a reproducible `scripts/fetch_data.py`. For anything behind a form, tell me what to download.

**Phase 1 deliverable:** RESEARCH.md, DATA_SOURCES.md, datasets fetched, and a 1-page recommendation of which models/lexicons to use. → CHECKPOINT.

---

## 4. PHASE 2 — Architecture (design first, no heavy code yet)

Write `docs/ARCHITECTURE.md` with diagrams (Mermaid) covering the following design. Improve on it if research shows a better option, and record why in DECISIONS.md.

### 4.1 Pipeline overview

```
[Input capture] → [Preprocessing] → [Parallel signal engines A–F] → [Fusion + calibration]
      → [Temporal state model + personal baseline] → [Emotional State Object] → [Domain adapters (later)]
```

### 4.2 Input capture
- `text`: the message.
- `keystroke_events` (optional): list of `{key_category, down_ts, up_ts}`. **Never log actual characters in the keystroke stream**, only categories (letter, space, backspace, delete, punctuation, enter) to protect privacy.
- `context`: session id, message index, timestamp, language.

### 4.3 Preprocessing
Language detection, sentence split, tokenisation, lemmatisation, **negation scope detection** ("not happy"), **intensifiers/diminishers** ("very", "kind of"), emoji/emoticon mapping, CAPS ratio, elongations ("soooo"), repeated punctuation ("!!!", "..."), typos and slang normalisation.

### 4.4 Signal engines (each one is an independent module with the SAME interface)
Every engine implements `analyze(input) -> EngineOutput` returning: `vad` (3 floats + std), `emotions` (dict of probabilities over our canonical label set), `features` (raw values for explainability), `confidence`, `coverage` (how much of the input it could actually read).

- **A. Lexicon engine.** Weighted VAD + EmoLex lookups with negation, intensifier and contrast ("but") rules (VADER-style heuristics). Fast, explainable, a baseline.
- **B. Transformer engine.** A fine-tuned encoder (start with a pretrained GoEmotions model, then fine-tune our own) for multi-label discrete emotions, plus a regression head for VAD (trained on EmoBank / lexicon-derived targets).
- **C. Stylometric / psycholinguistic engine.** First-person singular pronoun rate, absolutist words ("always", "never", "nothing"), past vs future tense, question rate, message length, lexical diversity, emoji use, punctuation intensity, certainty and hedging words.
- **D. Keystroke dynamics engine.** Typing speed (chars/sec), inter-key latency mean/variance, hold time, pause distribution (pauses > 2s), **backspace/delete ratio**, burst length, revision count, time-to-first-key. **Always as z-scores against the user's personal baseline**: absolute typing speed means little across people.
- **E. Situation & appraisal engine ("what happened").** An LLM call (Claude API) with a strict JSON schema that extracts: event type (from a fixed taxonomy derived from EmpatheticDialogues situations + appraisal theory), the agent responsible (self / other / circumstance), goal congruence, certainty, controllability, and the need (support, solution, information, distraction, rest, celebration). It must quote the text spans it used as evidence. Also build a cheaper local classifier trained on EmpatheticDialogues as a fallback and comparison.
- **F. Semantic clustering engine.** Sentence embeddings → position within a **hierarchical emotion-concept map** built by UMAP + HDBSCAN/agglomerative clustering over the training corpora. Gives "nearest emotional neighbourhoods" and catches emotions the label set misses.

### 4.5 Canonical representation
- One canonical discrete label set (GoEmotions' 27 + neutral), mapped to Plutchik's 8 and to Ekman's 6 for coarse views. Write and test the mapping tables explicitly.
- VAD on a common scale [-1, 1]. Write explicit conversion functions for each lexicon's native scale (Warriner 1–9, NRC 0–1 or -1–1), with unit tests.

### 4.6 Fusion & calibration (the mathematical core)
- Start with **confidence-weighted late fusion**. Then train a **stacking meta-model** (logistic regression / small MLP) on held-out data so the weights are learned, not guessed.
- **Calibration:** temperature scaling or isotonic regression. Report **Expected Calibration Error (ECE)** and reliability diagrams. A "70% sad" should be right about 70% of the time.
- **Uncertainty:** combine per-engine confidence, disagreement between engines and input coverage into a single `overall_confidence`. When engines disagree strongly, flag `mixed_or_ambiguous = true` rather than averaging into mush.
- Handle **missing modalities** cleanly (no keystrokes means fusion without engine D, and confidence drops accordingly).

### 4.7 Temporal state model ("emotional pattern")
- A per-user **baseline profile** (rolling means/variances of VAD, style and keystroke features).
- A state smoother across messages: start with an exponential moving average; evaluate a **Kalman filter** on VAD and a **Hidden Markov Model** over discrete mood states. Pick by metrics.
- Pattern outputs: trend (improving/worsening), volatility, deviation from baseline, recurring triggers (event types associated with negative shifts).

### 4.8 Emotional State Object (the universal API output)
Define it as a Pydantic model + JSON Schema, versioned (`schema_version`):

```json
{
  "schema_version": "0.1",
  "timestamp": "...",
  "language": "en",
  "vad": {"valence": -0.62, "arousal": 0.41, "dominance": -0.30, "std": [0.1, 0.15, 0.2]},
  "emotions": [{"label": "sadness", "p": 0.71}, {"label": "disappointment", "p": 0.44}],
  "coarse": {"plutchik": {...}, "ekman": {...}},
  "intensity": 0.66,
  "situation": {"event_type": "loss_or_failure", "agent": "circumstance", "goal_congruence": -0.8,
                "controllability": -0.5, "evidence_spans": ["..."]},
  "need": [{"type": "emotional_support", "p": 0.6}, {"type": "solution", "p": 0.3}],
  "pattern": {"baseline_deviation": -1.4, "trend": "worsening", "volatility": 0.2},
  "overall_confidence": 0.58,
  "mixed_or_ambiguous": false,
  "explanations": [{"engine": "lexicon", "top_features": ["..."]}],
  "engine_versions": {...}
}
```

### 4.9 Domain adapters (interface only, for now)
An `Adapter` interface: `adapt(EmotionalStateObject, user_profile) -> DomainSuggestion`. Implement ONE toy adapter (`travel_v0`) to show it works: low valence + low arousal + need=rest → calm, nature-heavy destinations, etc. Keep the core completely independent of any domain.

**Phase 2 deliverable:** ARCHITECTURE.md with diagrams, the Pydantic schemas, engine interface stubs, the mapping tables and their tests. → CHECKPOINT.

---

## 5. PHASE 3 — Build and validate each engine in isolation

For each engine A–F, in this order (A, C, B, E, F, D), do the following:
1. Implement it behind the shared interface.
2. Write unit tests + **behavioural tests** (CheckList-style):
   - *Negation:* "I am happy" vs "I am not happy" → valence must flip sign.
   - *Intensity:* "sad" < "very sad" < "devastated" → monotonic arousal/intensity.
   - *Invariance:* changing names or places must not change the emotion.
   - *Robustness:* typos, lowercase, missing punctuation must give a similar output.
   - *Property-based tests* (Hypothesis) for the scale conversions and fusion maths (outputs always in range, probabilities valid).
3. Evaluate on held-out data with the right metrics:
   - Discrete emotions: **macro-F1, micro-F1, per-label F1, ROC-AUC**.
   - VAD: **Pearson r and Concordance Correlation Coefficient (CCC)**, MAE.
   - Situation: accuracy / macro-F1 vs EmpatheticDialogues labels.
   - **95% bootstrap confidence intervals** for every headline number. Use proper train/val/test splits, no leakage, fixed seeds.
4. Compare against simple baselines (majority class, VADER, the pretrained model off the shelf). An engine that does not beat its baseline gets flagged, not hidden.
5. Log every experiment (MLflow/trackio) and write an `eval/reports/<engine>.md`.

For engine D, explain honestly that the public keystroke data is small. Build the feature extractor and personal-baseline logic now, validate it on EmoSurv, and expect the real validation to come from my own data in Phase 5.

**Phase 3 deliverable:** all engines + test suite green + an evaluation report per engine. → CHECKPOINT.

---

## 6. PHASE 4 — Integration: fusion, temporal model, API

1. Implement fusion + calibration (§4.6) and train the stacking model on a held-out split.
2. **Ablation study:** measure the performance of each engine alone and each combination. Prove that fusion beats the best single engine, or tell me it doesn't.
3. Implement the temporal model (§4.7) and test it on synthetic conversations whose ground-truth trajectory we know (e.g. a scripted conversation that moves sad → neutral → happy).
4. Expose a **FastAPI** service: `POST /analyze` (single message), `POST /session/{id}/message` (stateful), `GET /session/{id}/pattern`, `DELETE /user/{id}/data`.
5. Performance target: < 300 ms per message without the LLM engine on CPU, and the LLM engine async/optional. Measure and report.
6. Add `make eval`, a one-command script that runs the full evaluation suite and regenerates `eval/reports/SUMMARY.md`.

**Phase 4 deliverable:** a working API, fusion metrics, ablation table, latency numbers. → CHECKPOINT.

---

## 7. PHASE 5 — Test it on me (personal validation study)

This is where we find out if it really works.

1. **Minimal chat UI.** Next.js page (my stack) or a Gradio app, whichever is faster: a chat box that captures keystroke *timings and categories only* (§4.2) and calls the API. Show the Emotional State Object in a side panel, with a toggle to hide it while I'm typing so it doesn't bias me.
2. **Ground truth via experience sampling.** After each message I quickly self-report: a 1–9 **Self-Assessment Manikin (SAM)** slider for valence/arousal/dominance, plus up to 3 emotion tags and one line on "what happened". It must take less than 10 seconds.
3. **Protocol:** I'll use it for ~2 weeks and aim for 150+ labelled messages across different moods and times of day. Store the data in my Supabase project (ask me for credentials) or locally in SQLite, with export.
4. **Analysis script.** Correlation (r, CCC) between the Engine and my self-reports for V/A/D; F1 for emotions; calibration curve; whether the personal baseline improves accuracy over time (learning curve); which engines help *for me* specifically; and whether keystroke features add signal beyond text (the ablation again, on my data).
5. **Personal calibration.** Fit a lightweight per-user adapter (re-weighting the fusion / bias correction) and show the before/after.
6. Write `eval/reports/PERSONAL_STUDY.md` with honest conclusions: what works, what doesn't, what to fix next.

**Phase 5 deliverable:** the UI, the collected data, the personal study report and a prioritised improvement list. → FINAL CHECKPOINT.

---

## 8. Repository structure

```
emotion-engine/
├── PROJECT_BRIEF.md            # this prompt
├── docs/  ARCHITECTURE.md  DECISIONS.md  DATA_SOURCES.md  RESEARCH.md
├── data/  raw/  processed/     # git-ignored
├── scripts/ fetch_data.py  build_lexicons.py  build_cluster_map.py
├── engine/
│   ├── schemas.py              # Pydantic: EngineOutput, EmotionalStateObject
│   ├── preprocessing/
│   ├── engines/  lexicon.py  transformer.py  stylometric.py  keystroke.py  appraisal.py  semantic_cluster.py
│   ├── fusion/   weighted.py  stacking.py  calibration.py
│   ├── temporal/ baseline.py  ema.py  kalman.py  hmm.py
│   ├── mappings/ labels.py  scales.py
│   └── adapters/ base.py  travel_v0.py
├── api/  main.py
├── ui/                         # Next.js or Gradio test app
├── eval/  run_all.py  reports/
└── tests/ unit/  behavioural/  property/
```

---

## 9. Multilingual plan (design now, build after MVP)
- Keep all language-specific pieces (lexicons, tokenisers, models) behind a `language` key.
- Russian: check for existing Russian VAD/emotion lexicons and models (e.g. RuBERT-based emotion classifiers, translated GoEmotions variants).
- Armenian: resources are scarce. Plan for a multilingual encoder (XLM-R) + translated/adapted lexicons, and note the accuracy risk. My own Armenian messages from Phase 5 can become seed data.
- Mixed-language messages (Armenian written in Latin script, Russian/English switching) should go to the multilingual model, not the lexicon engine.

## 10. Ethics, safety, legal (put this in ARCHITECTURE.md as well)
- The Engine estimates; it does not diagnose. No clinical claims.
- **Crisis safety:** if the text suggests self-harm or danger, the Engine sets a `safety_flag` and any product built on it must show human help resources instead of a "solution". Write this as a hard rule in the core, not left to the adapter.
- **Legal:** note that the EU AI Act restricts emotion recognition in workplaces and schools, and that keystroke/emotion data counts as sensitive personal data under GDPR-style laws. Every future product needs explicit consent and a clear purpose. Flag this for my tutoring use case specifically.
- Measure bias: check whether performance differs by writing style, dialect or non-native English. Report it.

---

## 11. Definition of done for the MVP
- [ ] All engines implemented, tested, and each one beats its baseline (or is documented as not doing so).
- [ ] Fusion beats the best single engine on held-out data, with 95% CIs.
- [ ] Calibration ECE reported and < 0.1 on held-out data.
- [ ] Stateful API running locally with one command.
- [ ] Personal study completed with an honest report.
- [ ] All data sources licensed and documented; a clear list of what must be replaced before commercial use.

**Start now with Phase 0.** First tell me in 5–10 lines how you understand the project, then show me the Phase 0 plan.
