# Wiymood AI: roadmap and task list

Tick boxes as work lands. One session = one block below. Rules: see CLAUDE.md.

## Session 1 (done 2026-11-19), Phase 3 engine A (lexicon) + engine C (stylometric)

Start prompt: `Read CLAUDE.md and docs/ROADMAP.md. Do Session 1.`

### 0. Warm-up (5 min)
- [x] Check `data/raw/` and `data/models/` exist (else check `/tmp/fetch_data.log`, re-run `scripts/fetch_data.py`)
- [x] `make test` is green (89 tests)

### 1. Shared evaluation helpers: `eval/metrics.py`
- [x] macro/micro/per-label F1, ROC-AUC (multi-label)
- [x] Pearson r, CCC, MAE for VAD
- [x] bootstrap 95% CI (fixed seed, 1,000 resamples)
- [x] `eval/datasets.py`: loaders returning canonical labels / [-1, 1] VAD for GoEmotions test, EmoBank test split, SemEval EI-reg (intensity)
- [x] property tests: CCC(x, x) = 1, CI contains the point estimate, metrics in range

### 2. Engine A, lexicon: `engine/engines/lexicon.py`
- [x] `scripts/build_lexicons.py`: NRC VAD v2.1 (+ multi-word expressions) and EmoLex → compact `data/processed/lexicon_en.parquet`
- [x] Token lookup on lemma, longest multi-word match first
- [x] Rules from preprocessing: negation flips valence (×−0.74, VADER constant), dampens arousal; booster scales; after-"but" tokens weighted 1.5×, before 0.5×; CAPS / "!!!" raise arousal
- [x] EmoLex 8 emotions → canonical probabilities (via the inverse of the Plutchik table)
- [x] `confidence` from matched-token count; `coverage` = matched / content tokens
- [x] Behavioural tests: negation flips sign; "sad" < "very sad" < "devastated" (intensity); swapping names or cities leaves the output unchanged; lowercase / no punctuation / 1 typo gives a similar output
- [x] Evaluate vs VADER baseline: EmoBank valence r/CCC, GoEmotions coarse (Ekman) F1 → `eval/reports/lexicon.md`

### 3. Engine C, stylometric: `engine/engines/stylometric.py`
- [x] Features: first-person singular rate, absolutist words (19-word list), past/future tense ratio (spaCy morphology), hedges vs certainty words, question rate, length, type-token ratio, emoji, punctuation intensity, Empath top categories
- [x] Output = features + weak VAD (small ridge model trained on EmoBank train split) + low confidence
- [x] Tests: every feature on hand-made sentences; empty / emoji-only text does not crash
- [x] Evaluate: EmoBank r/CCC vs the mean-prediction baseline → `eval/reports/stylometric.md`

### 4. Wrap-up
- [x] ruff + full tests green, commit, push
- [x] 5-line summary per engine; update CLAUDE.md "State"

### Later (from the 2025 findings)
- [ ] Lexicon engine: down-weight past-tense and other-experiencer clauses using `engine/preprocessing/clauses.py`
- [x] Schema v0.2, `clauses.py`, `eval/challenge_set.jsonl` + `eval/challenge.py` (added after Session 1)

## Session 2: engine B (transformer), done 2026-11-19
- [x] Wrap `SamLowe/roberta-base-go_emotions` (v0); per-label thresholds tuned on validation
- [x] GoEmotions test: macro/micro F1, ROC-AUC, CIs; latency per message on CPU
- [x] VAD head: ridge on frozen embeddings → EmoBank (cheap); fine-tune DistilRoBERTa only if time allows (background, 5k rows first)
- [x] Behavioural tests (shared suite from Session 1) → `eval/reports/transformer.md`

## Session 3: engine F (semantic map)
- [ ] `scripts/build_cluster_map.py`: MiniLM embeddings (GoEmotions train) → UMAP → HDBSCAN → agglomerative over centroids
- [ ] Label each node (dominant emotions + top words); purity / NMI vs labels; seed stability
- [ ] Engine: nearest leaf + path + k nearest neighbourhoods → `eval/reports/semantic_cluster.md`
- [ ] Fit UMAP/HDBSCAN once offline and save artifacts; assign new messages by nearest prototype (never refit per request)
- [ ] K-means baseline for comparison
- [ ] c-TF-IDF + representative phrases for node names
- [ ] Soft membership; stability across seeds
- [ ] Standardize/weight before any feature concatenation (a 384-D embedding must not drown VAD)

## Session 4: engine E (appraisal, local)
- [ ] Zero-shot NLI over the 16 event types + 7 needs + agent (multilingual MiniLM)
- [ ] Fallback classifier: DistilRoBERTa on EmpatheticDialogues prompts → event type (5k subset first)
- [ ] Evidence spans (sentence with the highest entailment); merge safety with `engine/safety.py`
- [ ] Claude API path: stub, off by default
- [ ] Evaluate on the ED test split → `eval/reports/appraisal.md`
- [ ] Use `clauses.py`: desired state from desire clauses, experiencer attribution, temporal weighting; explicit user correction overrides inference

## Session 5: engine D (keystrokes), needs EmoSurv CSVs from you
- [ ] Feature extractor (speed, dwell, flight, pauses > 2 s, backspace ratio, bursts, time to first key)
- [ ] Personal baseline z-scores (Welford); validate on EmoSurv → `eval/reports/keystroke.md`

## Sessions 6–7: Phase 4, integration
- [ ] Run `eval/challenge.py` on fused output; abstention quality metric
- [ ] Weighted fusion → stacking LR + temperature scaling; ECE + reliability plot
- [ ] Ablation table (each engine alone, plus combinations) with CIs
- [ ] Temporal: baseline, EMA, Kalman (HMM only if it wins); synthetic sad → neutral → happy test
- [ ] FastAPI "Wiymood AI": `/analyze`, `/session/{id}/message`, `/session/{id}/pattern`, `DELETE /user/{id}/data`
- [ ] Latency < 300 ms/message (without engine E); `make eval` → `eval/reports/SUMMARY.md`

## Sessions 8–9: Phase 5 UI + study
- [ ] Wiymood AI chat (Next.js), keystroke categories only, hide-panel toggle, SAM self-report < 10 s
- [ ] SQLite storage + export + delete
- [ ] You: 2 weeks, 150+ messages
- [ ] Analysis script + personal calibration → `eval/reports/PERSONAL_STUDY.md`

## Waiting on you
- [ ] EmoSurv CSVs (before Session 5)
- [ ] Decide: SQLite or Supabase; Next.js or Gradio; study languages (before Session 8)
