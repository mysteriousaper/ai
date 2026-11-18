# Research notes (Phase 1)

Scope kept deliberately narrow (the brief's resource table, the top pretrained models, and the theory each engine needs). Licence details are in [DATA_SOURCES.md](DATA_SOURCES.md).

## 1. What I verified, and what changed from the brief

| Brief said | Verified | Consequence |
|---|---|---|
| NRC VAD v2, 55k+ terms, [-1, 1] | **v2.1 (March 2025)**, 54,801 terms incl. multi-word expressions, scores in [-1, 1]; non-commercial only | Main VAD lexicon. Already on our common scale, so no conversion is needed. |
| Warriner 13,915 lemmas | Correct, 1–9 scale, **CC BY-NC-ND 3.0** | Research comparison only. |
| GoEmotions ~58k, 27 + neutral | 54,263 in the simplified split (43,410 / 5,426 / 5,427); 211k raw ratings; Apache-2.0 | The only large, commercially clean emotion set. Engine B is built on it. |
| EmoBank ~10k VAD | 10,062 sentences, 1–5 scale, CC-BY-SA | Engine B VAD-head training and evaluation; share-alike flag. |
| EmpatheticDialogues ~25k | ~25k conversations, 32 situation labels, CC BY-NC | Situation (engine E) eval and local-fallback training: research only. |
| EmoSurv | 124 participants, 5 emotions; login wall; non-commercial | **Manual download needed** (see bottom). Small data, as the brief expected. |
| ISEAR | Licence unclear, request-only | Skipped. Optional. |
| SemEval-2018 Task 1 | Downloaded; research only | Intensity evaluation (does "very sad" > "sad"?). |

**Licensing gap:** there is **no free, commercially licensed English VAD lexicon** of useful size. Plan: research pack now; at commercialisation, buy the NRC licence (NRC states it licenses to companies "for a small fee").

## 2. Pretrained emotion models

| Model | Data | Reported | Licence | Verdict |
|---|---|---|---|---|
| `SamLowe/roberta-base-go_emotions` | GoEmotions | F1 0.450 at threshold 0.5 (P 0.575, R 0.396); better with per-label thresholds | MIT | **Engine B baseline.** RoBERTa-base is ~125M params; CPU latency to be measured (an INT8 ONNX version exists if needed). |
| `j-hartmann/emotion-english-distilroberta-base` | 6 mixed datasets | 66% accuracy, 7 classes | none stated | Comparison only. |
| Our `distilroberta-base` fine-tune | GoEmotions (+ EmoBank VAD head) | target: match SamLowe macro-F1 at ~half the latency | Apache-2.0 | **Engine B production candidate.** |
| `paraphrase-multilingual-MiniLM-L12-v2` | — | — | Apache-2.0 | Multilingual embeddings for engine F and as the ru/hy route. |
| Russian: `Djacon/rubert-tiny2-russian-emotion-detection` (MIT), `cointegrated/rubert-tiny2-cedr-emotion-detection` (no licence) | CEDR / ru | F1 macro ~0.6–0.85 per card | — | Phase-after-MVP. |
| Armenian | — | **No dedicated emotion model found.** | — | XLM-R / multilingual MiniLM + NRC auto-translated Armenian lexicon (low quality: Google-Translate based); your own Phase 5 messages become seed data. |

Note: GoEmotions is notoriously hard. The original paper reports macro-F1 ≈ 0.46 for BERT-base. Any engine B result around 0.45–0.52 macro-F1 is state-of-the-art-ish for this data, not a failure.

## 3. Theory each engine uses

### Appraisal (engine E "what happened")
- **Scherer's Component Process Model:** emotions come from a sequence of appraisal checks: *relevance* (novelty, intrinsic pleasantness, goal relevance), *implications* (who caused it, outcome probability, goal conduciveness, urgency), *coping potential* (control, power, adjustment), *normative significance* (fits my values / social norms?).
- **OCC model (Ortony, Clore & Collins):** three branches: consequences of **events** (joy/distress, hope/fear, relief/disappointment), actions of **agents** (pride/shame for self, admiration/reproach for others), aspects of **objects** (love/hate).
- **Our schema** takes the useful, measurable subset: `event_type` (taxonomy below), `agent` ∈ {self, other, circumstance}, `goal_congruence` [-1, 1], `certainty` [0, 1] (≈ outcome probability), `controllability` [-1, 1] (≈ coping potential), `norm_violation` bool, and `need`.
- **Event taxonomy v0** (collapses EmpatheticDialogues' 32 situations + OCC branches into ~12 groups): loss_or_failure, achievement_or_success, conflict_or_mistreatment, threat_or_danger, uncertainty_about_future, social_connection, rejection_or_loneliness, unfairness, surprise_event, routine_or_boredom, health_or_body, other. The mapping from ED labels → taxonomy is written and tested in Phase 2.

### Psycholinguistics (engine C)
- **First-person singular pronouns** ("I, me, my") are reliably elevated in depression/distress (Rude et al. 2004; Tackman et al. 2019 meta-analysis: small but robust effect).
- **Absolutist words** (19-word list: absolutely, all, always, complete, completely, constant, constantly, definitely, entire, ever, every, everyone, everything, full, must, never, nothing, totally, whole) are elevated in anxiety/depression/suicidal-ideation forums (Al-Mosaiwi & Johnstone 2018).
- **Empath** (MIT, 194 categories, validated against LIWC with r ≈ 0.9 on overlapping categories) replaces LIWC.
- Also: tense (past/future focus via spaCy morphology), hedges vs certainty words, question rate, punctuation intensity, CAPS, elongation.
- These are weak signals per message. They are mostly useful **relative to the person's baseline** (engine C feeds the temporal model more than the per-message emotion).

### Keystroke dynamics (engine D)
- Epp, Lippold & Mandryk (CHI 2011): keystroke + content features classified several emotional states (e.g. confidence, hesitance, nervousness, relaxation, sadness, tiredness) above chance in 2-class setups, collected in the wild via experience sampling, the same protocol as our Phase 5.
- Kołakowska & Landowska and the Maalej & Kallel 2020 review (EmoSurv authors): the most informative features are dwell (hold) time, flight (inter-key) latency, typing speed, pause distribution, and error/backspace rate; **between-person variance dwarfs within-person emotional variance**, so features must be z-scored per user. This confirms the brief's design.
- Honest expectation: public data is small (EmoSurv, 124 people). Real validation comes from your own data.

### Hierarchical emotion map (engine F, your "clusters that split into sub-clusters")
Current best practice (BERTopic-style pipeline, plus the GoEmotions paper's own label-hierarchy analysis):
1. Embed sentences with a sentence-transformer (MiniLM).
2. **UMAP** to ~5–10 dims (cosine metric). This preserves local structure and makes density clustering work.
3. **HDBSCAN** for the leaf clusters (it finds the number of clusters itself and marks noise). Its *condensed tree* is already a hierarchy.
4. For a clean, readable tree: **agglomerative clustering (Ward/average linkage) over HDBSCAN cluster centroids** gives the "big clusters that split into sub-clusters" dendrogram.
5. Name each node by its dominant GoEmotions labels + top c-TF-IDF words; evaluate with label purity / NMI against GoEmotions labels and stability across random seeds.
Output per message: nearest leaf + path to root, distances to the k nearest emotional neighbourhoods.

## 4. Recommendation (one page)

| Engine | Use for the MVP | Why | Commercial swap |
|---|---|---|---|
| A. Lexicon | **NRC VAD v2.1** (VAD) + **EmoLex** (discrete) + VADER rules (negation, intensifiers, "but") | Largest coverage, native [-1, 1], multi-word expressions, has ru/hy translations | NRC commercial licence |
| B. Transformer | Baseline `SamLowe/roberta-base-go_emotions`; build **our own DistilRoBERTa fine-tune on GoEmotions + VAD head on EmoBank** | Owned weights, ~2× faster on CPU | EmoBank share-alike check |
| C. Stylometric | spaCy + Empath + absolutist list + our features | Free, explainable, baseline-relative | — (all commercial-OK) |
| D. Keystroke | Our feature extractor + per-user z-scores; validate on EmoSurv | Matches literature | Train on consented user data |
| E. Appraisal | Claude API with strict JSON schema + evidence spans; local fallback = DistilRoBERTa classifier on ED situations → our 12-type taxonomy | LLM handles "what happened" far better than any small model; fallback keeps it working offline | Replace ED fallback data |
| F. Semantic map | MiniLM embeddings → UMAP → HDBSCAN → agglomerative over centroids, built on GoEmotions | Matches your clustering idea; commercial-clean data | — |
| Fusion | Stacking logistic regression over engine outputs, temperature scaling, ECE | Brief §4.6 | — |
| Temporal | EMA first, then Kalman on VAD; HMM only if metrics justify | Simple first | — |

**Language plan:** English via all engines. Russian via multilingual MiniLM + rubert-tiny2 + NRC-ru (Phase 6). Armenian via multilingual encoder + NRC-hy, flagged low-confidence.

## 5. Things you need to do

- [ ] **EmoSurv:** log in (free IEEE account) at the DataPort page in DATA_SOURCES.md, download the 4 CSVs, put them in `data/raw/emosurv/` (or upload them here and I'll place them). Needed by Phase 3 engine D, not before.
- [ ] **Anthropic API key** for engine E: needed in Phase 3 (I'll request it via the secure credential form, never chat).
- [ ] (Optional) ISEAR request from University of Geneva.
