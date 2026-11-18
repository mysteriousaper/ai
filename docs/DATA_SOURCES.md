# Data sources

Every resource the Engine touches. Fetch with `uv run python scripts/fetch_data.py`; checksums are written to `data/raw/MANIFEST.json`.

**Status key**
- `commercial-OK`: free to use in a paid product (keep attribution).
- `share-alike`: commercial use allowed, but derived *data* must be shared under the same licence (CC-BY-SA). Trained model weights are a grey area, so take legal advice before selling.
- `needs-licence`: research use free, commercial use needs a paid or negotiated licence.
- `research-only`: non-commercial only. Must be **replaced or retrained without it** before any commercial launch.

## Datasets and lexicons

| Name | What | Size (verified) | Licence | Status | Local path | URL |
|---|---|---|---|---|---|---|
| GoEmotions (simplified + raw) | Reddit comments, 27 emotions + neutral, multi-label | 43,410 / 5,426 / 5,427 (train/val/test); raw 211,225 ratings | Apache-2.0 | **commercial-OK** | `data/raw/go_emotions/` | https://huggingface.co/datasets/google-research-datasets/go_emotions |
| EmoBank | Sentences with V/A/D (1–5 scale), writer + reader perspective | 10,062 | CC-BY-SA 4.0 | **share-alike** | `data/raw/emobank/` | https://github.com/JULIELab/EmoBank |
| NRC VAD Lexicon v2.1 (2025) | Words + multi-word expressions → V/A/D in [-1, 1]; auto-translations into 100+ languages incl. Armenian, Russian | 54,801 terms | NRC terms: free for non-commercial research and education; no redistribution | **needs-licence** (NRC sells commercial licences) | `data/raw/nrc/NRC-VAD-Lexicon-v2.1/` | https://saifmohammad.com/WebPages/nrc-vad.html |
| NRC Emotion Lexicon (EmoLex) v0.92 | Word → 8 Plutchik emotions + pos/neg; translations incl. Armenian, Russian | ~14k words | Same NRC terms | **needs-licence** | `data/raw/nrc/NRC-Emotion-Lexicon/` | https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm |
| Warriner et al. 2013 (via JULIE Lab XANEW) | V/A/D norms, 1–9 scale | 13,915 lemmas | CC BY-NC-ND 3.0 | **research-only** | `data/raw/warriner/` | https://github.com/JULIELab/XANEW |
| EmpatheticDialogues | Conversations grounded in 32 emotion situations, with situation prompt | 84k / 12k / 11k utterance rows | CC BY-NC 4.0 | **research-only** | `data/raw/empathetic_dialogues/` | https://github.com/facebookresearch/EmpatheticDialogues |
| DailyDialog | Daily dialogues, 7 emotion labels per turn + dialogue acts | 11,118 / 1,000 / 1,000 dialogues | CC BY-NC-SA 4.0 | **research-only** | `data/raw/daily_dialog/` | https://huggingface.co/datasets/li2017dailydialog/daily_dialog |
| SemEval-2018 Task 1 (Affect in Tweets) | Emotion intensity (EI-reg/oc), valence (V-reg), multi-label (E-c) | EN/AR/ES | Task terms: research only, no redistribution | **research-only** | `data/raw/semeval2018/` | https://competitions.codalab.org/competitions/17751 |
| VADER lexicon | ~7.5k tokens → valence, plus heuristics | — | MIT | **commercial-OK** | bundled in `vaderSentiment` | https://github.com/cjhutto/vaderSentiment |
| Empath | 194 psycholinguistic categories (open LIWC alternative) | — | MIT | **commercial-OK** | bundled in `empath` | https://github.com/Ejhfast/empath-client |
| Absolutist word list (Al-Mosaiwi & Johnstone 2018) | 19 words | — | Published word list (facts, not copyrightable) | **commercial-OK** | `engine/engines/stylometric.py` (Phase 3) | https://doi.org/10.1177/2167702617747074 |
| EmoSurv | Keystroke timings, 124 participants, 5 emotions | 4 CSVs | IEEE DataPort open-access: non-commercial, no redistribution | **research-only** | `data/raw/emosurv/` (**manual**) | https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using |
| ISEAR | Self-reported emotional events, 7 emotions | ~7.6k | Unclear, available on request | **needs-licence** (optional, not fetched) | `data/raw/isear/` (**manual**) | https://www.unige.ch/cisa/research/materials-and-online-research/research-material/ |
| ANEW (Bradley & Lang 1999) | 1,034 words V/A/D | — | Request-only from CSEA | **skipped** (superseded by Warriner/NRC) | — | https://csea.phhp.ufl.edu/media.html |

## Pretrained models

| Model | Labels | Reported metric | Licence | Status | Role |
|---|---|---|---|---|---|
| `SamLowe/roberta-base-go_emotions` | 28 GoEmotions | test F1 0.450 (thr 0.5); higher with per-label thresholds (model card) | MIT | **commercial-OK** | Engine B off-the-shelf baseline |
| `j-hartmann/emotion-english-distilroberta-base` | Ekman 6 + neutral | 66% accuracy on its balanced eval (model card) | **none stated** | **needs-licence** | Comparison only; trained partly on research-only sets |
| `distilroberta-base` | backbone | — | Apache-2.0 | **commercial-OK** | Our fine-tune (GoEmotions + VAD head) |
| `sentence-transformers/all-MiniLM-L6-v2` | embeddings | — | Apache-2.0 | **commercial-OK** | Engine F (English) |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | embeddings, 50+ langs | — | Apache-2.0 | **commercial-OK** | Engine F multilingual; ru/hy later |
| `Djacon/rubert-tiny2-russian-emotion-detection` | ru emotions | card reports F1 micro/macro per label | MIT | commercial-OK | Russian candidate (not fetched yet) |
| `cointegrated/rubert-tiny2-cedr-emotion-detection` | ru, CEDR | card F1 macro ≈ 0.83 | none stated | needs-licence | Russian comparison (not fetched yet) |

## What must be replaced before commercial use

1. **NRC VAD + EmoLex:** buy NRC's commercial licence (the most practical option), or fall back to VADER valence plus a lexicon we derive ourselves.
2. **Warriner:** drop it. NRC VAD v2.1 covers it; we keep Warriner for research comparison only.
3. **EmpatheticDialogues** (situation classifier training) and **DailyDialog / SemEval** (evaluation): retrain the situation fallback on commercially licensed or self-collected data. Using them for evaluation-only during R&D is fine.
4. **EmoSurv:** used only to validate the keystroke feature extractor. The product model trains on consented user data.
5. **EmoBank** (share-alike): get legal advice before shipping weights trained on it, or regress VAD from GoEmotions labels + licensed NRC instead.

The architecture keeps every lexicon and model behind a `language` + `pack` key (see ARCHITECTURE.md), so swapping a research-only resource for a commercial one changes config, not code.
