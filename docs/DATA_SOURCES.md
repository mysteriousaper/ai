# Data sources

> Status: Phase 1, last updated 2026-11-18. Built from `docs/research/notes/R1-lexicons-datasets.md`,
> the models section of `R2-models-keystroke.md`, and `V1-verification-data-models.md`. **Where the notes
> disagree, V1 wins.** `V3-verification-keystroke-multilingual.md` did not exist when this was written; fold
> its corrections in when it lands.

Every dataset, lexicon and pretrained model the Engine touches gets a row. Licence class:

- `commercial-OK`: can ship in a paid product (attribution rules noted).
- `research-only`: fine for experiments; must be replaced before commercial use.
- `needs-licence`: commercial use requires a paid or negotiated licence.
- `unclear`: no licence found, or the stated licence does not cover the underlying data. Treat as `research-only` until resolved. (This class is used by the notes and V1; it is not in the brief's three-way split.)

Conventions used in the tables:

- **Licence (as stated)** quotes or closely paraphrases what the source says. Nothing here is a legal opinion.
- **Verified?** is the V1 verdict (`verified`, `corrected`, `unverified`). "unverified (R1/R2 only)" means a researcher read the source but V1 did not re-check it. "unverified" with no qualifier means only a search summary or secondary source exists.
- **Used by** is the planned engine from the brief (§4.4: A lexicon, B transformer, C stylometric, D keystroke, E situation/appraisal, F semantic clustering) or `eval`. It is a plan, not a measurement of what the code does today. No resource in the notes feeds engine C.
- **Access**: `auto` = scriptable download for `scripts/fetch_data.py` (not written yet) into `data/raw/<id>/`; `manual` = needs a human, see "Manual downloads for the owner".
- Marker **†** = trained on, or derived from, GoEmotions (Reddit text). See open legal question 1.

## 1. Lexicons

| Resource | Version | URL | Licence (as stated) | Class | Verified? | Used by | Access |
|---|---|---|---|---|---|---|---|
| NRC VAD Lexicon v2.1 (`nrc_vad_v2`) | 2.1 (Mar 2025); 54,801 terms, scale -1..1 | https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip (6,488,651 B) | "can be used freely for non-commercial research and educational purposes"; "No Redistribution: Do not redistribute the data"; "Commercial License: If interested in commercial use ... contact the author: saif.mohammad@nrc-cnrc.gc.ca" | needs-licence | verified | A (main VAD lexicon) | auto for research; manual for a commercial licence (email) |
| NRC VAD Lexicon v1 (`nrc_vad_v1`) | 1.0 (Jul 2018, README Aug 2022); >20k words | https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip (44,536,009 B) | Same NRC terms as v2.1 | needs-licence | corrected (scale: default file 0..1, but zip also has `BipolarScale/` at -1..1; pin the path) | A (ru/hy translations only) | auto for research; same commercial email. Holds machine-translated Russian and Armenian in `OneFilePerLanguage/` |
| NRC Emotion Lexicon, EmoLex (`nrc_emolex`) | 0.92 (2011, README Aug 2022); 14,182 unigrams; 8 Plutchik emotions + pos/neg, binary | https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip (25,878,449 B) | "freely available for research, and has been commercially licensed to companies for a small fee"; same Research Use / No Redistribution / Commercial License items | needs-licence | verified | A | auto for research; manual for commercial (email) |
| NRC Emotion Intensity Lexicon (`nrc_eil`) | 1 (Mar 2020); ~10k entries, 8 emotions, scores 0..1 | https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Intensity-Lexicon.zip (29,608,461 B) | Same NRC terms ("Research Use ... non-commercial research and educational purposes") | needs-licence | verified (URL and size; licence page not re-fetched by V1) | A (optional) | auto for research; manual for commercial (email) |
| Warriner, Kuperman & Brysbaert 2013 (`warriner2013`) | 2013; 13,915 lemmas, scale 1..9 | Mirror only: https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv (3,722,885 B); second mirror hwalicki/Twitter-Sentiment-Analysis. Official host `crr.ugent.be` is gone | Secondary source only (JULIELab README): "released ... under a Creative Commons Attribution-NonCommercial-NoDerivs 3.0 Unported License". Mirrors carry no LICENSE file | research-only | unverified (primary page 404; Wayback/Springer unreachable; 13,915 rows and scale verified) | A (VAD prior) | auto from the mirror (research only); manual to ask Brysbaert for commercial/derivative rights |
| ANEW (`anew`) | 1999; 1,034 words, 1..9 | https://csea.phhp.ufl.edu/media/anewmessage.html | Signed user agreement from UF CSEA; wording not retrieved | needs-licence | unverified (search summary; host unreachable) | A (historical; low value) | manual (form + agreement). Recommended: skip, Warriner/NRC cover it |
| RuSentiLex (`rusentilex`) | 2016/2017 (unverified) | https://www.labinform.ru/pub/rusentilex/index.htm (paper: https://aclanthology.org/L16-1186/) | Not found | unclear | unverified | A (Russian sentiment) | manual (browser download, record licence) |
| Chen & Skiena multilingual sentiment lexicons (`senti_lex_hy_ru`) | HF `senti-lex/senti_lex` sha fef91cf5...; hy 720 pos / 937 neg; ru 1,266 / 1,648 | https://huggingface.co/datasets/senti-lex/senti_lex/resolve/main/data.zip | GPL-3.0: "does not allow its incorporation into any type of distributed proprietary software ... For commercial applications please contact the dataset creators" | needs-licence | unverified (R1 only) | A (hy/ru sentiment; Armenian list is noisy) | auto; commercial: contact the creators |
| ArmenianNLP lexicons (`armeniannlp`) | unknown; repo not inspected | https://github.com/nigkal/ArmenianNLP | "Creative Commons Attribution-NonCommercial 4.0 International License" | research-only | unverified (README only) | A (hy sentiment/emotion lexicons, unchecked) | manual: clone to inspect (file list was not readable) |
| Russian VAD / affective norms (ENRuN, Solovyev ~1,000 words, Kazan 1,000 words) | papers only | Authors' pages (not fetched) | Not verified | unclear | unverified (search summaries) | A (ru VAD, if found) | manual (authors' pages) |

## 2. Datasets

| Resource | Version | URL | Licence (as stated) | Class | Verified? | Used by | Access |
|---|---|---|---|---|---|---|---|
| GoEmotions (`goemotions`) | 2020; 58,009 Reddit comments, 27 emotions + neutral; simplified splits 43,410 / 5,426 / 5,427; HF sha add49224...(2024-01-04) | Raw: https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_{1,2,3}.csv. Simplified: https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/{train,dev,test}.tsv, `emotions.txt`, `ekman_mapping.json`. HF: google-research-datasets/go_emotions | "The GitHub repository which houses this dataset has an Apache License 2.0." (repo level only; the README has no licence line; card warns it "includes the original usernames of the Reddit users") | unclear (R1 said commercial-OK; V1 corrected) | corrected | B (train/eval), eval | auto. Do not redistribute the text; raw files carry Reddit usernames |
| EmoBank (`emobank`) | 2017 (+2019 split, +2022 per-annotator ratings); 10,062 sentences, 8,062 / 1,000 / 1,000; scale 1..5 | https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/{emobank,reader,writer}.csv and `meta.tsv` | "This work is licensed under CC-BY-SA 4.0" | unclear (licence allows commercial use; unclear from ShareAlike and from 1,250 SemEval-2007 headline sentences) | verified | B (VAD head), eval | auto |
| EmpatheticDialogues (`empathetic_dialogues`) | 2019; ~25k conversations, 32 situation labels | https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz (28,022,709 B) | "Attribution-NonCommercial 4.0 International" (CC BY-NC 4.0) | research-only | verified | E (situation taxonomy, fallback classifier), F | auto |
| DailyDialog (`dailydialog`) | 2017; 13,118 dialogues, 7 emotion labels | https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/train.zip (data-only copy; original `yanran.li` unreachable) | "DailyDialog dataset is licensed under CC BY-NC-SA 4.0." | research-only | verified | eval | auto via the roskoN copy |
| SemEval-2018 Task 1, Affect in Tweets (`semeval2018_task1`) | 2018; E-c English 6,838 / 886 / 3,259 | https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip (5,975,590 B) | NRC terms: "Do not redistribute the data"; "If interested in commercial use ... A nominal one-time licensing fee may apply." Tweets also fall under Twitter/X terms (R1 inference) | research-only (NRC commercial licence possible) | verified | eval (intensity) | auto for research; commercial by email |
| ISEAR (`isear`) | Scherer & Wallbott; 7,666 reports, 7 emotions | https://www.unige.ch/cisa/research/materials-and-online-research/research-material/ | Not found (page returns Access Denied to the container). HF copies tagged apache-2.0 are unofficial relicences, ignore | unclear | unverified | eval (optional) | manual |
| EmoSurv (`emosurv`) | IEEE DataPort DOI 10.21227/eae6-pk42; 124 participants, 4 CSVs, 5 labels | https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using | Search-summary paraphrase only: non-exclusive, non-transferable, revocable licence "solely for non-commercial, educational, and research purposes" | research-only | unverified (host unreachable; no verbatim text) | D (prior and evaluation), eval | manual |
| CEDR (`cedr`) | v0.0.1; ~9.4k sentences (HF) / 9,668 (README); 5 emotions + none | https://huggingface.co/datasets/sagteam/cedr_v1 (repo https://github.com/sag111/CEDR) | Repo Apache-2.0; HF card: "The GitHub repository which houses this dataset has an Apache License 2.0." Texts come from LiveJournal, Lenta.ru and Twitter | unclear | unverified (R1 only; V1 did not check) | B/eval (Russian) | auto |
| ru_go_emotions (`ru_go_emotions`) | HF seara/ru_go_emotions sha 0323e396...; Google-Translate of GoEmotions | https://huggingface.co/datasets/seara/ru_go_emotions | HF tag `mit` (self-declared by the translator). Alternatives: Djacon/ru_goemotions (mit), AiLab-IMCS-UL/go_emotions-ru (apache-2.0) | unclear | unverified (R1 only) | B (Russian) † | auto |
| Armenian news sentiment (`armenian_news_sentiment`) | HF Elen22/Armenian-news-sentiment-dataset sha eb847abe...; test 956 news articles, 3-class | https://huggingface.co/datasets/Elen22/Armenian-news-sentiment-dataset | HF tag apache-2.0; train/dev machine-translated from SST, MELD, SLSD (search summary) | unclear | unverified (card gated) | eval (Armenian, optional) | manual (HF login, accept conditions) |
| OpenSLR SLR153 Armenian news text | not recorded | OpenSLR | CC BY-SA 4.0 (search summary) | unclear | unverified | F (unlabelled pretraining/annotation text only) | not fetched |

Related but not assessed: `cardiffnlp/tweet_eval` (emotion config, HF licence "unknown"), `dair-ai/emotion` (licence tag "other").

## 3. Pretrained models

### 3.1 Emotion classifiers

| Resource | Version | URL | Licence (as stated) | Class | Verified? | Used by | Access |
|---|---|---|---|---|---|---|---|
| SamLowe/roberta-base-go_emotions (+ `-onnx`) | roberta-base, 124.7M, 28 labels multi-label; macro-F1 0.450 @0.5, 0.541 tuned on test (optimistic), 0.519 tuned on validation (third-party) | https://huggingface.co/SamLowe/roberta-base-go_emotions ; https://huggingface.co/SamLowe/roberta-base-go_emotions-onnx | `license: mit` | commercial-OK † | verified | B (default English baseline) | auto |
| Horizon-Labs/multilingual-emotions-base | mmBERT-base, 307.6M; GoEmotions test 0.473 @0.5 / 0.516 tuned on val; BRIGHTER-6 0.420; Russian 0.657 (all self-reported); no Armenian | https://huggingface.co/Horizon-Labs/multilingual-emotions-base | Apache-2.0 (LICENSE + NOTICE; mmBERT base is MIT) | commercial-OK † | verified (self-reported metrics; low trust) | B (en/ru) | auto. Org is new (first model 2026-09-23), base has ~330 downloads: re-test on our own split |
| Horizon-Labs/multilingual-emotions-small | mmBERT-small, 140.7M; 0.447 @0.5 / 0.494 tuned | https://huggingface.co/Horizon-Labs/multilingual-emotions-small | Apache-2.0 (+ MIT notice) | commercial-OK † | verified (self-reported) | B (en/ru, smaller) | auto |
| cirimus/ultrabert-380M-go-emotions | 377.6M; 0.521 @0.5 / 0.568 tuned on train thresholds (self-reported) | https://huggingface.co/cirimus/ultrabert-380M-go-emotions | Apache-2.0 | commercial-OK † | verified (self-reported) | B (optional, English quality option) | auto. Needs `trust_remote_code=True` and transformers>=5.2: audit the remote code first |
| cirimus/ultrabert-150M-go-emotions | 151.1M; metrics not extracted | https://huggingface.co/cirimus/ultrabert-150M-go-emotions | Apache-2.0 | commercial-OK † | unverified (R2 card only) | B (optional) | auto; same `trust_remote_code` caveat |
| cirimus/modernbert-large-go-emotions | 395.9M; 0.490 @0.5 / 0.543 tuned (card says superseded by UltraBERT-380M) | https://huggingface.co/cirimus/modernbert-large-go-emotions | MIT | commercial-OK † | unverified (R2 card only) | B (optional) | auto |
| fyaronskiy/ModernBERT-large-english-go-emotions | 395.9M; 0.52 tuned on validation; ONNX documented | https://huggingface.co/fyaronskiy/ModernBERT-large-english-go-emotions | MIT | commercial-OK † | unverified (R2 card only) | B (optional) | auto |
| minuva/MiniLMv2-goemotions-v2 | 30.3M; metrics not read | https://huggingface.co/minuva/MiniLMv2-goemotions-v2 | Apache-2.0 | commercial-OK † | unverified (R2 card only) | B (small candidate) | auto |
| fyaronskiy/ruRoberta-large-ru-go-emotions | 355.4M; 0.41 @0.5 / 0.48 tuned on val (self-reported); trained on seara/ru_go_emotions | https://huggingface.co/fyaronskiy/ruRoberta-large-ru-go-emotions | MIT | commercial-OK † (training data is a machine-translated, MIT-tagged copy of GoEmotions) | verified (self-reported) | B (Russian) | auto |
| seara/rubert-tiny2-ru-go-emotions | 29.2M; macro-F1 0.33 @0.5 (several labels 0.00) | https://huggingface.co/seara/rubert-tiny2-ru-go-emotions | MIT | commercial-OK † | unverified (R2 card only) | B (Russian, tiny/fast) | auto |
| seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions | 177.9M; macro-F1 0.36 (from the fyaronskiy table) | https://huggingface.co/seara/rubert-base-cased-russian-emotion-detection-ru-go-emotions | MIT | commercial-OK † | unverified (R2 only) | B (Russian) | auto |
| cointegrated/rubert-tiny2-cedr-emotion-detection | 29.2M; CEDR labels; macro-F1 0.8348 on an easy 6-label task | https://huggingface.co/cointegrated/rubert-tiny2-cedr-emotion-detection | No licence on the card or HF tags | unclear | unverified (R2 card only) | B/eval (Russian) | auto |
| j-hartmann/emotion-english-distilroberta-base (and `-roberta-large`) | Ekman 6 + neutral, single-label; 66% accuracy on its own held-out mix | https://huggingface.co/j-hartmann/emotion-english-distilroberta-base | None stated (no `license:` tag). Trained on a mix incl. ISEAR, MELD, SemEval-2018, GoEmotions | unclear | verified (no licence confirmed) | B (comparison baseline only) | auto, research use |
| MilaNLProc/xlm-emo-t | 4 labels (anger, fear, joy, sadness); ~278M | https://huggingface.co/MilaNLProc/xlm-emo-t | No licence field; card: "intended as a research output ... restrictions may apply to the data ... or Twitter" | research-only | unverified (R2 card only) | eval (multilingual comparison) | auto, research use |
| tabularisai/multilingual-emotion-classification | XLM-R base, 278.05M, 11 labels, synthetic LLM training data | https://huggingface.co/tabularisai/multilingual-emotion-classification | HF tag `mit` (card modified 2026-11-03). The CC-BY-NC-4.0 claim comes only from Horizon's comparison table. Synthetic-data generator terms unknown | unclear | corrected (MIT tag and 278.05M params confirmed; NC claim not reproduced) | none planned | auto, research use |
| AnasAlokla/multilingual_go_emotions (V1.2) | mBERT, 177.9M; ar/en/fr/es/nl/tr, no ru or hy | https://huggingface.co/AnasAlokla/multilingual_go_emotions | MIT | unclear (training-data licence not checked) | unverified (R2 only) | none planned | auto |

### 3.2 VAD regression

| Resource | Version | URL | Licence (as stated) | Class | Verified? | Used by | Access |
|---|---|---|---|---|---|---|---|
| BenRongey/deberta-v3-base-emobank-vad | LoRA on deberta-v3-base; Pearson V .774 / A .572 / D .383 | https://huggingface.co/BenRongey/deberta-v3-base-emobank-vad | `license: mit`; card says trained "on the None dataset" (name implies EmoBank, CC BY-SA 4.0) | unclear | corrected: A and D MAE (0.191 / 0.175) are no better than predicting the EmoBank mean (0.189 / 0.156). **Do not use**, not even as a baseline | none | do not fetch |

### 3.3 Sentence embedders (engine F, and frozen features for the VAD head)

| Resource | Version | URL | Licence (as stated) | Class | Verified? | Used by | Access |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 117.7M, 384-d, 512 tokens, 100 languages incl. hy and ru, ONNX; needs `query: ` / `passage: ` prefixes | https://huggingface.co/intfloat/multilingual-e5-small | `license: mit` | commercial-OK | verified | F (default), B (VAD head features) | auto |
| sentence-transformers/all-MiniLM-L6-v2 | 22.7M, 384-d, English only, ONNX | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | `license: apache-2.0` | commercial-OK (training pairs include Reddit, not itemised) | verified | F (English fast path) | auto |
| intfloat/multilingual-e5-base | 278.0M, 768-d | https://huggingface.co/intfloat/multilingual-e5-base | MIT | commercial-OK | unverified (R2 card only) | F (offline quality check) | auto |
| BAAI/bge-m3 | ~568M, 1024-d, 8192 tokens | https://huggingface.co/BAAI/bge-m3 | MIT | commercial-OK | unverified (R2 only; hy unconfirmed) | F (quality option) | auto |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 117.7M, hy and ru listed | https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache-2.0 | commercial-OK | unverified (R2 only) | F (alternative) | auto |
| sentence-transformers/LaBSE | 470.9M, 109 languages | https://huggingface.co/sentence-transformers/LaBSE | Apache-2.0 | commercial-OK | unverified (R2 only) | F (cross-lingual matching, optional) | auto |
| sentence-transformers/all-mpnet-base-v2 | 109.5M, English | https://huggingface.co/sentence-transformers/all-mpnet-base-v2 | Apache-2.0 | commercial-OK | unverified (R2 only) | F (optional) | auto |
| Metric-AI/armenian-text-embeddings-2-base | 278M, hy/en, fine-tune of e5-base on Armenian Reddit-derived data | https://huggingface.co/Metric-AI/armenian-text-embeddings-2-base | MIT (model); dataset licence not checked | unclear | unverified (R2 only) | F (Armenian, optional) | auto |
| google/embeddinggemma-300m | 302.9M; gated, card not read | https://huggingface.co/google/embeddinggemma-300m | Gemma terms (HF tag `license:gemma`) | needs-licence | unverified | none (skip) | manual (gated) |
| jinaai/jina-embeddings-v3 | 572.3M | https://huggingface.co/jinaai/jina-embeddings-v3 | CC-BY-NC-4.0 (HF tag) | research-only | unverified (R2 only) | none (excluded) | do not use |

Other embedders R2 saw but did not read (Qwen3-Embedding-0.6B Apache-2.0, multilingual-e5-large-instruct MIT, snowflake-arctic-embed-m-v2.0 Apache-2.0) are not in the plan; add rows if one is adopted.

### 3.4 Encoders for our own fine-tunes

XLM-R (`hy` in its 100 languages) and mmBERT (`hye` in its 1,833-language list; MIT) saw Armenian in pretraining and are the candidate bases for an Armenian fine-tune on our own data. No Armenian emotion classifier exists in the notes. Licences were not separately verified; add rows when one is chosen.

## 4. Pinned scales (loaders must not guess)

| Resource | Scale |
|---|---|
| NRC VAD v2.1 | -1..1 (verified min/max) |
| NRC VAD v1 | `NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt` is 0..1; `BipolarScale/NRC-VAD-Lexicon.txt` is -1..1. Pin the file path |
| Warriner 2013 | 1..9 (columns `V.Mean.Sum`, `A.Mean.Sum`, `D.Mean.Sum`) |
| EmoBank | 1..5 (observed V 1.2-4.6, A 1.8-4.4, D 1.78-4.2); `emobank.csv` has `id,split,V,A,D,text` |
| ANEW | 1..9 |
| NRC Emotion Intensity, SemEval-2018 EI-reg and V-reg | 0..1 |

## 5. Must replace before commercial use

Every `research-only`, `needs-licence` or `unclear` item. "No replacement in notes" means the research notes name none; do not assume one exists.

| Item | Class | Why | Candidate replacement (from the notes) |
|---|---|---|---|
| NRC VAD v2.1 / v1, EmoLex, Emotion Intensity | needs-licence | Free for non-commercial research only; no redistribution; shipping in a product, or models trained on them, needs a commercial licence. Price not published ("a small fee" per EmoLex page) | Buy the NRC licence (email below). No open VAD or emotion lexicon replacement in the notes: Warriner is also non-commercial |
| NRC ru/hy translations (inside the v1 zip and EmoLex) | needs-licence | Machine translations of NRC entries count as NRC derivatives | Include in the same NRC licence request. No replacement in notes |
| Warriner 2013 | research-only | Reported CC BY-NC-ND 3.0 (secondary source): NC, and ND bars derivative or rescaled lexicons. Mirrors have no LICENSE | Ask Brysbaert for commercial/derivative permission, or NRC VAD under a licence. No replacement in notes |
| ANEW | needs-licence | Signed user agreement, commercial terms unknown | Skip (superseded by Warriner/NRC) |
| SemEval-2018 Task 1 | research-only | NRC terms, no redistribution; Twitter/X terms; "nominal one-time licensing fee may apply" | Keep as eval only, or licence from NRC |
| EmpatheticDialogues | research-only | CC BY-NC 4.0. Feeds engine E's situation taxonomy and fallback classifier | No replacement in notes. Option: derive the taxonomy from appraisal theory plus our own annotated data (Phase 5) |
| DailyDialog | research-only | CC BY-NC-SA 4.0 | Eval only; no replacement needed |
| EmoSurv | research-only | Non-commercial licence (unverified paraphrase) | None. The brief already expects real validation of engine D from the owner's own data in Phase 5 |
| ArmenianNLP lexicons | research-only | CC BY-NC 4.0 | Own annotated Armenian data (R1's conclusion) |
| Chen & Skiena sentiment lexicons | needs-licence | GPL-3.0 cannot be incorporated into distributed proprietary software | Contact the creators, or exclude |
| xlm-emo-t | research-only | "Research output"; Twitter-derived training data | Horizon-Labs multilingual-emotions-base/-small |
| jina-embeddings-v3 | research-only | CC-BY-NC-4.0 | multilingual-e5-small |
| google/embeddinggemma-300m | needs-licence | Gemma terms, gated, not reviewed | multilingual-e5-small |
| GoEmotions (corpus) | unclear | Apache-2.0 covers Google's labels, not the Reddit comment text; raw files carry usernames | Train on it, do not redistribute text; get legal sign-off. Long-term replacement: own labelled data (no specific open corpus named in the notes) |
| EmoBank | unclear | CC BY-SA 4.0 ShareAlike; 1,250 of 10,062 sentences are SemEval-2007 headlines of uncertain rights | Use only the MASC subset (`meta.tsv`), pending legal view |
| ISEAR | unclear | No licence found | Treat as research-only; eval at most |
| CEDR | unclear | Apache-2.0 on the repo, but texts scraped from LiveJournal, Lenta.ru, Twitter | None in notes; eval only until reviewed |
| ru_go_emotions and other Russian GoEmotions copies | unclear | MIT/Apache tags self-declared by translators; derivative of Reddit text via Google Translate | Same as GoEmotions |
| RuSentiLex, Russian VAD norms | unclear | Licence not found | Check on the authors' pages (manual) |
| Armenian news sentiment (Elen22), OpenSLR SLR153 | unclear | Train/dev machine-translated from research datasets (SST, MELD, SLSD); SLR153 unlabelled | Eval only, or own Armenian data |
| j-hartmann/emotion-english-* | unclear | No licence; trained on a mix incl. ISEAR, MELD, SemEval | SamLowe/roberta-base-go_emotions (MIT) |
| cointegrated/rubert-tiny2-cedr-emotion-detection | unclear | No licence on the model; CEDR provenance | seara/rubert-tiny2-ru-go-emotions or fyaronskiy/ruRoberta-large (MIT) |
| tabularisai/multilingual-emotion-classification | unclear | MIT tag, but synthetic-data generator terms unknown; third-party table claims CC-BY-NC | Horizon-Labs models |
| BenRongey/deberta-v3-base-emobank-vad | unclear (and unusable) | Undocumented data; A and D no better than a constant | Train our own head on frozen e5 embeddings + EmoBank (rows above), evaluate with CCC |
| AnasAlokla/multilingual_go_emotions, Metric-AI Armenian embedder | unclear | Training-data licence not checked | multilingual-e5-small |

All GoEmotions-trained classifiers (marked †) have `commercial-OK` model licences but inherit the Reddit question below. They are not on the replace list as models, but the whole B-engine classifier stack depends on question 1.

## 6. Open legal questions

These need a lawyer or the rights holder; the notes do not resolve them.

1. **GoEmotions Reddit text (highest leverage).** Only a repository-level Apache-2.0 statement exists. Google can license its labels, not Reddit users' comments, and the raw files include usernames. Training on the labels is common practice, and each model's own MIT/Apache licence stands, but every recommended classifier (SamLowe, Horizon-Labs, cirimus, fyaronskiy, seara) is trained on this text. Get legal sign-off before selling; do not redistribute GoEmotions text.
2. **EmoBank ShareAlike.** CC BY-SA 4.0 permits commercial use, but does ShareAlike reach trained model weights or derived datasets? Also, 1,250 sentences are SemEval-2007 news headlines (rights unverified). Fallback: train only on the MASC subset identified by `meta.tsv`.
3. **NRC commercial terms.** Price is unpublished for VAD; EmoLex says "a small fee"; labelled data says "a nominal one-time licensing fee may apply". Ask: price; whether model weights trained on the lexicons count as redistribution; whether the Russian and Armenian machine translations are covered; the credit/About-page requirement. Until answered, NRC stays research-only in practice and behind the swappable-resource interface.
4. **Warriner's licence.** The CC BY-NC-ND 3.0 claim comes only from the JULIELab README. The original page `crr.ugent.be/archives/1003` is 404 and the site is "under reconstruction"; the Wayback Machine and Springer were not reachable. The mirrors have no LICENSE file, so redistributing from them is itself unlicensed, and ND may forbid a rescaled or merged prior. Class stays `research-only` (stricter reading).

Also flagged by V1 (not in the owner's list but unresolved): EmoSurv's licence text is unread (paraphrase only); Horizon-Labs and cirimus models rest on self-reported metrics from very new authors; cirimus needs `trust_remote_code`.

## 7. Manual downloads for the owner

Put every file in `data/raw/<id>/` (git-ignored; `<id>` as given in the tables). Record the licence text you see in this file when you download.

1. **EmoSurv** (`data/raw/emosurv/`). Needed for engine D.
   1. Go to https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using (DOI 10.21227/eae6-pk42).
   2. Log in with a free IEEE account (no paid membership needed).
   3. Accept the dataset licence and **copy its exact text into the EmoSurv row above** (currently only a paraphrase).
   4. Download the 4 CSVs (fixed-text typing, free-text typing, frequency features, participant info) into `data/raw/emosurv/`.
2. **NRC commercial licence** (no download; only if shipping NRC data). Email saif.mohammad@nrc-cnrc.gc.ca covering VAD v2.1, EmoLex, Emotion Intensity and SemEval-2018 data embedded in a sold product. Ask the four questions in open question 3. The EmoLex page also links an NRC shop (shop-magasin.nrc-cnrc.gc.ca); it was not reachable, check it in a browser.
3. **Warriner permission** (optional). Write to Marc Brysbaert (Ghent University; address on https://crr.ugent.be/) for commercial and derivative permission. Optionally, if you can reach it, read the primary licence on the Springer paper (doi 10.3758/s13428-012-0314-x) or the Wayback copy of `crr.ugent.be/archives/1003` and record it. The research copy comes from the mirror via the fetch script into `data/raw/warriner2013/`.
4. **ISEAR** (`data/raw/isear/`). Open https://www.unige.ch/cisa/research/materials-and-online-research/research-material/ in a browser, download the ISEAR databank, copy the stated terms into this file.
5. **RuSentiLex** (`data/raw/rusentilex/`). Open https://www.labinform.ru/pub/rusentilex/index.htm in a browser, download the lexicon, record its licence.
6. **Armenian news sentiment** (`data/raw/armenian_news_sentiment/`; optional). Log in to Hugging Face, open https://huggingface.co/datasets/Elen22/Armenian-news-sentiment-dataset and accept the share-contact-information condition, then download.
7. **ArmenianNLP** (`data/raw/armeniannlp/`; optional). `git clone https://github.com/nigkal/ArmenianNLP` into that folder, record the lexicon file list and sizes (not inspected so far), and confirm the CC BY-NC 4.0 licence applies to the data folders.
8. **ANEW** (`data/raw/anew/`; low priority, recommended skip). Submit the ANEW Researcher Information Form on https://csea.phhp.ufl.edu/media/anewmessage.html, sign the User Agreement, and ask in writing whether commercial use is allowed.
9. **Russian VAD norms** (optional). Look up ENRuN (Lyusin & Sysoeva 2025, Experimental Psychology) and the Solovyev / Kazan 1,000-word norms on the authors' pages; record licences. The notes give conflicting sizes for ENRuN (378 vs 1,800 nouns).
10. **Audit before use:** `cirimus/ultrabert-*` requires `trust_remote_code=True`. Read the remote Python before running it.

Everything marked `auto` is fetched by `scripts/fetch_data.py` (to be written) into `data/raw/<id>/`. Verified sizes (HEAD checks) are in the tables for integrity checks.
