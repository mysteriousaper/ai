# V3 - Verification: keystroke literature, embedders, ISEAR/ANEW/SemEval/DailyDialog, Russian and Armenian resources

Verifier agent, 2026-11-18. Status: IN PROGRESS (skeleton written first in case of container restart).

Scope: R2 Part B (embedder licences, hy/ru coverage), R2 Part C (keystroke literature); R1 §1 ANEW, §8 ISEAR /
DailyDialog / SemEval-2018, §10 Russian, §11 Armenian.

## Verdict summary
(pending)

## Claim table
(pending)

## Still needs a human
(pending)

## Progress log (raw, superseded by tables above when final)
- Network: arxiv, ACM, PMC, MDPI, PLoS, medRxiv, Zenodo, IEEE DataPort, CSEA, labinform, aclanthology, codalab, yanran.li,
  web.archive.org all blocked (curl 000 / WebFetch ENOTFOUND). Reachable: huggingface.co, raw.githubusercontent.com,
  saifmohammad.com, unige.ch (403 bot wall). WebSearch works (snippet grade).
- Embedders: HF API licence/params/hy/ru tags/ONNX/max_seq checked; tokenizer spike shows 0 UNK for hy/ru in e5-small,
  paraphrase-multilingual, LaBSE, arctic-m-v2.0; all-MiniLM-L6-v2 produces [UNK] for Armenian.
- Metric-AI/reddit_10K_hye (training data of the Armenian embedder) is CC-BY-NC-4.0 and Gemma-2-27B translated.
- ANEW: quanteda doc reproduces CSEA statement "will not provide the ANEW materials to profit making companies" -> research-only.
- RuSentiLex: rulexicon R doc: CC BY-NC-SA 4.0 per author -> research-only (secondary).
- SemEval-2018 zip: HEAD 200 5,975,590 B; README "Do not redistribute"; E-c En 6,838/886/3,259 verified.
- DailyDialog roskoN: 11,118/1,000/1,000 verified; train labels 82.8% "no emotion".
- Elen22 Armenian card readable (not gated): 956 test (295/476/185) verified; train/dev Google-Translated SST, MELD, SLSD.
- Kolakowska dataset repo: 49 participants x 5 vectors, 51 features, PAD 9-point, pleasure/arousal REVERSED (low = positive/aroused); no LICENSE file.
