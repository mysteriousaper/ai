# Engine A (lexicon) evaluation

## EmoBank test (n=1000), 95% bootstrap CI (500 resamples)

| Dim | Model | Pearson r | CCC | MAE |
|---|---|---|---|---|
| valence | Lexicon | 0.523 [0.471, 0.578] | 0.364 [0.320, 0.406] | 0.231 [0.220, 0.241] |
| valence | VADER compound (baseline) | 0.458 [0.407, 0.504] | 0.331 [0.288, 0.372] | 0.256 [0.241, 0.269] |
| arousal | Lexicon | 0.244 [0.179, 0.309] | 0.201 [0.143, 0.260] | 0.169 [0.160, 0.177] |
| dominance | Lexicon | 0.131 [0.069, 0.195] | 0.107 [0.056, 0.162] | 0.162 [0.154, 0.170] |
| valence | Constant 0 (baseline) | nan [nan, nan] | 0.000 [0.000, 0.000] | 0.119 [0.112, 0.128] |

## GoEmotions test, first 2000 rows, Ekman-6 macro F1 (threshold 0.3, multi-label)

| Model | Macro F1 [95% CI] |
|---|---|
| Lexicon (EmoLex) | 0.245 [0.229, 0.259] |
| Majority class ('joy' always on) | 0.093 [0.089, 0.097] |

## Verdict

- EmoBank valence: lexicon r=0.523 beats VADER r=0.458 (see CI overlap above).
- Caveat: the constant-0 row shows raw MAE is dominated by EmoBank's narrow label range; rank by r/CCC.
- GoEmotions Ekman-6: lexicon F1=0.245 beats majority baseline F1=0.093.
