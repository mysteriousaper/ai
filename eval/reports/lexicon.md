# Engine A (lexicon) evaluation

## EmoBank test (n=1000), 95% bootstrap CI (500 resamples)

| Dim | Model | Pearson r | CCC | MAE |
|---|---|---|---|---|
| valence | Lexicon | 0.525 [0.472, 0.576] | 0.366 [0.324, 0.409] | 0.233 [0.222, 0.244] |
| valence | VADER compound (baseline) | 0.458 [0.407, 0.504] | 0.331 [0.288, 0.372] | 0.256 [0.241, 0.269] |
| arousal | Lexicon | 0.248 [0.179, 0.312] | 0.203 [0.142, 0.259] | 0.171 [0.161, 0.180] |
| dominance | Lexicon | 0.131 [0.070, 0.196] | 0.106 [0.057, 0.160] | 0.164 [0.156, 0.172] |
| valence | Constant 0 (baseline) | nan [nan, nan] | 0.000 [0.000, 0.000] | 0.119 [0.112, 0.128] |

## GoEmotions test, first 2000 rows, Ekman-6 macro F1 (threshold 0.3, multi-label)

| Model | Macro F1 [95% CI] |
|---|---|
| Lexicon (EmoLex) | 0.245 [0.230, 0.260] |
| Majority class ('joy' always on) | 0.093 [0.089, 0.097] |

## Verdict

- EmoBank valence: lexicon r=0.525 beats VADER r=0.458 (see CI overlap above).
- Caveat: the constant-0 row shows raw MAE is dominated by EmoBank's narrow label range; rank by r/CCC.
- GoEmotions Ekman-6: lexicon F1=0.245 beats majority baseline F1=0.093.
