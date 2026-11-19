# Engine B (transformer v0: SamLowe/roberta-base-go_emotions) evaluation

## GoEmotions test (n=5427, 28 labels), 95% bootstrap CI (200 resamples), backend=onnx

| Model | Macro F1 | Micro F1 | ROC-AUC macro |
|---|---|---|---|
| Transformer @0.5 | 0.446 [0.432, 0.459] | 0.585 [0.575, 0.595] | 0.928 [0.915, 0.936] |
| Transformer tuned thresholds | 0.522 [0.499, 0.540] | 0.599 [0.588, 0.608] | 0.928 [0.915, 0.936] |
| Majority ('neutral' always) | 0.018 [0.017, 0.018] | 0.304 [0.292, 0.314] | 0.500 (constant) |

## Ekman-6 macro F1 (first 2000 test rows, project with to_ekman, threshold 0.3)

| Model | Macro F1 |
|---|---|
| Transformer | 0.628 |
| Lexicon (from eval/reports/lexicon.md) | 0.245 [0.229, 0.259] |
| Majority class (lexicon report) | 0.093 |

## EmoBank test (n=1000), VAD from ridge on 28 probabilities

| Dim | Pearson r | CCC | MAE |
|---|---|---|---|
| valence | 0.619 [0.563, 0.672] | 0.546 [0.490, 0.597] | 0.097 [0.091, 0.102] |
| arousal | 0.362 [0.306, 0.415] | 0.247 [0.198, 0.289] | 0.087 [0.083, 0.091] |
| dominance | 0.385 [0.313, 0.442] | 0.217 [0.174, 0.256] | 0.073 [0.069, 0.077] |
| (lexicon valence, for reference) | 0.523 [0.471, 0.578] | 0.364 | 0.231 |

## Latency per single message, CPU, 100 GoEmotions test messages

| Backend | mean ms | p95 ms |
|---|---|---|
| onnx | 10.2 | 17.4 |
| torch | 124.7 | 442.6 |

## Verdict

- GoEmotions: macro F1 0.446 @0.5 -> 0.522 tuned (thresholds tuned on validation, evaluated on test) vs majority 0.018; clearly beats baseline, in line with published ~0.46-0.52 for this model.
- Ekman-6 F1 0.628 vs lexicon 0.245; EmoBank r (V/A/D) = 0.619/0.362/0.385 vs lexicon valence 0.523. VAD is an indirect ridge on emotion probabilities, not a trained regression head, so arousal/dominance stay weak.
- Caveat: the model was trained on GoEmotions (Reddit); EmoBank (mixed genres) is out-of-domain. Bootstrap CIs ignore threshold-tuning variance.
