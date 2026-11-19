# Engine C (stylometric) evaluation

EmoBank test (n=1000), 95% bootstrap CI (500 resamples). Baseline = constant train mean (Pearson r is undefined for a constant; effectively 0).

| Dim | Model | Pearson r | CCC | MAE |
|---|---|---|---|---|
| valence | Stylometric ridge | 0.308 [0.245, 0.372] | 0.164 [0.130, 0.204] | 0.117 [0.111, 0.125] |
| valence | Mean-of-train baseline | undefined (0) | 0.000 [-0.000, 0.000] | 0.122 [0.115, 0.131] |
| arousal | Stylometric ridge | 0.362 [0.283, 0.438] | 0.230 [0.164, 0.292] | 0.089 [0.084, 0.093] |
| arousal | Mean-of-train baseline | undefined (0) | -0.000 [-0.000, 0.000] | 0.095 [0.090, 0.100] |
| dominance | Stylometric ridge | 0.288 [0.225, 0.351] | 0.120 [0.090, 0.153] | 0.075 [0.071, 0.080] |
| dominance | Mean-of-train baseline | undefined (0) | 0.000 [-0.000, 0.000] | 0.078 [0.074, 0.083] |

Verdict: weak (r V/A/D = 0.31/0.36/0.29), small MAE/CCC gain over the baseline: a low-confidence prior only.
Value is in the explainable raw features, best read relative to a personal baseline.
