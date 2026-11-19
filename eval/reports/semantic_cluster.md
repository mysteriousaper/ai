# Engine F (semantic / emotion-space cluster map) evaluation: v1 vs v2

v1 = MiniLM embeddings (topical). v2 = emotion-space map: standardized logits of Engine B's 28 probabilities + PCA-16 of MiniLM x0.3; UMAP cosine, HDBSCAN min_cluster_size 100 (eom), soft-membership T=0.1. Variant chosen on validation NMI among configs with noise <= 0.40.

## v1 vs v2 (GoEmotions test n=5427; one global threshold tuned on validation; 95% CI, 200 resamples)

| Metric | v1 | v2 |
|---|---|---|
| Macro F1 (28 labels) | 0.126 [0.120, 0.133] | 0.453 [0.440, 0.465] |
| Ekman-6 F1 (first 2000, thr 0.3) | 0.069 | 0.589 |
| NMI vs primary label (HDBSCAN, non-noise, train) | 0.120 | 0.526 |
| NMI, K-means same k (all pts) | 0.064 | 0.453 |
| Purity (HDBSCAN; majority share 0.295) | 0.355 | 0.642 |
| Purity, K-means | 0.328 | 0.632 |
| Noise share | 0.743 | 0.352 |
| Leaf clusters / boxes | 75 / 8 | 45 / 8 |
| Stability ARI (2 other seeds) | 0.938, 0.950 | 0.814, 0.881 |

| Reference | Macro F1 | Ekman-6 F1 |
|---|---|---|
| Transformer (B), per-label tuned thresholds | 0.522 [0.499, 0.540] | 0.628 |
| Lexicon (A) | n/a (see lexicon.md) | 0.245 [0.229, 0.259] |
| Majority ('neutral') | 0.018 | 0.093 |

v2 held-out validation (nearest-prototype assignment): NMI 0.364, purity 0.527. Variant search: 24 configs (logit-only vs +emb, cosine vs euclidean UMAP, 6 HDBSCAN settings), see cluster_map_v2/metrics.json.

## v2 top-level boxes (8) and child leaves

### Box 0: approval/optimism | hope, wish, luck (n=3796)
- leaf 26 (n=1127): caring 0.62, optimism 0.15, neutral 0.15; words: luck, good, worry, help, stay
- leaf 33 (n=908): approval 0.73, neutral 0.30, optimism 0.02; words: agree, right, yes, true, agreed
- leaf 27 (n=811): optimism 0.78, admiration 0.12, caring 0.06; words: hope, hopefully, hoping, soon, luck
- leaf 25 (n=628): desire 0.62, neutral 0.25, optimism 0.09; words: wish, want, wanted, need, wanna
- leaf 30 (n=211): approval 0.57, neutral 0.37, optimism 0.15; words: possible, ll, healthcare, support, try
- leaf 24 (n=111): neutral 0.74, caring 0.21, approval 0.05; words: care, mom, need, pain, just

### Box 1: surprise/excitement | wow, surprised, interesting (n=2246)
- leaf 9 (n=1335): surprise 0.62, neutral 0.13, admiration 0.10; words: wow, surprised, wonder, oh, omg
- leaf 13 (n=411): excitement 0.49, joy 0.29, neutral 0.20; words: happy, cake, day, birthday, new
- leaf 14 (n=202): excitement 0.84, neutral 0.09, curiosity 0.05; words: excited, wait, can't, exciting, excitement
- leaf 16 (n=182): curiosity 0.30, excitement 0.26, approval 0.18; words: interesting, interested, that's, cool, really
- leaf 15 (n=116): admiration 0.81, curiosity 0.42, confusion 0.05; words: great, impressive, good, did, looks

### Box 2: amusement/love | love, lol, haha (n=4835)
- leaf 6 (n=2331): amusement 0.75, neutral 0.13, joy 0.08; words: lol, haha, funny, fun, lmao
- leaf 4 (n=2160): love 0.80, admiration 0.16, approval 0.06; words: love, like, favorite, loved, loves
- leaf 5 (n=226): amusement 0.78, curiosity 0.25, confusion 0.13; words: lol, funny, haha, joke, did
- leaf 3 (n=118): amusement 0.66, love 0.64, admiration 0.09; words: love, lol, haha, like, hilarious

### Box 3: gratitude/admiration | thanks, thank, great (n=5725)
- leaf 2 (n=2104): gratitude 0.98, admiration 0.09, optimism 0.03; words: thanks, thank, sharing, i'll, appreciate
- leaf 18 (n=1974): admiration 0.89, approval 0.09, neutral 0.07; words: great, good, awesome, amazing, nice
- leaf 12 (n=1222): joy 0.56, admiration 0.14, neutral 0.12; words: glad, happy, enjoy, i'm, cheers
- leaf 17 (n=153): admiration 0.80, annoyance 0.08, anger 0.07; words: great, good, pretty, job, it's
- leaf 11 (n=145): gratitude 0.47, admiration 0.46, excitement 0.11; words: welcome, congrats, congratulations, new, awesome
- leaf 1 (n=127): gratitude 0.96, amusement 0.59, joy 0.05; words: thanks, thank, lol, laugh, haha

### Box 4: neutral/approval | just, got, remindme (n=2021)
- leaf 38 (n=1254): neutral 0.98, excitement 0.01, approval 0.00; words: remindme, big, way, hi, just
- leaf 37 (n=257): neutral 0.98, disapproval 0.02, approval 0.01; words: doesn't, said, bitcoin, just, didn't
- leaf 34 (n=197): neutral 0.95, admiration 0.01, amusement 0.01; words: got, big, tipped, cough, clutch
- leaf 39 (n=182): neutral 0.96, approval 0.04, realization 0.01; words: just, half, right, think, probably
- leaf 29 (n=131): neutral 0.92, annoyance 0.02, realization 0.02; words: got, dead, deny, ripped, face

### Box 5: disapproval/neutral | fuck, hate, don't (n=2342)
- leaf 36 (n=1346): disapproval 0.60, neutral 0.37, annoyance 0.06; words: don't, don, think, can't, doesn't
- leaf 43 (n=405): anger 0.80, annoyance 0.21, neutral 0.10; words: fuck, fucking, hell, dare, shut
- leaf 44 (n=309): annoyance 0.63, anger 0.26, neutral 0.15; words: stupid, dumb, idiots, shit, sucks
- leaf 42 (n=172): anger 0.74, annoyance 0.23, disapproval 0.10; words: hate, people, angry, fuck, infuriating
- leaf 35 (n=110): neutral 0.56, annoyance 0.35, disapproval 0.13; words: racist, delusional, you're, care, people

### Box 6: curiosity/neutral | curious, sure, know (n=2689)
- leaf 22 (n=1613): neutral 0.47, curiosity 0.43, confusion 0.17; words: does, did, think, mean, what's
- leaf 10 (n=652): confusion 0.72, neutral 0.19, disapproval 0.05; words: sure, confused, know, don't, understand
- leaf 19 (n=226): curiosity 0.85, neutral 0.13, optimism 0.04; words: curious, curiosity, i'm, just, like
- leaf 21 (n=198): curiosity 0.36, confusion 0.18, neutral 0.18; words: wrong, bad, what's, weird, did

### Box 7: sadness/neutral | sorry, sad, i'm (n=4474)
- leaf 28 (n=832): realization 0.46, neutral 0.42, approval 0.07; words: thought, realize, realized, didn't, forgot
- leaf 0 (n=831): remorse 0.58, sadness 0.26, caring 0.05; words: sorry, i'm, regret, loss, apologies
- leaf 8 (n=650): fear 0.68, neutral 0.13, disgust 0.07; words: terrible, scared, horrible, afraid, scary
- leaf 40 (n=570): sadness 0.86, disappointment 0.13, neutral 0.08; words: sad, sadly, miss, poor, hurts
- leaf 32 (n=504): disgust 0.70, annoyance 0.15, neutral 0.09; words: disgusting, worse, awful, worst, nasty
- leaf 41 (n=384): disappointment 0.71, neutral 0.13, sadness 0.10; words: disappointed, disappointing, bad, missed, unfortunately
- leaf 23 (n=233): embarrassment 0.69, annoyance 0.08, neutral 0.08; words: shame, awkward, embarrassing, ashamed, embarrassment
- leaf 20 (n=177): curiosity 0.38, sadness 0.27, disappointment 0.20; words: sad, really, did, crying, bad
- leaf 7 (n=154): nervousness 0.52, fear 0.26, sadness 0.08; words: worried, anxiety, nervous, anxious, worrying
- leaf 31 (n=139): neutral 0.33, disgust 0.28, annoyance 0.24; words: weird, flex, odd, it's, ok

## v1 top-level boxes (for reference; leaves omitted)

- Box 0: neutral/love | love, oh, thank (n=852)
- Box 1: neutral/annoyance | religion, women, racist (n=1651)
- Box 2: neutral/admiration | team, game, season (n=1288)
- Box 3: neutral/admiration | video, joke, cat (n=1651)
- Box 4: admiration/gratitude | happy, cake, birthday (n=747)
- Box 5: neutral/gratitude | game, thanks, play (n=2089)
- Box 6: neutral/caring | kids, mom, luck (n=1117)
- Box 7: neutral/admiration | pizza, hair, food (n=1757)

## Verdict

- v2 is clearly more emotional than v1 (NMI 0.12 -> 0.53, macro F1 0.126 -> 0.453, noise 0.74 -> 0.35); it still does not beat B (0.522), because its output is a smoothed mix of cluster label distributions. Train NMI/purity are in-sample and non-noise only; held-out validation NMI is 0.36, and the 24 variants differ by <0.04 val NMI (emb+logit vs logit-only is within noise).
- Independence: v2 emotions are DERIVED from Engine B's probabilities, so they are NOT independent evidence for fusion. Fusion should treat F as explanation/neighbourhood (box path, leaf words, similar texts) with low weight; do not count it as a second vote next to B.
- Caveats: English only; noise points get no leaf (assignment is nearest prototype, so it can differ from offline labels); boxes are data-driven neighbourhoods, not natural psychological categories; GoEmotions labels are noisy and neutral-heavy.
