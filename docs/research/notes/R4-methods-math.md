# R4: Mathematical core (clustering, fusion, calibration, temporal model, evaluation)

Researcher: R4. Date: 2026-11-18.

**Reading caveats (important).** In this session WebFetch could not reach arxiv.org, pnas.org, pmc.ncbi.nlm.nih.gov, scikit-learn.org, any `*.readthedocs.io` or `*.github.io` host (DNS failure, outside the allowlist). I could reach `raw.githubusercontent.com` and `pypi.org`. So: library facts (PyPI JSON, scikit-learn and hdbscan and UMAP source/docs files from GitHub, BERTopic docs source) are **primary and deep-read**; paper claims are mostly from **search-result abstracts and secondary summaries** and are marked `[abstract-level]` or `single-source`. Statements marked **(inference)** are my own recommendations, not sourced facts. The one video attempt failed (see the end).

## Summary

1. **Hierarchy (engine F):** do not trust HDBSCAN alone to give "big clusters that split into sub-clusters". Recommended recipe: sentence embeddings (L2-normalised) -> UMAP (cosine, `n_components` 10-15, `n_neighbors` 30, `min_dist` 0.0) -> HDBSCAN with `leaf` selection for fine "neighbourhoods" -> **Ward agglomerative on the cluster centroids** (scipy) to get the coarse-to-fine dendrogram. Validate with DBCV, noise fraction, ARI across >= 20 seeds, and bootstrap Jaccard stability. Cowen & Keltner's own result is that emotion categories are bridged by continuous gradients, so expect soft boundaries: report soft membership and nearest neighbourhoods, never hard cluster IDs as truth.
2. **Fusion:** start with confidence-weighted late fusion as the baseline, then a **per-label** stacking logistic regression on out-of-fold, group-aware predictions; train with modality dropout/masking, and keep a **small set of meta-models per available-engine subset**.
3. **Calibration:** per-engine, per-label Platt (sigmoid) on logits when positives are plentiful, with a shared global temperature as the fallback for rare labels; isotonic only above roughly 2,000 calibration samples per label. Report per-label ECE with adaptive (equal-mass) bins, macro-averaged, plus Brier and reliability diagrams with bootstrap bands.
4. **VAD uncertainty:** split conformal intervals per dimension, hand-rolled (about 10 lines) or `crepes`/MAPIE.
5. **Overall confidence:** learn it. Fit a small logistic model on out-of-fold data that predicts "fused top label is correct" from fused max-probability, entropy, inter-engine Jensen-Shannon disagreement and coverage. `mixed_or_ambiguous` comes from conformal set size >= 2 or high disagreement, with the threshold tuned on validation.
6. **Temporal:** EMA as the baseline, a **Kalman local-level model on VAD** (heteroscedastic measurement noise taken from the engine std, time-scaled process noise, skip update when no data) as the main candidate, and an HMM only as an optional discrete-mood layer once a user has hundreds of observations. Personal baseline = Welford/exponentially-forgetting mean and variance with shrinkage to a population prior; no z-scores until enough data.
7. **Libraries:** use `pykalman` (BSD, release 2026-01-31) or `statsmodels`, or hand-roll a 3-dim Kalman. **Do not rely on `filterpy`**: it was dormant from 2018-10 to 2026-10-26, then a "2.0.0" appeared (needs provenance check). `hmmlearn` is fine but its last release was 2024-10-31.
8. **Evaluation:** CCC as the headline VAD metric (Pearson and MAE alongside); per-label thresholds tuned on validation only; 2,000+ group-level bootstrap resamples (BCa via `scipy.stats.bootstrap`); CheckList MFT/INV/DIR with pass rates. **Ceiling:** GoEmotions authors report average F1 0.46 (28 labels), 0.64 (Ekman-6), 0.69 (sentiment); treat anything much above about 0.55 macro-F1 on the 28-label task as a leakage suspect (inference).

---

## 1. Emotion-concept hierarchy by clustering

### 1.1 What the literature says
- **Cowen & Keltner (PNAS 2017)**: 2,185 short videos elicited reported experience of **27 distinct categories** "bridged by continuous gradients"; the authors say "We don't get finite clusters of emotions in the map because everything is interconnected." Sample size is reported inconsistently across secondary coverage (853 vs "more than 800"). [abstract/press-level, via search; [sci.news](https://www.sci.news/othersciences/psychology/27-categories-emotion-05212.html), [neurosciencenews](https://neurosciencenews.com/emotion-27-states-7431/)]. Barrett et al. (2018, Trends Cogn Sci) disputed the discrete reading; Cowen & Keltner replied (PMC 5866789 per search results). **Implication (inference):** our hierarchy should be a *soft, navigable map* (nearest neighbourhoods plus gradients), which matches the brief's "nearest emotional neighbourhoods".
- **The owner's half-remembered idea** ("common words go into big clusters which split into sub-clusters") is exactly what a **dendrogram / condensed tree** is: agglomerative clustering produces it directly; HDBSCAN produces it internally (the condensed tree from single linkage on mutual-reachability distance).
- **BERTopic hierarchical topics** do not use the HDBSCAN tree. Per the BERTopic docs source: the hierarchy is "approximated using our topic-term matrix (c-TF-IDF matrix)", merged with **SciPy hierarchical clustering, default linkage Ward**, and a custom `linkage_function` lambda (e.g. `sch.linkage(x, 'single', optimal_ordering=True)`) is the only documented customisation; merged topics get their representation by summing bag-of-words ([BERTopic docs source](https://raw.githubusercontent.com/MaartenGr/BERTopic/master/docs/getting_started/hierarchicaltopics/hierarchicaltopics.md)). So the proven pattern is "flat density clusters first, then Ward on the cluster representations". I recommend the same, but on **embedding centroids** instead of c-TF-IDF (inference).
- I found no 2023-2026 paper that specifically benchmarks hierarchical clustering of *emotion-word* embeddings; searches returned only general topic-modelling and GoEmotions-taxonomy work (not found; searched "hierarchical clustering emotion embeddings", "Cowen Keltner semantic space"). Treat the recipe below as engineering judgement backed by tool docs.

### 1.2 Recommended pipeline and parameters
| Step | Setting | Basis |
|---|---|---|
| Embeddings | L2-normalise; use cosine | standard (inference) |
| UMAP | `n_neighbors` 30 (explore 15-50), `min_dist` 0.0, `n_components` 10-15 (not 2), `metric="cosine"`, fixed `random_state` for the production map only | UMAP docs: use larger `n_neighbors` than default 15 ("double it ... up to 30"), `min_dist` 0, "you can reduce to 10 dimensions instead since the goal is to cluster, not visualize" ([UMAP clustering doc source](https://raw.githubusercontent.com/lmcinnes/umap/master/doc/clustering.rst)) |
| HDBSCAN (fine layer) | `min_cluster_size` 30-100 for about 50k texts (about 0.1-0.2% of N); `min_samples` 5-10 (lower = less noise); `cluster_selection_method="leaf"`; `cluster_selection_epsilon` only if micro-clusters appear; keep `alpha=1.0` | hdbscan docs: set `min_cluster_size` to "the smallest size grouping that you wish to consider a cluster"; `leaf` for "many small, homogeneous clusters"; `eom` returns "a few large clusters plus several small extras"; "best not to mess with" `alpha` ([hdbscan parameter selection source](https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/master/docs/parameter_selection.rst)). Concrete numbers are my inference. |
| Coarse layer | `scipy.cluster.hierarchy.linkage(centroids, 'ward')` on UMAP-space or original-space cluster centroids (use original cosine space centroids, then re-normalise, as the check on UMAP distortion); cut the dendrogram at 5-10 super-clusters; compare the cut with Plutchik-8/Ekman-6 | BERTopic pattern above (Ward default); cut count is inference |
| Alternative for the 28 label names alone | embed the 28 GoEmotions label words/definitions and Ward-cluster them directly (28 points: instant, fully interpretable) | inference; useful as a sanity map before the big corpus run |

**API note for the ADR-0004 revisit:** `sklearn.cluster.HDBSCAN` (the default we chose) keeps its tree as the private `_single_linkage_tree_` and the docs I read do not expose the condensed tree or `relative_validity_` ([scikit-learn source](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/cluster/_hdbscan/hdbscan.py)). The standalone `hdbscan` package (BSD, 0.8.44, 2026-06-01) exposes `condensed_tree_`, soft membership and `validity_index`. For engine F, the Ward-on-centroids route does not need the condensed tree, so sklearn's HDBSCAN is still sufficient for flat labels; adopt `hdbscan` only if we want soft membership vectors (inference; this matches the ADR's own stated trigger).

### 1.3 Quality and stability evaluation
- **DBCV** (Moulavi et al., SDM 2014; range [-1, 1], higher is better) is the density-based validity index; **noise points lower the score** and recent work (DISCO, arXiv 2503.00127) argues DBCV-like indices penalise noise rather than rate its labelling and can bias selection [[DBCV paper](https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf), [R dbscan docs](https://rdrr.io/cran/dbscan/man/dbcv.html), [arXiv 2503.00127](https://arxiv.org/pdf/2503.00127v2)]. Report DBCV together with the noise fraction, not alone.
- **Silhouette**: compute only on non-noise points and in the original embedding space; it favours convex clusters and is weak for density clusters (inference). Use as a secondary number.
- **Seed stability:** run the whole UMAP+HDBSCAN pipeline for >= 20 seeds and report mean pairwise **ARI** over the points that are clustered in both runs, plus the fraction of points changing noise status. Published reference values are low-to-moderate: median ARI about 0.61 for unseeded UMAP runs (ibUMAP preprint), about 0.72 across seeds (TriTopic preprint), 0.67-0.81 (a scribe-census preprint), 0.876 over 20 seeds (PASM preprint) [all arXiv preprints, single-source each: [2610.01445](https://arxiv.org/pdf/2610.01445), [2602.19079](https://arxiv.org/pdf/2602.19079), [2608.31017](https://arxiv.org/pdf/2608.31017), [2604.00074](https://arxiv.org/pdf/2604.00074)]. **Acceptance bar (inference):** mean ARI >= 0.7 for the fine layer, >= 0.85 for the coarse cut; otherwise ship the hierarchy as "exploratory".
- **Bootstrap stability:** resample 80% of the texts 50-100 times, recluster, and compute per-cluster Jaccard against the best-matching original cluster (Hennig's clusterboot idea; unsourced here, background knowledge). Clusters with mean Jaccard < 0.6 are flagged unstable.
- **External sanity:** NMI/purity against GoEmotions labels and Plutchik/Ekman groupings (labels are noisy, so use as a sanity check, not as a target).
- **Pitfalls:** UMAP "does not completely preserve density" and "can also create false tears in clusters"; the doc calls clustering on UMAP output "somewhat controversial" ([UMAP doc source](https://raw.githubusercontent.com/lmcinnes/umap/master/doc/clustering.rst)). A fixed seed only gives identical output for identical input *order* ([datarock](https://datarock.com.au/?p=667), single-source). So: (a) cross-check by also running HDBSCAN on a PCA-reduced (50-d) space, (b) always store the fitted UMAP model and use `transform` for new text rather than refitting, (c) pin versions (UMAP 0.6.0 was released 2026-11-10, only days ago; prefer the 0.5.12 pin or re-verify before upgrading).

---

## 2. Fusion

- **Evidence:** within the multimodal affect literature, late fusion is the simplest but often reported to underperform feature-level/attention fusion because it "only captures features learned within each modality" ([arXiv 2201.11095](https://arxiv.org/pdf/2201.11095.pdf), [arXiv 2505.04642](https://arxiv.org/pdf/2505.04642v1)) [abstract-level]. A Transformer-decoder weighting of modalities reports about 30% valence gain over a late-fusion baseline and lower sensitivity to missing modalities ([arXiv 2311.10119](https://arxiv.org/pdf/2311.10119v1)) [abstract-level, single-source]. **Modality dropout** substantially improves performance when a modality is missing ([arXiv 2201.11095](https://arxiv.org/pdf/2201.11095.pdf)) [abstract-level]. Caveat: those works fuse audio/video features; ours fuses the *outputs* of six heterogeneous engines on a small labelled set, so stacking is the right analogue and attention fusion is over-parameterised for us (inference). No source I found benchmarks late fusion vs stacking head-to-head.
- **Recommendation (inference):**
  1. **Baseline:** confidence-weighted late fusion of logits/probabilities (weights from each engine's validation per-label F1 or inverse Brier). This is the "no learning" control for the ablation.
  2. **Stacking:** one logistic regression **per label** whose inputs are only *that label's* logit from each available engine (about 4-6 features), plus the engine-present mask. This keeps the parameter count tiny, which matters because many GoEmotions labels are rare. A small shared MLP is an optional second candidate; judge by the ablation.
  3. **Leakage control:** build the meta-training set from **out-of-fold** engine predictions (`sklearn.model_selection.cross_val_predict`-style, 5 folds), with **group-aware folds** (group by author/thread/conversation so near-duplicate and same-person texts never straddle folds). Engines that were pre-trained on GoEmotions (e.g. the off-the-shelf transformer) must not be evaluated on the data they trained on: the meta-set must come from a split none of the engines saw. Calibration of the stacker uses a *third* held-out fold (nested), never the stacker's own training predictions.
  4. **Missing modalities:** (a) engines output `coverage`; fusion treats `coverage==0` as masked; (b) train the stacker with **random modality dropout** on the meta-features so it sees realistic missing patterns; (c) because text engines A/B/C/F are always present while D (keystroke) and E (LLM) are optional, keep **4 meta-models, one per {D, E} availability pattern**, rather than 2^6; (d) overall confidence drops when engines are missing (via the features in section 3.4).

---

## 3. Calibration

### 3.1 Methods
- **Temperature scaling** (Guo et al. 2017, single-parameter Platt variant) is the usual best post-hoc method for multi-class nets ([Guo et al. via ar5iv](https://ar5iv.labs.arxiv.org/html/1706.04599)), but it assumes a single-label softmax. For multi-label (independent sigmoids) a **separate temperature per emotion** has been used in speech emotion recognition with reported ECE improvements of about 15-20% ([NSF PAR 10441271](https://par.nsf.gov/servlets/purl/10441271)) [abstract-level, single-source].
- **Dirichlet calibration** (Kull et al., NeurIPS 2019) generalises beta calibration; for neural nets it is a log-transform, one linear layer and a softmax; its advantage is largest on **classwise** calibration ([NeurIPS abstract](https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html), [arXiv 1910.12656](https://arxiv.org/pdf/1910.12656)). It is a multiclass (softmax) method; for 28 independent sigmoids it has K^2 parameters (about 784) and would overfit rare labels. **Not recommended** for the main multi-label head; maybe for a collapsed Ekman-7 softmax view (inference).
- **Platt vs isotonic and sample size:** in the boosting study, below roughly **2,000** calibration cases Platt beats isotonic because isotonic is "less constrained" and overfits ([Niculescu-Mizil & Caruana, arXiv 1207.1403](https://arxiv.org/pdf/1207.1403)); practitioner rules of thumb put Platt below 1k labels and "fit both, pick by held-out cross-entropy" at 1k-10k (non-authoritative secondary guides; treat as heuristic). Isotonic output is a jagged step function that moves between retrains. scikit-learn's `CalibratedClassifierCV` offers `sigmoid`, `isotonic` and (in recent versions) `temperature`, with calibration data drawn from internal CV, so per-fold calibration sets are smaller than the total ([sklearn calibration.py mirror](https://gitverse.ru/githubmirror/scikit-learn/content/main/sklearn/calibration.py)).
- **Recommendation (inference):** for each engine and label, fit Platt (2 parameters) on the logit; if a label has fewer than about 100 positives in the calibration fold, fall back to one **global temperature** plus a per-label bias (1 + K parameters). Use isotonic only for labels with >= 2,000 calibration samples **and** >= 200 positives, and only if it beats Platt on held-out log-loss. Rare GoEmotions classes (grief, pride, relief, nervousness have the fewest examples; exact counts not verified this session) will therefore get the simple method. Calibrate the **fused** output last, on its own held-out fold.

### 3.2 Metrics
- ECE is the default but has binning pathologies. Nixon et al. recommend **class-conditional**, **adaptive (equal-mass) binning** (more stable rank ordering as bin count varies) and note the L2 norm helps; method rankings change with the choice of measure ([Nixon et al., arXiv 1904.01685](https://arxiv.org/abs/1904.01685v2)).
- Multi-label specifics: a 2026 preprint evaluates each label as its own binary problem, argues standard adaptive binning breaks on these binary tasks (most predictions near zero, one huge bin), and proposes binning positive and negative predictions separately; system-level calibration = mean of per-label errors ([arXiv 2609.26468](https://arxiv.org/pdf/2609.26468), single-source, abstract-level). Classwise-ECE (Kull 2019) is the older formulation ([arXiv 2203.08958](https://arxiv.org/html/2203.08958v4)).
- **Recommended report (inference):** per-label ECE with equal-mass bins (bins = min(15, n_label / 50)), macro-averaged, **plus** Brier score and log-loss (bin-free), plus reliability diagrams with bootstrap bands. Because ECE with few samples is noisy, **always report a bootstrap CI** and mark labels with < 50 positives as "insufficient to assess".
- **How many samples:** no rigorous number found. Platt becomes stable at a few hundred, isotonic needs thousands (see above); reliability at the 0.9 bin needs about 100 predictions in the bin for a +/- 6% binomial standard error (arithmetic: sqrt(0.9*0.1/100) = 0.03, so about 6% at 95%). Hence report-worthy per-label ECE needs roughly 1-2k test samples with >= 100 positives per label (inference).

### 3.3 Calibrating VAD (regression)
- **Split conformal**: fit on a proper training set, compute absolute residuals on a disjoint calibration set, take the ceil((n+1)(1-alpha))/n quantile; the coverage guarantee is **marginal**, not conditional ([CQR paper, arXiv 1905.03222](https://arxiv.org/pdf/1905.03222); [overview](https://www.tidymodels.org/learn/models/conformal-regression/)). Constant-width intervals ignore heteroscedasticity, so **normalise the score by a predicted uncertainty** (our engine `vad.std`) ([oreilly chapter](https://www.oreilly.com/library/view/practical-guide-to/9781805122760/B19925_07.xhtml), secondary).
- For bounded targets ([-1, 1]), clip the interval to the range or conformalise after a logit-type transform (secondary source above). I found no paper applying conformal prediction to VAD specifically (not found).
- **Recommendation (inference):** one normalised split-conformal interval per V/A/D dimension at alpha 0.1, calibration set >= 200 (more is better; the minimum for a 90% interval is n >= 9 but quantiles are very noisy there). Verify **empirical coverage on the test set** and report it, since intervals can undercover under distribution shift (EmoBank vs real chat text is a real shift). Hand-rolled numpy is enough; `crepes` (BSD, 0.9.1, 2026-06-12) and MAPIE (BSD-3, 1.6.0, 2026-10-19) are optional.

### 3.4 Overall confidence, disagreement, coverage, "mixed/ambiguous"
**(All inference; no source found that defines this exact combination.)**
- Features per message: calibrated fused max probability, fused entropy, mean pairwise **Jensen-Shannon divergence** between engine distributions (weighted by availability), VAD disagreement (std of V/A across engines), `coverage` (e.g. fraction of tokens covered by lexicon/model), number of engines present, text length.
- **Learn it**: logistic regression (or isotonic on its output) trained on out-of-fold data to predict "fused top-1 label is in the gold set". Its output **is** `overall_confidence` and is calibrated by construction. This beats a hand-made product of the three terms, which is uncalibrated.
- `mixed_or_ambiguous = true` when (a) the conformal prediction set (fit on the fused probabilities) contains >= 2 labels from different Plutchik families or opposite poles, **or** (b) mean JSD between engines exceeds a threshold tuned on validation (maximise F1 of detecting "engines' top labels disagree and fused is wrong"). Report the rate of ambiguous flags; if > 40% of messages are flagged, the threshold is too loose.
- GoEmotions has genuine disagreement: models trained on hard majority labels maximise F1 but give "poorly calibrated probabilities that do not reflect annotator uncertainty" ([arXiv 2604.27345](https://arxiv.org/pdf/2604.27345), abstract-level). Consider training with soft labels (fraction of raters choosing each emotion) for the transformer head; the raw per-rater GoEmotions file supports this ([GoEmotions README](https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md)).

---

## 4. Temporal model

### 4.1 Which smoother
| Option | Pros | Cons | Verdict |
|---|---|---|---|
| EMA | 1 parameter, trivial, robust | no uncertainty, fixed gain, no missing-data or irregular-gap handling | **baseline / fallback** |
| Kalman filter (local level or AR(1) around personal mean) on [V, A, D] | measurement noise from engine std (heteroscedastic), handles gaps and irregular intervals (scale Q by dt), outputs state uncertainty usable for "trend is real?" | linear-Gaussian; needs Q/R tuning | **main candidate** |
| HMM over discrete moods | interpretable states, switching behaviour | needs lots of data per person; number of states is a modelling choice | **optional, later** |

- **Evidence:** a 2024 PLOS Comput Biol paper applies a **Kalman filter to EMA mood data** (>700 participants, with and without depressive symptoms) and reports it "performs favourably compared to the standard vector autoregressive approach"; the framework "can handle unequal sampling intervals, missing data, and can account for measurement error and inputs" ([Malamud et al. 2024, PMC11449358](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11449358/); [PubMed 39312537](https://pubmed.ncbi.nlm.nih.gov/39312537/)). A Monte Carlo arXiv study found the Kalman filter handled missing data better than multiple imputation in some settings ([arXiv 2305.03214](https://arxiv.deeppaper.ai/papers/2305.03214v1?locale=en), abstract-level).
- **HMMs on mood:** a multilevel HMM on about 9,820 assessments from 20 bipolar patients found four states (neutral, elevated, mixed, lowered); Bayesian multilevel HMMs on 29 depression-treatment patients favoured three states with switching on about 24% of occasions, and between-person variation was high ([search summaries; [Utrecht record](https://research-portal.uu.nl/en/publications/evidence-for-mood-instability-in-patients-with-bipolar-disorder-a/), [Zenodo code](https://zenodo.org/doi/10.5281/zenodo.15775394)]; abstract-level). These are clinical samples with about 500 observations per person, far above what a new user has. **(inference)** Fit HMMs only per user after about 200+ observations, or fit one population HMM and only *decode* per user.
- **Verdict by metrics (inference), per brief "pick by metrics":** evaluate on synthetic scripted trajectories (sad->neutral->happy; step, ramp, noisy plateau, missing blocks) and later the owner's data using one-step-ahead predictive error, lag/response time to a true change, and false-trend rate on pure noise. Expect EMA to be competitive on noisy short series; keep the Kalman only if it beats EMA by a margin on these tests.

### 4.2 Within-person dynamics and pattern outputs
- **Inertia** = autocorrelation / AR(1) coefficient of an affect series; higher inertia was associated with low self-esteem and depression and later predicted depression onset in adolescents ([Kuppens, Allen & Sheeber 2010 via search](https://findanexpert.unimelb.edu.au/scholarlywork/323690-emotional-inertia-and-psychological-maladjustment); onset study [here](https://findanexpert.unimelb.edu.au/scholarlywork/344108-emotional-inertia-prospectively-predicts-the-onset-of-depressive-disorder-in-adolescence)) [abstract-level].
- **Variability** = within-person SD; **instability** = **MSSD** (mean squared successive difference), proposed as preferred because it reflects both variability and temporal dependency, with MSSD = 2*SD^2*(1 - AR(1)) per a secondary summary ([Jahng, Wood & Trull 2008 via search](https://documentserver.uhasselt.be/bitstream/1942/24009/2/Vansteelandt%20K%20Revision%20Multivariate%20Behavioral%20Research%20Blinded%20Text%20Revision%203.pdf); Wang et al. 2012 critique noted in the same results) [secondary, verify formula]. Because MSSD mixes both, **report SD and lag-1 autocorrelation separately and MSSD as a convenience** (inference). Important: our series are model *estimates* with measurement error, so estimated inertia/variability are biased; the Kalman state estimates (not raw per-message VAD) should feed these statistics (inference).
- **Pattern outputs (inference):** trend = slope of the Kalman level over the last N messages with a "real change" flag when |slope| > 2 posterior SE; volatility = SD or MSSD of the smoothed state; deviation = z-score of the current smoothed state against the personal baseline; triggers = event-type -> mean next-step delta in valence with bootstrap CI, shown only above a minimum count (e.g. >= 10 occurrences).

### 4.3 Personal baselines and cold start
- Use **Welford** online mean/variance with an exponential forgetting factor (half-life about 14 days of data, tunable), and a robust variant (median/MAD) for keystroke features that have heavy tails (inference).
- **Cold start by shrinkage:** hierarchical/empirical-Bayes logic says estimates shrink to the population mean when a person has few observations and approach their own as data accumulate ([MetricGate hierarchical models](https://metricgate.com/blogs/bayesian-hierarchical-models-explained/), [empirical Bayes](https://metricgate.com/docs/empirical-bayes-estimation/); secondary, non-authoritative). Practical rule (inference): baseline mean = (n0 * mu_pop + n * xbar) / (n0 + n) with n0 about 10-20 pseudo-observations, variance similarly; **withhold z-scores (or flag low confidence) until n >= about 20-30 per feature**; population priors come from the training corpora (EmoBank/GoEmotions for VAD, EmoSurv for keystrokes).
- Never persist the baseline in a way that cannot be deleted (brief requires `DELETE /user/{id}/data`).

### 4.4 Python libraries
See the library table in the machine-readable block. Key points:
- **filterpy** (MIT): the PyPI release history is 1.4.4 (2018-08-24), 1.4.5 (2018-10-10), then **2.0.0 on 2026-10-26** with no releases in between ([PyPI JSON](https://pypi.org/pypi/filterpy/json)). An eight-year gap followed by a major release needs a provenance check (maintainer, repository commits, diff against 1.4.5) before anyone depends on it. **Verdict: avoid as a dependency**; a Kalman filter for 3 independent states is about 25 lines of numpy and easy to property-test with Hypothesis.
- **pykalman** 0.11.2 (BSD-3, 2026-01-31, recent cadence 0.11.0 2025-11-27, 0.11.1 2025-12-31); supports EM for Q/R and missing observations (masked arrays) per its docs (not re-verified this session).
- **statsmodels** 0.15.0 (BSD-3, 2026-08-27): `UnobservedComponents`/`MLEModel` give a Kalman filter with ML estimation and missing data; already a standard dependency in this ecosystem.
- **hmmlearn** 0.3.3 (BSD; last release 2024-10-31, so about 13 months old at today's date; PyPI shows the BSD classifier but null `license` fields, so confirm the LICENSE file in the repo before shipping).

---

## 5. Evaluation methodology

### 5.1 VAD: CCC vs Pearson
- Pearson ignores systematic offset and scale; **CCC** = 2*cov / (var_x + var_y + (mean_x - mean_y)^2) penalises shifts in mean and variance, which is why AVEC/SEWA/A-VB style challenges report it ([SEWA DB, arXiv 1901.02839](https://arxiv.org/pdf/1901.02839); [Atmaja & Akagi, arXiv 2003.10724](https://arxiv.org/pdf/2003.10724v2); [A-VB Two](https://hyper.ai/en/sota/tasks/a-vb-two)). CCC loss gave better averaged CCC than MSE in multitask SER ([arXiv 2003.10724](https://ar5iv.labs.arxiv.org/html/2003.10724)). Lin (1989) itself was not retrievable (cited via secondary papers).
- **Recommendation (inference):** headline = **CCC per dimension and mean**, with Pearson, MAE and RMSE alongside, plus the interval coverage from 3.3. Compute CCC on the *same unit scale* ([-1, 1]) after the explicit scale-conversion functions; a CCC far below Pearson reveals a scale/offset bug in a converter, which is a useful unit-test signal. Note CCC is not defined on the lexicon engine's discretised outputs if variance is near zero; guard it.

### 5.2 Multi-label F1 and thresholds
- Report **macro-F1** (primary, rare labels count equally), **micro-F1**, per-label F1, plus ROC-AUC/PR-AUC (threshold-free) and sample-averaged F1 (inference).
- **Threshold selection:** pick a per-label threshold that maximises F1 (or a global one) on the **validation** split only, freeze, then apply to test; for stacked models do it on the calibration fold. Never tune thresholds on test. After calibration, a threshold near the calibrated-probability optimum (about F1/2 for the F1-optimal decision, a known property; unsourced, background knowledge) is expected, which gives a sanity check.
- **Per-label thresholds on rare labels overfit**: with < 50 positives in validation, fall back to the global threshold (inference).

### 5.3 Bootstrap CIs
- `scipy.stats.bootstrap` **defaults to `n_resamples=9999` and BCa** ([SciPy docs](https://docs.scipy.org/doc/scipy-1.9.1/reference/generated/scipy.stats.bootstrap.html)). Efron & Tibshirani's floor is R = 1,000; BCa estimates tail quantiles so needs more than percentile (secondary guidance: >= 1,000, prefer >= 2,000) ([TQMP paper](https://www.tqmp.org/RegularArticles/vol21-3/p125/p125.pdf), [MetricGate](https://metricgate.com/blogs/choosing-number-of-bootstrap-replicates/); the latter non-authoritative). I did not retrieve Efron/DiCiccio themselves.
- **Recommendation (inference):** percentile CIs with **B = 2,000** for exploratory work; **BCa with B = 9,999** for headline numbers (cheap: macro-F1 over about 6k test texts takes seconds); for macro-F1 use a *vectorised* resample of indices, not refitting. **Resample by group** (user/thread/conversation) when texts are not independent. For comparing two systems (fusion vs best single engine) use a **paired bootstrap** (same resample indices) on the metric *difference* and report the CI of the difference; "fusion beats best single" is claimed only if the CI excludes 0. Fix seeds and log them. With macro-F1 on rare labels, CIs will be wide; report them rather than hiding them.

### 5.4 CheckList behavioural tests
- CheckList (Ribeiro et al., ACL 2020, Best Overall Paper) has three test types: **MFT** (minimum functionality tests from templates, like unit tests), **INV** (invariance under label-preserving perturbations such as typos) and **DIR** (directional expectation, e.g. adding a positive sentence must not lower positivity) ([paper PDF](https://homes.cs.washington.edu/%7Emarcotcr/acl20_checklist.pdf), [arXiv 2005.04118 via ar5iv](https://ar5iv.labs.arxiv.org/html/2005.04118)). Reported effect: practitioners using CheckList wrote about twice as many tests and found almost three times as many bugs (secondary summary, [ymeadows review](https://ymeadows.com/en/articles/paper-review-beyond-accuracy-behavioral-testing-of-nlp-models-with-checklist)).
- **Mapping to the brief (inference):** negation = MFT/DIR on valence sign; intensity "sad < very sad < devastated" = DIR monotonic; name/place swaps and typos/lowercase = INV with a tolerance (e.g. |delta V| < 0.1 and unchanged top label in >= 95% of cases); report **pass rate with a Wilson 95% interval per test** and track per release. Generate templates in-house with fixed seeds (no extra dependency needed; the `checklist` package's licence/maintenance was not checked).

### 5.5 GoEmotions label noise and the ceiling
- **Authors' numbers:** fine-tuned BERT-base reaches an **average F1 of .46** on the 28-label taxonomy, **.64** on an Ekman-style 6-class grouping and **.69** on a sentiment grouping ([GoEmotions abstract via search](https://arxiv.org/abs/2005.00547v2), [ar5iv](https://ar5iv.labs.arxiv.org/html/2005.00547)). The abstract says "average F1", not explicitly macro-F1; one secondary summary gives 0.46 micro and 0.43 macro (unconfirmed) ([emergentmind](https://www.emergentmind.com/topics/goemotions-dataset)). **Action:** read the original paper's evaluation section when arxiv.org is reachable.
- **Noise sources:** the filtered release keeps examples where >= 2 raters agree; raters could mark an example "very unclear or difficult to label"; the README warns that bias "likely affect[s] labelling, precision, and recall for a trained model" and that annotators were native English speakers from India ([README](https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md)). Agreement figures in secondary summaries differ (Cohen's kappa about 0.44-0.45, Krippendorff's alpha about 0.30-0.31, unverified; [emergentmind](https://www.emergentmind.com/topics/goemotions-dataset), [koaning blog](https://koaning.io/posts/go-emotions)). Per-annotator label-count inconsistency is blamed for part of the noise, and label correction in a multitask setup improved results ([Noise Correction on Subjective Datasets, arXiv 2311.00619](https://arxiv.org/pdf/2311.00619)). A small 2026 re-annotation pilot (31 training items, 64 binary label changes) showed slight gains on GoEmotions but no consistent benefit across datasets ([search result summary, 2026 preprint](https://www.emergentmind.com/papers/2606.21069); single-source). A commentary claims 0.46 F1 against consensus beat 87% of individual annotators ([kucharski substack](https://kucharski.substack.com/p/when-is-the-ground-truth-not-quite-the-whole-truth), secondary, single-source) which, if true, suggests the practical ceiling is not far above the baseline.
- **What to expect (inference):** realistic macro-F1 for a good 28-label model is about 0.45-0.55 with tuned thresholds; Ekman-6 about 0.6-0.7; sentiment about 0.7. Because the corpora are Reddit comments from crowd labellers, **do not expect these numbers on the owner's own chat text**; the Phase 5 personal study is the real test. Anything above about 0.6 macro-F1 on the 28 labels should trigger a leakage check (duplicate texts across splits, a model that trained on the test split).
- **Mitigations (inference):** use the multi-rater file for soft labels, evaluate on the rater-agreed subset *and* on all examples, report per-label F1 (some labels like gratitude, amusement, love have high agreement; grief, relief, pride, realization have low), and pre-register the metric definition (macro-F1 over 28 labels, threshold tuned on val) in `eval/reports`.

---

## Machine-readable

```json
{
  "recommendations": [
    {
      "area": "clustering",
      "choice": "L2-normalised sentence embeddings -> UMAP -> HDBSCAN (leaf) for fine neighbourhoods -> Ward agglomerative (scipy) on cluster centroids for the coarse-to-fine hierarchy; report soft membership / nearest neighbourhoods",
      "params": {"umap": {"n_neighbors": 30, "min_dist": 0.0, "n_components": 12, "metric": "cosine"}, "hdbscan": {"min_cluster_size": "30-100 for ~50k texts", "min_samples": "5-10", "cluster_selection_method": "leaf", "alpha": 1.0}, "ward_cut_superclusters": "5-10"},
      "why": "UMAP docs recommend larger n_neighbors, min_dist 0 and ~10 dims for clustering; hdbscan docs say leaf gives many small homogeneous clusters; BERTopic builds its hierarchy by Ward on cluster representations, not from the HDBSCAN tree. Cowen & Keltner report categories bridged by continuous gradients, so keep the map soft.",
      "sources": ["https://raw.githubusercontent.com/lmcinnes/umap/master/doc/clustering.rst", "https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/master/docs/parameter_selection.rst", "https://raw.githubusercontent.com/MaartenGr/BERTopic/master/docs/getting_started/hierarchicaltopics/hierarchicaltopics.md", "https://neurosciencenews.com/emotion-27-states-7431/"]
    },
    {
      "area": "clustering",
      "choice": "Validate with DBCV + noise fraction, mean pairwise ARI over >=20 seeds, per-cluster bootstrap Jaccard (80% subsamples), NMI vs GoEmotions/Plutchik as a sanity check; ship hierarchy as exploratory if ARI < 0.7 (fine) / 0.85 (coarse)",
      "params": {"seeds": 20, "bootstrap_runs": "50-100", "jaccard_min": 0.6, "ari_fine_min": 0.7, "ari_coarse_min": 0.85},
      "why": "DBCV is penalised by noise and can be biased, so pair it with noise rate; seed-to-seed ARI of UMAP+HDBSCAN pipelines in the literature ranges 0.6-0.88, so stability must be measured, not assumed. Thresholds are my inference.",
      "sources": ["https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf", "https://arxiv.org/pdf/2503.00127v2", "https://arxiv.org/pdf/2610.01445", "https://arxiv.org/pdf/2602.19079"]
    },
    {
      "area": "fusion",
      "choice": "Baseline: confidence-weighted late fusion. Main: per-label stacking logistic regression on out-of-fold, group-aware engine logits plus presence mask; modality dropout during meta-training; 4 meta-models for {keystroke, LLM-appraisal} availability patterns",
      "params": {"folds": 5, "group_by": "author/thread/conversation", "meta_features_per_label": "one logit per available engine + mask", "modality_dropout_p": 0.2},
      "why": "Stacking on out-of-fold predictions avoids leakage and learns weights with few parameters; modality dropout improves performance with missing modalities in audiovisual emotion work; late fusion is the weakest in the cited multimodal papers (abstract-level). No head-to-head late vs stacking benchmark found; confirm via the Phase 4 ablation.",
      "sources": ["https://arxiv.org/pdf/2201.11095.pdf", "https://arxiv.org/pdf/2505.04642v1", "https://arxiv.org/pdf/2311.10119v1"]
    },
    {
      "area": "calibration",
      "choice": "Per engine and label: Platt (sigmoid) on logits; fallback global temperature + per-label bias for labels with <100 calibration positives; isotonic only with >=2000 calibration samples and >=200 positives and only if it beats Platt on held-out log-loss; recalibrate the fused output last on a separate fold",
      "params": {"min_positives_platt": 100, "isotonic_min_samples": 2000, "isotonic_min_positives": 200},
      "why": "Platt beats isotonic below ~2000 calibration samples; per-label temperature has been used for multi-label emotion output; Dirichlet calibration has ~K^2 parameters and is a softmax method, too heavy for 28 sparse sigmoids.",
      "sources": ["https://arxiv.org/pdf/1207.1403", "https://par.nsf.gov/servlets/purl/10441271", "https://arxiv.org/pdf/1910.12656", "https://ar5iv.labs.arxiv.org/html/1706.04599"]
    },
    {
      "area": "calibration",
      "choice": "Report macro-averaged per-label ECE with adaptive equal-mass bins, plus Brier, log-loss and reliability diagrams with bootstrap bands; mark labels with <50 positives as not assessable",
      "params": {"bins": "min(15, n_label/50)", "norm": "L1 for reporting, L2 allowed for optimisation"},
      "why": "Adaptive binning gives stable metric rankings; multi-label ECE must be per-label because pooled binning degenerates (most scores near zero).",
      "sources": ["https://arxiv.org/abs/1904.01685v2", "https://arxiv.org/pdf/2609.26468", "https://arxiv.org/html/2203.08958v4"]
    },
    {
      "area": "calibration",
      "choice": "VAD intervals by split conformal prediction per dimension, scores normalised by engine vad.std, clipped to [-1,1]; verify empirical coverage on test. overall_confidence = learned logistic model on out-of-fold data (fused max prob, entropy, mean pairwise JSD, VAD disagreement, coverage, engines present); mixed_or_ambiguous from conformal set size>=2 across different Plutchik families or JSD above a validation-tuned threshold",
      "params": {"alpha": 0.1, "calibration_n_min": 200, "flag_rate_alarm": 0.4},
      "why": "Split conformal gives finite-sample marginal coverage; normalised scores handle heteroscedasticity; a learned confidence is calibrated by construction unlike a hand-built product. The overall-confidence design is my inference; no source defines it.",
      "sources": ["https://arxiv.org/pdf/1905.03222", "https://www.tidymodels.org/learn/models/conformal-regression/", "https://www.oreilly.com/library/view/practical-guide-to/9781805122760/B19925_07.xhtml"]
    },
    {
      "area": "temporal",
      "choice": "EMA as baseline; Kalman local-level model on [V,A,D] with measurement noise from engine std, process noise scaled by time gap, update skipped when no data, as main candidate; HMM only as optional discrete layer after ~200 observations per user; choose by one-step-ahead error, response lag and false-trend rate on synthetic scripted conversations",
      "params": {"ema_alpha": "0.2-0.4 (tune)", "kalman_state": "3 independent local levels (or AR(1) around personal mean)", "hmm_states": "3-4", "hmm_min_obs": 200},
      "why": "A Kalman filter on EMA mood data outperformed VAR in a >700-participant study and natively handles missing data, irregular intervals and measurement error; HMM mood studies used ~500 observations per person in clinical samples.",
      "sources": ["https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11449358/", "https://pubmed.ncbi.nlm.nih.gov/39312537/", "https://research-portal.uu.nl/en/publications/evidence-for-mood-instability-in-patients-with-bipolar-disorder-a/"]
    },
    {
      "area": "temporal",
      "choice": "Personal baseline = Welford online mean/variance with exponential forgetting, shrunk to a population prior; withhold or flag z-scores until n>=20-30 per feature; report SD, lag-1 autocorrelation (inertia) and MSSD separately, computed on smoothed state estimates",
      "params": {"prior_pseudo_obs_n0": "10-20", "half_life_days": 14, "min_n_for_z": 25},
      "why": "Shrinkage is the standard cold-start remedy; inertia and MSSD are the established within-person dynamics indices but MSSD mixes variability and autocorrelation.",
      "sources": ["https://metricgate.com/docs/empirical-bayes-estimation/", "https://findanexpert.unimelb.edu.au/scholarlywork/323690-emotional-inertia-and-psychological-maladjustment", "https://documentserver.uhasselt.be/bitstream/1942/24009/2/Vansteelandt%20K%20Revision%20Multivariate%20Behavioral%20Research%20Blinded%20Text%20Revision%203.pdf"]
    },
    {
      "area": "evaluation",
      "choice": "VAD: CCC headline per dimension (+ Pearson, MAE, RMSE, conformal coverage). Multi-label: macro-F1 primary, micro-F1, per-label F1, PR/ROC-AUC; per-label thresholds tuned on validation only, global-threshold fallback for labels with <50 validation positives",
      "params": {"min_val_positives_for_own_threshold": 50},
      "why": "CCC penalises offset and scale errors that Pearson ignores; thresholds tuned on test leak.",
      "sources": ["https://arxiv.org/pdf/1901.02839", "https://arxiv.org/pdf/2003.10724v2"]
    },
    {
      "area": "evaluation",
      "choice": "Group-level bootstrap CIs: percentile B=2000 for exploration, BCa B=9999 for headline numbers (scipy.stats.bootstrap); paired bootstrap on metric differences for fusion-vs-single-engine claims; CheckList MFT/INV/DIR tests reported as pass rates with Wilson 95% intervals",
      "params": {"B_explore": 2000, "B_headline": 9999, "method": "BCa", "resample_unit": "group"},
      "why": "SciPy defaults to BCa with 9999 resamples; BCa needs more resamples than percentile; CheckList is the standard behavioural-testing framework and maps directly onto the brief's negation/intensity/invariance/robustness tests.",
      "sources": ["https://docs.scipy.org/doc/scipy-1.9.1/reference/generated/scipy.stats.bootstrap.html", "https://www.tqmp.org/RegularArticles/vol21-3/p125/p125.pdf", "https://homes.cs.washington.edu/%7Emarcotcr/acl20_checklist.pdf"]
    },
    {
      "area": "evaluation",
      "choice": "Expect GoEmotions macro-F1 around 0.45-0.55 (28 labels), ~0.6-0.7 Ekman-6, ~0.7 sentiment; treat >0.6 on 28 labels as a leakage suspect; use multi-rater file for soft labels; do not expect these numbers on the owner's own chat text",
      "params": {"authors_reported_avg_F1": {"28_labels": 0.46, "ekman6": 0.64, "sentiment": 0.69}},
      "why": "Authors report average F1 0.46/0.64/0.69 with BERT-base; agreement is only moderate and raters were crowd workers; the range above is my inference, no published human ceiling was found.",
      "sources": ["https://arxiv.org/abs/2005.00547v2", "https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md", "https://arxiv.org/pdf/2311.00619"]
    }
  ],
  "libraries": [
    {"name": "scipy (cluster.hierarchy, stats.bootstrap)", "licence": "BSD-3-Clause", "last_release": "not checked (core dependency)", "verdict": "use"},
    {"name": "scikit-learn (HDBSCAN, calibration, model_selection)", "licence": "BSD-3-Clause", "last_release": "not checked (core dependency)", "verdict": "use"},
    {"name": "umap-learn", "licence": "BSD-3-Clause", "last_release": "0.6.0 (2026-11-10); previous stable 0.5.12 (2026-04-08)", "verdict": "use"},
    {"name": "hdbscan", "licence": "BSD-3-Clause", "last_release": "0.8.44 (2026-06-01)", "verdict": "maybe"},
    {"name": "bertopic", "licence": "MIT", "last_release": "0.17.4 (2025-12-03); 1.0.0b1 (2026-11-01) beta", "verdict": "avoid"},
    {"name": "pykalman", "licence": "BSD-3-Clause", "last_release": "0.11.2 (2026-01-31)", "verdict": "use"},
    {"name": "statsmodels", "licence": "BSD-3-Clause", "last_release": "0.15.0 (2026-08-27)", "verdict": "use"},
    {"name": "filterpy", "licence": "MIT", "last_release": "2.0.0 (2026-10-26) after 1.4.5 (2018-10-10); eight-year gap, provenance unverified", "verdict": "avoid"},
    {"name": "hmmlearn", "licence": "BSD (classifier only; confirm LICENSE file in repo)", "last_release": "0.3.3 (2024-10-31)", "verdict": "maybe"},
    {"name": "dynamax", "licence": "MIT", "last_release": "1.0.3 (2026-10-13)", "verdict": "avoid"},
    {"name": "pomegranate", "licence": "MIT", "last_release": "1.1.2 (2025-02-07)", "verdict": "avoid"},
    {"name": "crepes", "licence": "BSD", "last_release": "0.9.1 (2026-06-12)", "verdict": "maybe"},
    {"name": "MAPIE", "licence": "BSD-3-Clause", "last_release": "1.6.0 (2026-10-19)", "verdict": "maybe"},
    {"name": "netcal", "licence": "Apache-2.0", "last_release": "1.4.0 (2026-04-16)", "verdict": "maybe"}
  ]
}
```

Notes on the library verdicts (dates and licences from the PyPI JSON API, fetched 2026-11-18; "use" means no concern found, not that I tested it): `bertopic` is avoided only because it pulls a heavy dependency set for a feature (Ward on centroids) that scipy gives us; `dynamax` needs JAX, which is poor value on a 4-core CPU box (ADR-0003); `pomegranate` is torch-based and its HMM features are more than we need.

## Open questions / things the user must do

1. **Allow more hosts** (ask the owner) if deeper paper reading is wanted: `arxiv.org`, `pmc.ncbi.nlm.nih.gov`, `*.readthedocs.io` (hdbscan, umap-learn, pykalman, hmmlearn docs), `www.pnas.org`, `scikit-learn.org`. Several claims above are abstract-level for lack of these.
2. **Gemini key is invalid** (`scripts/watch.sh` returned "HTTP 400: API key not valid"); 0 of 3 videos watched. Attempted: one talk on how HDBSCAN works (YouTube id dGsxd67IFiU; I believe it is Leland McInnes' HDBSCAN talk but could not confirm). The owner needs to re-issue/store the `GEMINI_API_KEY` secret on the `emotion-engine` environment.
3. **Verify filterpy 2.0.0 provenance** if anyone wants it (maintainer, repo commits, diff vs 1.4.5); until then do not add it.
4. **Decide the metric definition for GoEmotions** (macro-F1 over 28 labels vs 27+neutral) and read the original paper's evaluation section for the exact meaning of its ".46 average F1".
5. **GoEmotions data licence** was not stated in the README I read; defer to the data-sources researcher's note.
6. Decision for the owner at the Phase 3 checkpoint: keep `sklearn.cluster.HDBSCAN` (ADR-0004) or switch engine F to `hdbscan` for soft membership; I recommend keeping sklearn unless soft membership is wanted.

## Sources

1. HDBSCAN parameter-selection doc (GitHub source): https://raw.githubusercontent.com/scikit-learn-contrib/hdbscan/master/docs/parameter_selection.rst (min_cluster_size, min_samples, leaf vs eom, alpha guidance; deep-read)
2. UMAP "Using UMAP for Clustering" (GitHub source): https://raw.githubusercontent.com/lmcinnes/umap/master/doc/clustering.rst (recommended UMAP params, density caveats; deep-read)
3. scikit-learn HDBSCAN source: https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/cluster/_hdbscan/hdbscan.py (defaults, tree not public; deep-read)
4. BERTopic hierarchical topics docs source: https://raw.githubusercontent.com/MaartenGr/BERTopic/master/docs/getting_started/hierarchicaltopics/hierarchicaltopics.md (Ward on c-TF-IDF; deep-read)
5. Cowen & Keltner 2017 coverage: https://www.sci.news/othersciences/psychology/27-categories-emotion-05212.html and https://neurosciencenews.com/emotion-27-states-7431/ (27 categories, gradients; secondary)
6. DBCV paper: https://www.dbs.ifi.lmu.de/~zimek/publications/SDM2014/DBCV.pdf ; R docs https://rdrr.io/cran/dbscan/man/dbcv.html ; critique https://arxiv.org/pdf/2503.00127v2 (abstract-level)
7. Seed-stability preprints: https://arxiv.org/pdf/2610.01445 , https://arxiv.org/pdf/2602.19079 , https://arxiv.org/pdf/2608.31017 , https://arxiv.org/pdf/2604.00074 ; https://datarock.com.au/?p=667 (ARI values, search-result level)
8. Multimodal fusion / modality dropout: https://arxiv.org/pdf/2201.11095.pdf , https://arxiv.org/pdf/2505.04642v1 , https://arxiv.org/pdf/2311.10119v1 (abstract-level)
9. Guo et al. 2017: https://ar5iv.labs.arxiv.org/html/1706.04599 (temperature scaling)
10. Kull et al. 2019 Dirichlet calibration: https://arxiv.org/pdf/1910.12656 , https://proceedings.neurips.cc/paper/2019/hash/8ca01ea920679a0fe3728441494041b9-Abstract.html
11. Nixon et al.: https://arxiv.org/abs/1904.01685v2 (adaptive binning, class-conditional ECE)
12. Multi-label calibration: https://arxiv.org/pdf/2609.26468 (per-label ECE and binning; single-source), https://par.nsf.gov/servlets/purl/10441271 (per-emotion temperature; single-source), https://arxiv.org/html/2203.08958v4 (classwise)
13. Niculescu-Mizil & Caruana: https://arxiv.org/pdf/1207.1403 (Platt vs isotonic below ~2000 samples)
14. Conformal regression: https://arxiv.org/pdf/1905.03222 (CQR), https://www.tidymodels.org/learn/models/conformal-regression/ , https://www.oreilly.com/library/view/practical-guide-to/9781805122760/B19925_07.xhtml
15. Kalman on EMA mood: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11449358/ , https://pubmed.ncbi.nlm.nih.gov/39312537/ ; state-space EMA considerations https://arxiv.deeppaper.ai/papers/2305.03214v1?locale=en
16. HMM mood: https://research-portal.uu.nl/en/publications/evidence-for-mood-instability-in-patients-with-bipolar-disorder-a/ , https://zenodo.org/doi/10.5281/zenodo.15775394
17. Inertia / MSSD: https://findanexpert.unimelb.edu.au/scholarlywork/323690-emotional-inertia-and-psychological-maladjustment , https://documentserver.uhasselt.be/bitstream/1942/24009/2/Vansteelandt%20K%20Revision%20Multivariate%20Behavioral%20Research%20Blinded%20Text%20Revision%203.pdf
18. Shrinkage / cold start (non-authoritative): https://metricgate.com/blogs/bayesian-hierarchical-models-explained/ , https://metricgate.com/docs/empirical-bayes-estimation/
19. CCC: https://arxiv.org/pdf/1901.02839 , https://arxiv.org/pdf/2003.10724v2 , https://hyper.ai/en/sota/tasks/a-vb-two
20. Bootstrap: https://docs.scipy.org/doc/scipy-1.9.1/reference/generated/scipy.stats.bootstrap.html , https://www.tqmp.org/RegularArticles/vol21-3/p125/p125.pdf , https://metricgate.com/blogs/choosing-number-of-bootstrap-replicates/ (non-authoritative)
21. CheckList: https://homes.cs.washington.edu/%7Emarcotcr/acl20_checklist.pdf , https://ar5iv.labs.arxiv.org/html/2005.04118
22. GoEmotions: https://arxiv.org/abs/2005.00547v2 , https://ar5iv.labs.arxiv.org/html/2005.00547 , https://raw.githubusercontent.com/google-research/google-research/master/goemotions/README.md (README deep-read); noise: https://arxiv.org/pdf/2311.00619 , https://www.emergentmind.com/papers/2606.21069 , https://arxiv.org/pdf/2604.27345 , https://koaning.io/posts/go-emotions , https://kucharski.substack.com/p/when-is-the-ground-truth-not-quite-the-whole-truth (secondary)
23. PyPI JSON metadata (deep-read for versions/dates/licences): https://pypi.org/pypi/filterpy/json , /pykalman/json , /hmmlearn/json , /hdbscan/json , /umap-learn/json , /bertopic/json , /statsmodels/json , /mapie/json , /crepes/json , /netcal/json , /dynamax/json , /pomegranate/json
