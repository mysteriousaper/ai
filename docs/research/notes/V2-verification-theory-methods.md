# V2 - Verification of R3 (theory, safety, legal) and R4 (methods, math)

Verifier: V2. Date: 2026-11-18. **Not legal advice.**

**Access limits in this session (read first).** Reachable: `pypi.org`, `files.pythonhosted.org`,
`raw.githubusercontent.com`, `huggingface.co` (API only), WebSearch. **Blocked** (DNS / proxy refusal):
`eur-lex.europa.eu`, `artificialintelligenceact.eu`, `curia.europa.eu`, `digital-strategy.ec.europa.eu`,
`arxiv.org`, `aclanthology.org`, `cs.cornell.edu`, `deepai.org`, `datasets-server.huggingface.co`;
`gh api` is refused for repos not attached to the session. So **no legal claim could be checked
against eur-lex itself**. Legal verdicts below rest on (a) my own knowledge of the Regulation text and
(b) agreement across several independent secondary sources found by search. They are marked
`verified (secondary + memory)`, which is weaker than a primary read. Library and dataset licence claims
were checked against **primary files** (PyPI metadata, wheel/sdist contents, LICENSE files on GitHub raw,
HF API tags). Spikes ran in a throwaway venv in the scratchpad (Python 3.13), never the repo venv.

## Verdict summary

Counts (64 claims checked): **verified 51** (many of the legal ones only at secondary-plus-memory
strength, see access limits), **corrected 6**, **unverified 6**, **wrong 1**.

Most important corrections:

1. **pykalman is NOT plain BSD-3 (R4 is wrong on this).** Since **0.11.1 (2025-12-31)** its LICENSE is
   "New BSD" with an asterisk ("permitted*") plus a Section 2 that requires a **paid commercial
   distribution licence (USD 10,000/year + USD 1 per download)** for anyone redistributing it via
   Anaconda, NumFOCUS infrastructure or **conda-forge** "and any mirrors, forks, or derivative
   services thereof", under German law (Munich court). PyPI distribution and end users are explicitly
   exempt (s. 2.2, 2.3). 0.11.0 (2025-11-27) and earlier carry the plain BSD text. Its README badge
   still says "BSD 3-clause". Class: **commercial-OK only via PyPI, non-standard licence**. Recommendation:
   use `statsmodels` (BSD-3) or the ~25-line hand-rolled numpy Kalman that R4 already proposes; if
   pykalman is used at all, pin `pykalman==0.11.0` and never ship it in a conda environment.
2. **Empath on PyPI has no licence metadata** (`License: UNKNOWN`, no LICENSE file in the sdist), not
   "Other" as R3 says. But the shipped lexicon `empath/data/categories.tsv` is **byte-identical**
   (md5 `3d8ffefc0a5a342ecaa4afb98f2cb480`) to the file in the MIT-licensed GitHub repo, so the MIT grant
   covers it. Class: **commercial-OK**, with the MIT notice copied into `THIRD_PARTY_NOTICES.md` (the
   PyPI package does not carry it). New finding: `Empath.create_category()` POSTs seed terms over plain
   HTTP to a hard-coded IP (`http://54.148.189.209:8000`); `analyze()` is local. **Never call
   `create_category` at runtime** (privacy, brief §1.8).
3. **Isotonic "only above ~2,000 calibration samples" is not supported by the cited source.** Search-level
   evidence for Niculescu-Mizil & Caruana (ICML 2005) puts the Platt-beats-isotonic region at roughly
   **200-1,000** cases; no "2,000" was found. The rule is conservative, so harmless, but cite it as
   project heuristic, not as the paper's number.
4. **ESConv's HF card main revision is `cc-by-nc-4.0`** (not only a PR revision as R3 says). It still
   conflicts with the README ("academic research use only"). Either way: **research-only**. Confirmed.
5. **GoEmotions ".46" is macro-F1** (macro P .40 / R .63 / F1 .46 per the reproduction paper's table of
   the original), so R4's secondary "0.46 micro / 0.43 macro" is wrong; and R4 is internally
   inconsistent on the leakage alarm (summary: "much above 0.55"; body and JSON: "above 0.6"). Use **0.6**:
   tuned RoBERTa-class models already reach ~0.5-0.55, so 0.55 would raise false alarms.

Legal headline: R3's core conclusions **hold**. Art. 5(1)(f) applies from 2 Feb 2025; the definition is
tied to biometric data; the Commission guidelines exclude written text and include keystrokes; the
Digital Omnibus (Reg. (EU) 2026/1744, OJ 24 Jul 2026, in force 27 Jul 2026 per several secondary sources)
moved Annex III to 2 Dec 2027 and did not change 5(1)(f). **Keystroke engine D must stay off for
education and workplace deployments.** All of this still needs one eur-lex read before it goes into a
product decision record.

## Claim table

### A. Legal (R3 section 6)

| # | Claim (short) | Source note | Verdict | Evidence |
|---|---|---|---|---|
| L1 | Art. 5(1)(f) text: bans placing on market / putting into service for this purpose / use of AI systems "to infer emotions of a natural person in the areas of workplace and education institutions, except ... for medical or safety reasons" | R3 6.1 | verified (memory + secondary) | Matches the OJ text of Reg. (EU) 2024/1689 as I know it; same wording in [FPF](https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/), [Reglog](https://reglog.io/blog/eu-ai-act-emotion-recognition-workplace). eur-lex blocked. |
| L2 | Prohibitions apply from 2 Feb 2025 (Art. 113(a)); penalties from 2 Aug 2025 | R3 6.1 | verified (secondary + memory), with nuance | Art. 113: Chapters I and II from 2 Feb 2025; "Chapter III Section 4, Chapter V, Chapter VII and Chapter XII and Article 78 shall apply from 2 August 2025, with the exception of Article 101" ([springlex Art. 113](https://www.springlex.eu/en/packages/ai-act/ai-act-regulation/article-113/), [Mishcon](https://www.mishcon.com/eu-uk-ai-navigator/article-113), [Noerr](https://www.noerr.com/en/insights/2-august-2025-another-milestone-towards-the-full-applicability-of-the-ai-act)). Nuance: Art. 99 obliges Member States to lay down penalties; actual fining depends on national law. |
| L3 | Fines up to EUR 35m or 7% of worldwide turnover for prohibited practices | R3 6.5 | verified (memory) | Art. 99(3): up to EUR 35,000,000 or up to 7 % of total worldwide annual turnover, whichever is higher. Not re-read on eur-lex. |
| L4 | Art. 3(39): emotion recognition system = inferring emotions or intentions "on the basis of their biometric data" | R3 6.2 | verified (memory + secondary) | Same wording in all summaries above. Art. 3(34) biometric data covers "behavioural characteristics", which is why keystrokes fall in. |
| L5 | Guidelines: written text is outside; keystrokes are inside | R3 6.1 | verified (secondary, 2+ independent) | "behavioural biometrics such as keystrokes, or body postures are also included, whereas written text, even where that contains emotional language, is not" and "inference from key stroke or body postures or movements would be prohibited" ([Oliver Patel](https://oliverpatel.substack.com/p/emotion-recognition-and-the-eu-ai), [DLA Piper](https://www.dlapiper.com/en/insights/publications/ai-outlook/2025/european-commission-publishes-guidelines-on-prohibited-ai-practices), [Inside Tech Law](https://www.insidetechlaw.com/blog/2025/03/prohibited-practices-under-the-ai-act-answered-and-unanswered-questions)). |
| L6 | Guidelines "published early February 2025" | R3 6.1 | unverified (version question) | The 4 Feb 2025 text was a Commission-approved draft. One source says the operative text is C(2025) 5052 final of 29 July 2025 ([Reglog](https://reglog.io/blog/eu-ai-act-emotion-recognition-workplace), single-source). A human must check which version is current on the Commission site before quoting paragraph numbers. |
| L7 | Medical exception does not cover burnout/depression detection at work/school | R3 6.1 | verified (secondary) | FPF and CMS summaries, consistent. |
| L8 | Annex III 1(c) lists emotion recognition as high-risk (where permitted) | R3 6.2 | verified (memory) | Annex III point 1 "Biometrics, in so far as their use is permitted under relevant Union or national law": (c) "AI systems intended to be used for emotion recognition". Note: this too is biometric-based, so text-only engines are probably outside Annex III 1(c) as well. |
| L9 | Digital Omnibus = Reg. (EU) 2026/1744, OJ 24 Jul 2026, in force 27 Jul 2026 | R3 6.2 (flagged single-source) | verified (secondary, now multi-source) | Same number and dates in [aiactguide](https://aiactguide.net/digital-omnibus/), [Acompli](https://acompli.ie/news/ai-omnibus-enters-into-force-july-2026/), [Usercentrics](https://usercentrics.com/knowledge-hub/eu-ai-act-high-risk-delay-article-50-transparency-consent/). Still not read on eur-lex. |
| L10 | Annex III high-risk moved to 2 Dec 2027; Annex I to 2 Aug 2028 | R3 6.2 | verified (secondary) | [Gibson Dunn](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/), [CSA](https://labs.cloudsecurityalliance.org/research/csa-research-note-eu-ai-act-high-risk-deadline-omnibus-20260/), [SetAIComply](https://www.setaicomply.com/resources/eu-ai-act-new-2027-deadlines-digital-omnibus). |
| L11 | Omnibus leaves 5(1)(f) unchanged; adds two new prohibitions (intimate imagery, CSAM) | R3 6.2 | verified (secondary) | New Art. 5(1)(ba)/(bb), applying from 2 Dec 2026 per several sources; 5(1)(f) reported unchanged ([techjacksolutions](https://techjacksolutions.com/ai-brief/eu-ai-omnibus-prohibitions-csam-intimate-imagery-bans/), [VdA](https://www.vda.pt/fr/publications/insights/what-has-the-digital-omnibus-on-ai-changed-in-practice/28745/)). One blog renumbers the emotion ban "5(1)(j)"; treat that as an error. |
| L12 | Art. 50 transparency stays 2 Aug 2026; "Art. 50(2) watermarking moves to 2 Dec 2026" | R3 6.2 | corrected | Art. 50 applied 2 Aug 2026. The 2 Dec 2026 date is reported as a **transition for systems already on the market** for 50(2), not a general move ([Usercentrics](https://usercentrics.com/knowledge-hub/eu-ai-act-high-risk-delay-article-50-transparency-consent/)). Sources disagree on details. Not relevant to us unless we generate synthetic content. |
| L13 | Art. 50(3): deployers of emotion recognition must inform exposed persons | R3 6.2 | verified (memory) | Text as R3 gives it; applies only to emotion recognition systems as defined (biometric), so mainly engine D. |
| L14 | GDPR: inference revealing sensitive data is special-category (CJEU C-184/20) | R3 6.3 | verified (secondary) | Grand Chamber, 1 Aug 2022, *OT v Vyriausioji tarnybinės etikos komisija*, ECLI:EU:C:2022:601: data "liable to disclose indirectly" special categories fall under Art. 9 ([Inside Privacy](https://www.insideprivacy.com/eu-data-protection/special-category-data-by-inference-cjeu-significantly-expands-the-scope-of-article-9-gdpr/), [Matheson](https://publications.matheson.com/data-protection-technology-and-cyber-security-bulletin-october-2022/cjeu-expands-scope-of-special-category-data)). R3's "treat mood/risk inference as Art. 9" is labelled inference and is the prudent reading. |
| L15 | Keystrokes are GDPR Art. 9 biometric only when used for unique identification | R3 6.3 | verified (memory) | Art. 9(1) "biometric data for the purpose of uniquely identifying a natural person"; Art. 4(14) definition itself requires data that "allow or confirm the unique identification". The AI Act's Art. 3(34) is broader. R3 is right. |
| L16 | Armenia 2015 law, prior notification to the Agency for biometric/special data | R3 6.4 | unverified | Search-level only in R3; hosts not reachable here; out of my priority scope. Needs a human or counsel. |

### B. Licences: lexicons, datasets, models (R3 sections 2-5)

| # | Claim (short) | Source note | Verdict | Evidence |
|---|---|---|---|---|
| D1 | Empath client repo is MIT | R3 4 | verified | "Copyright (c) 2016 Ethan Fast. Permission is hereby granted, free of charge, ... to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies" - https://raw.githubusercontent.com/Ejhfast/empath-client/master/LICENSE.txt (HTTP 200). **commercial-OK**. |
| D2 | Empath PyPI package licence shows "Other"; lexicon terms unconfirmed | R3 4 | corrected | PyPI `empath` 0.89 (2016-06-06): `License: UNKNOWN`, no classifiers, no LICENSE file in `empath-0.89.tar.gz` (57,627 B). `categories.tsv` (135,655 B, 194 lines) is md5-identical to the GitHub master file, so MIT applies. Ship the MIT notice ourselves. |
| D3 | Empath has about 200 categories (194 in client) | R3 4 | verified | Spike: `len(Empath().cats) == 194`. |
| D4 | Empath "not updated since release" | R3 4 | verified | Last PyPI upload 2016-06-06. Unmaintained; vendor the TSV rather than depend on the package. New: `create_category` sends terms to `http://54.148.189.209:8000` (core.py line 9/64). |
| D5 | ESConv: "academic research use only", all rights reserved | R3 3.1 | verified | README line 2: "Copyright © 2021 CoAI Group, Tsinghua University. All rights reserved. Data and codes are for academic research use only." https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/README.md. **research-only**. |
| D6 | ESConv HF card lists cc-by-nc-4.0 only in a PR revision | R3 3.1 | corrected | HF API `datasets/thu-coai/esconv`, revision `main` (sha 32c54c1...): `cardData.license = cc-by-nc-4.0`. Conflict with README remains; stricter wins: **research-only**. |
| D7 | ESConv 1,300 conversations, 10 topics, 196 failed samples, strategy counts, "Other" 3,661 / 3,341 | R3 3.1 | verified (with a note) | Numbers match the README exactly. Note: the README says "10 topic problems" but its table lists 12 categories (5 old with counts + 7 new). |
| D8 | EmpatheticDialogues licence CC BY-NC 4.0 (R3: "from memory, not confirmed") | R3 2 | verified | Repo LICENSE line 1: "Attribution-NonCommercial 4.0 International" (https://raw.githubusercontent.com/facebookresearch/EmpatheticDialogues/main/LICENSE, 19,332 B); HF `facebook/empathetic_dialogues` tag `cc-by-nc-4.0`. **research-only** for a sold product. |
| D9 | The 32 ED labels as listed | R3 2 | verified | Extracted from the local ED csv (already in scratchpad): exactly 32 labels, identical to R3's list. All 32 appear in R3's event table. |
| D10 | GoEmotions data licence (R4: not stated in README) | R4 open q. 5 | verified (note) | README has no licence; HF `google-research-datasets/go_emotions` tag `apache-2.0`. R1/V1 own this; flagging only. |
| D11 | Llama Guard 3 under Llama Community Licence, "not commercial-OK without reading it" | R3 5 | verified (classify needs-licence) | HF `meta-llama/Llama-Guard-3-8B`: tag `license:llama3.1`, `gated: manual`. Commercial use is allowed under conditions (AUP, 700M MAU clause); class **needs-licence** (read and accept terms). |
| D12 | Absolutist-word list (~19 words) in paper supplement, not retrieved | R3 4 | unverified | Not reachable here. My recollection is a 19-word dictionary; R3's caution stands. |
| D13 | Find a Helpline terms forbid scraping and AI training | R3 5 | unverified | findahelpline.com not reachable. Keep R3's "hand-curate" recommendation. |
| D14 | Hofmann et al. 2020 / x-enVENT / crowd-enVENT licences "not found" | R3 1.4 | unverified (not attempted; hosts blocked) | Treat as research-only until a human reads the IMS Stuttgart page. |

### C. Libraries (R4 section 4.4 and JSON)

| # | Claim (short) | Source note | Verdict | Evidence |
|---|---|---|---|---|
| B1 | **pykalman 0.11.2 is BSD-3, "use"** | R4 4.4 / JSON | **wrong** | PyPI `info.license` and wheel `dist-info/licenses/LICENSE`: "1. New BSD License ... permitted*, ... * subject to provisions in section 2"; "2.4 Separate Distribution License Required ... Distribution through the channels listed above requires a separately executed commercial distribution license"; "2.5 ... initial annual distribution fee of USD 10,000 ... per-download fee of USD 1"; channels: Anaconda, NumFOCUS infra, "conda-forge and any mirrors, forks, or derivative services thereof"; "2.3 ... distribution ... via the Python Package Index ... remains fully governed by Sections 1". Clause present in 0.11.1 and 0.11.2, absent in 0.11.0 and earlier (checked per-version PyPI JSON). Class: **unclear / commercial-OK only via PyPI**. Release dates 0.11.0 2025-11-27, 0.11.1 2025-12-31, 0.11.2 2026-01-31 verified. Spike: installs, imports, EM + masked observations run. |
| B2 | filterpy: 1.4.5 (2018-10-10), then 2.0.0 (2026-10-26), MIT; provenance unverified; avoid | R4 4.4 | verified (with extra evidence) | PyPI JSON confirms the gap. LICENSE on GitHub master is MIT ("Copyright (c) 2015 Roger R. Labbe Jr"). Metadata author is the original author. GitHub master `__init__.py` also says `__version__ = "2.0.0"`; `common/helpers.py` is byte-identical to master, `kalman_filter.py` differs from master only in docstring examples (157 diff lines; master is newer). No network/exec/base64 code found in a grep of the 2.0.0 sdist. Diff vs 1.4.5: 6,556 lines, `monte_carlo/quadrature.py` removed. Points toward a genuine release by the author, but the uploader account and commit history could not be checked (`gh api` refused). Keep **avoid** until a human checks the GitHub release/tag. Spike: imports and runs on Python 3.13 / numpy 2.5. |
| B3 | hmmlearn 0.3.3, BSD, last release 2024-10-31; LICENSE file to confirm | R4 4.4 | verified | LICENSE.txt: "Copyright (c) 2014, hmmlearn authors and contributors ... Redistribution and use in source and binary forms, with or without modification, are permitted" with the "Neither the name" clause (BSD-3-Clause). **commercial-OK**. Additional: README line 29 "**Note**: This package is under limited-maintenance mode." Spike: works but emits a numpy 2.5 `DeprecationWarning` ("Setting the shape on a NumPy array has been deprecated") in `hmmlearn/utils.py:27`, so it will break on a future numpy. Verdict "maybe" stands; prefer hand-rolled forward-backward if an HMM is ever needed. |
| B4 | statsmodels 0.15.0 (2026-08-27), BSD-3 | R4 4.4 | verified | PyPI `license_expression: BSD-3-Clause`, upload 2026-08-27. **commercial-OK**. Best Kalman dependency if we don't hand-roll. |
| B5 | hdbscan 0.8.44 (2026-06-01), BSD; exposes condensed tree, soft membership, validity index | R4 1.2 | verified | PyPI upload 2026-06-01; LICENSE "Copyright (c) 2015, Leland McInnes" with BSD-3 clauses. Spike: `condensed_tree_` present, `hdbscan.validity_index` present. |
| B6 | sklearn HDBSCAN keeps the tree private (`_single_linkage_tree_`) | R4 1.2 | verified | Spike (sklearn 1.9.1): public fitted attrs are `labels_`, `probabilities_`, `n_features_in_`; tree only as `_single_linkage_tree_`. `probabilities_` is membership strength, not soft vectors, so R4's ADR-0004 reading is right. |
| B7 | umap-learn 0.6.0 released 2026-11-10; prior 0.5.12 (2026-04-08); BSD-3 | R4 1.3 | verified | PyPI dates match; LICENSE.txt "BSD 3-Clause License". Spike: 0.6.0 installs and runs on Python 3.13 with the recommended params. Pin advice is reasonable. |
| B8 | crepes 0.9.1 BSD; MAPIE 1.6.0 BSD-3; netcal 1.4.0 Apache-2.0 | R4 JSON | verified | PyPI dates match; crepes LICENSE is BSD-3 text ("Copyright 2026 Henrik Boström", no fee clause); MAPIE `license_expression: BSD-3-Clause`; netcal Apache classifier. |
| B9 | bertopic 0.17.4 / 1.0.0b1; dynamax 1.0.3; pomegranate 1.1.2; all MIT | R4 JSON | verified | PyPI dates and licence fields match. |

### D. Methods and parameters (R4)

| # | Claim (short) | Source note | Verdict | Evidence |
|---|---|---|---|---|
| M1 | UMAP for clustering: raise `n_neighbors` to 30, `min_dist` 0, ~10 dims | R4 1.2 | verified | clustering.rst: "In this case we'll double it from the default 15 up to 30"; "set ``min_dist`` to be 0"; "you can reduce to 10 dimensions instead since the goal is to cluster, not visualize"; also "somewhat controversial", "does not completely preserve density", "false tears" (https://raw.githubusercontent.com/lmcinnes/umap/master/doc/clustering.rst). Spike confirms 0.6.0 defaults 15 / 0.1 / 2 / euclidean. |
| M2 | HDBSCAN: `leaf` gives many small homogeneous clusters; `eom` one or two large plus small extras; don't touch `alpha` | R4 1.2 | verified | parameter_selection.rst: "Excess of Mass has a tendency to pick one or two large clusters and then a number of small extra clusters"; "select ``'leaf'`` ... producing many small homogeneous clusters"; "it is best not to mess with this parameter". Numeric `min_cluster_size` 30-100 is R4's inference (labelled). |
| M3 | BERTopic hierarchy = scipy Ward on c-TF-IDF, not the HDBSCAN tree | R4 1.1 | verified | hierarchicaltopics.md: "approximate this potential hierarchy by making use of our topic-term matrix (c-TF-IDF matrix)" and "As a default, we are using the ward". |
| M4 | Ward on cluster centroids in cosine space | R4 1.2 | verified (method note) | Sound if centroids are re-normalised to unit length (Euclidean on unit vectors is monotone in cosine, which Ward needs). R4 says this. Add: Ward on centroids ignores cluster sizes; pass sizes or use `scipy` on the points if sizes vary a lot. |
| M5 | Seed-stability ARI reference values (0.61-0.876) from four 2026 preprints | R4 1.3 | unverified | arxiv blocked; single-source each. Acceptance bars (0.7 / 0.85) are R4's inference; fine as pre-registered targets. |
| M6 | Platt beats isotonic below ~2,000 calibration cases (Niculescu-Mizil & Caruana) | R4 3.1 | corrected | No "2,000" found; secondary reading of the ICML 2005 paper gives ~200-1,000 ([search result list incl. cs.cornell.edu PDF](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf), not reachable to read). Also, arXiv 1207.1403 is the UAI 2005 boosting paper; the sample-size curves are in the ICML 2005 "Predicting Good Probabilities" paper. Keep the 2,000 rule as a conservative project heuristic. |
| M7 | sklearn `CalibratedClassifierCV` offers sigmoid, isotonic, temperature | R4 3.1 | verified | Spike (sklearn 1.9.1) docstring: "method : {'sigmoid', 'isotonic', 'temperature'}, default='sigmoid'". |
| M8 | Guo 2017 (temperature), Kull 2019 (Dirichlet, NeurIPS), Nixon 2019 (adaptive ECE), Romano 2019 (CQR), Ribeiro 2020 (CheckList, ACL best paper) | R4 3, 5 | verified (citation identity, memory) | arXiv ids 1706.04599, 1910.12656, 1904.01685, 1905.03222, 2005.04118 match those papers. Content claims are standard. |
| M9 | Dirichlet calibration ~K^2 params, too heavy for 28 sparse sigmoids | R4 3.1 | verified (math) | Full-matrix Dirichlet for K=28 has K(K+1)=812 params; it is a softmax method. Correct reasoning. |
| M10 | Split conformal quantile ceil((n+1)(1-alpha))/n; min n=9 for 90%; marginal coverage only | R4 3.3 | verified (math) | n=9: ceil(10*0.9)=9 <= 9. Correct. |
| M11 | 0.9-bin with 100 predictions gives ~+/-6% at 95% | R4 3.2 | verified (math) | sqrt(0.09/100)=0.03; 1.96*0.03 = 0.059. |
| M12 | F1-optimal threshold ~ F1*/2 for calibrated probabilities ("unsourced") | R4 5.2 | verified (source supplied) | Lipton, Elkan & Naryanaswamy 2014, "Optimal Thresholding of Classifiers to Maximize F1 Measure" (arXiv 1402.1892): optimal threshold equals half the optimal F1 for calibrated scores. Cite it. |
| M13 | `scipy.stats.bootstrap` defaults: 9,999 resamples, BCa | R4 5.3 | verified | Spike (scipy 1.18.1): `n_resamples=9999`, `method='BCa'`. |
| M14 | MSSD = 2*SD^2*(1 - AR(1)) | R4 4.2 | verified (math) | E[(x_t - x_{t-1})^2] = 2*var - 2*cov = 2*var*(1 - rho) for a stationary series. |
| M15 | Kalman filter on EMA mood data (>700 participants) beats VAR | R4 4.1 | verified (secondary) | Malamud et al., PLOS Comput Biol 20(9): e1012457, Sept 2024 (PMC11449358, PubMed 39312537): "naive Kalman filter approach performs favourably compared to the standard vector autoregressive approach" ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39312537/)). Note: their data are self-reported mood items, not model estimates; the transfer to noisy text-inferred VAD is our assumption. |
| M16 | GoEmotions ".46 average F1" (28 labels), .64 Ekman, .69 sentiment; secondary says 0.46 micro / 0.43 macro | R4 5.5 | corrected | .46/.64/.69 confirmed from the abstract; .46 is the **macro-average** (macro P .40, R .63, F1 .46 as reproduced in [arXiv 2403.06108 Table 3](https://arxiv.org/html/2403.06108v1), via search). The "0.43 macro" claim is wrong. |
| M17 | Leakage alarm threshold for 28-label macro-F1 | R4 summary vs 5.5 vs JSON | corrected | R4 says "much above about 0.55" (summary) and "above about 0.6" (5.5, JSON). Use 0.6 consistently. |
| M18 | 4 meta-models for {D, E} availability; per-label stacking on out-of-fold, group-aware predictions | R4 2 | verified (reasoning) | 2^2 = 4 is right given A/B/C/F always present. Leakage control (OOF + group folds + separate calibration fold) is standard and correctly specified. No source benchmarks late fusion vs stacking (R4 says so). |

### E. Citations for key papers (R3)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| C1 | Hofmann, Troiano, Sassenberg, Klinger, COLING 2020, 7 dimensions, 1,001 events | verified (memory) | Matches the paper as I know it; anthology id 2020.coling-main.11 is plausible but the host is blocked. |
| C2 | Troiano et al., Computational Linguistics 2023, DOI 10.1162/coli_a_00461 | verified (memory) | Correct venue and DOI pattern; 21 appraisal variables matches. |
| C3 | ESConv: Liu et al., ACL 2021, arXiv 2106.01144 | verified | README links the paper; id matches. |
| C4 | Empath: Fast, Chen, Bernstein, CHI 2016, arXiv 1602.06979, ~0.90 LIWC correlation | verified (memory) | Paper reports r = 0.906 on shared categories. |
| C5 | Edwards & Holtzman 2017 r = .13; Tackman et al. 2019 JPSP N = 4,754 | verified (memory) | Figures match my knowledge of both papers; R3's "use as weak feature" is right. |
| C6 | Kuppens, Allen & Sheeber 2010 inertia; Jahng, Wood & Trull 2008 MSSD | verified (memory) | Psychological Science 2010; Psychological Methods 2008. |

## Machine-readable

```json
{
  "licence_classes": {
    "empath_lexicon_categories_tsv": "commercial-OK (MIT via GitHub repo; PyPI package lacks licence file; never call create_category)",
    "esconv": "research-only",
    "empathetic_dialogues": "research-only (CC BY-NC 4.0, confirmed)",
    "llama_guard_3": "needs-licence (llama3.1 community licence, gated)",
    "pykalman>=0.11.1": "unclear (BSD + paid conda-channel distribution clause; PyPI exempt)",
    "pykalman<=0.11.0": "commercial-OK (BSD-3)",
    "filterpy": "commercial-OK licence (MIT) but provenance of 2.0.0 unverified: avoid",
    "hmmlearn": "commercial-OK (BSD-3), limited-maintenance, numpy 2.5 deprecation warning",
    "statsmodels": "commercial-OK (BSD-3)",
    "hdbscan": "commercial-OK (BSD-3)",
    "umap-learn": "commercial-OK (BSD-3)",
    "crepes": "commercial-OK (BSD-3)",
    "mapie": "commercial-OK (BSD-3)",
    "netcal": "commercial-OK (Apache-2.0)"
  },
  "parameter_corrections": {
    "isotonic_min_samples": "keep 2000 as heuristic; paper suggests ~1000",
    "goemotions_leakage_alarm_macro_f1": 0.6
  },
  "legal": {
    "art5_1_f_applies_from": "2025-02-02",
    "annex_iii_high_risk_from": "2027-12-02 (Reg. (EU) 2026/1744, secondary sources)",
    "keystroke_engine_D_in_scope_of_art5_1_f": true,
    "text_only_engines_in_scope": false,
    "primary_text_checked": false
  }
}
```

## Still needs a human

1. **Allow `eur-lex.europa.eu`** (and ideally `digital-strategy.ec.europa.eu`) for one session and re-read:
   Art. 3(34), 3(39), 5(1)(f), 50(3), 99(3), 113, Annex III 1(c) of Reg. (EU) 2024/1689, and the
   consolidated text after Reg. (EU) 2026/1744. Every legal row above is secondary + memory only.
2. **Which version of the Commission guidelines on prohibited practices is operative** (the 4 Feb 2025
   approved draft vs a reported C(2025) 5052 final of 29 Jul 2025). Paragraph numbers depend on it.
3. **Counsel:** B2C tutoring app outside an institution vs "in the areas of ... education institutions";
   Art. 9 GDPR basis for mood / safety-flag inference; Armenian Agency notification. Unchanged from R3.
4. **pykalman licence decision:** accept the non-standard licence (PyPI only, never conda), pin 0.11.0,
   or (recommended) use statsmodels / hand-rolled Kalman. The project owner decides; record in an ADR.
5. **filterpy 2.0.0 provenance:** check the GitHub releases/tags page and the PyPI uploader in a browser
   (the session cannot call `gh api` on that repo). Not needed if we hand-roll the filter.
6. **Absolutist word list** (Al-Mosaiwi & Johnstone 2018 supplement), **crowd-enVENT / Hofmann** dataset
   licences on the IMS Stuttgart page, and **Find a Helpline terms**: all on hosts this session cannot reach.
7. **Helpline numbers** for Armenia and Russia: still must be confirmed by the owner from official sources.
