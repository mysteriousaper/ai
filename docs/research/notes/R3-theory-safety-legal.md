# R3 - Appraisal theory, event and need taxonomies, psycholinguistic lexicons, crisis safety, legal

Researcher: R3. Date: 2026-11-18. **This is not legal advice.**

**Access caveat (read first).** In this session `WebFetch` could only reach `github.com` and `raw.githubusercontent.com`. Every other host (eur-lex, arxiv, huggingface, aclanthology, law-firm sites, findahelpline) failed DNS. All non-GitHub claims below therefore come from WebSearch result summaries, not from deep reads of the primary page. The brief and agent rules ask for deep reads; I could not do that, so I mark each claim: `[searched]` (search summary only), `[github-read]` (page read in full), or `[memory]` (my own prior knowledge, unverified here). **Before anything legal or licence-related goes into DATA_SOURCES.md or a product, a session with eur-lex.europa.eu, arxiv.org, huggingface.co, aclanthology.org and findahelpline.com allowed must re-check it.** No video was watched (the text sources were enough, and the network was restricted; `scripts/watch.sh` was not attempted).

## Summary

1. Appraisal theory gives a defensible, small set of dimensions to extract from event text: pleasantness / goal congruence, certainty, responsibility (self / other / circumstance), control / coping potential, anticipated effort, attention, and (Scherer only) novelty and norm compatibility. Hofmann et al. (COLING 2020) showed that these dimensions can be annotated in text and that high-quality appraisal labels improve emotion classification. crowd-enVENT (Troiano et al., Computational Linguistics 2023) has 21 appraisal variables self-reported by the event's author. Both are non-commercial-style data, so use them for evaluation and prompt design, not as shipped training data.
2. Proposed event taxonomy (my proposal): 13 types plus `other_or_unclear`, mapped from the 32 EmpatheticDialogues labels. ED labels are emotions, not events, so the mapping is many-to-many and only a weak label. Re-annotate with an LLM, then hand-check about 200 items.
3. Needs: social-support theory (emotional / informational / instrumental / esteem) maps cleanly onto four of our six needs. `distraction`, `rest` and `celebration` have no theory-backed data that I found. ESConv labels supporter strategies, not seeker needs, and is academic-research-only.
4. Open LIWC alternatives: Empath is MIT (verbatim text below) with about 200 built-in categories. The absolutist-word list is in a paper's supplement and I could not retrieve it. A first-person-singular pronoun list is trivial to write ourselves. No permissively licensed hedging or certainty lexicon was found. Build one from open linguistics literature and mark it as our own.
5. Crisis safety: use a layered, recall-biased design (multilingual patterns plus a classifier plus an optional LLM judge), a hard `safety_flag` in the core, and a human-resource display. Do not rely on a free public helpline directory. Find a Helpline's terms forbid scraping, and ThroughLine's API is the licensed route.
6. Legal: the EU AI Act emotion-recognition ban (Art. 5(1)(f)) has applied since 2 Feb 2025. Its definition covers inference from **biometric data**. The Commission's guidelines (as summarised by law firms) say emotion inference from written text is outside the definition, but keystroke patterns count as biometric. **Our text engines are probably outside; the keystroke engine D is inside, so it must be disabled for any education or workplace deployment.** This is the owner's tutoring flashpoint.
7. High-risk and transparency: emotion recognition systems are Annex III high-risk and carry Art. 50(3) transparency duties. The Digital Omnibus delays Annex III to 2 Dec 2027. Art. 50 stays on 2 Aug 2026 `[searched]`.
8. GDPR: emotion or mental-state inference is likely special-category data when it reveals or is used to deduce health (CJEU inference case law). Keystroke dynamics are Art. 9 biometric only if used to uniquely identify a person. Armenia's 2015 data-protection law has its own special-category and notification rules.

---

## 1. Appraisal theory

### 1.1 Scherer's Component Process Model (CPM)
- Stimulus Evaluation Checks (SECs) are grouped into four objectives: **relevance** (novelty: suddenness, familiarity, predictability; intrinsic pleasantness; goal relevance), **implication** (causal attribution, outcome probability, discrepancy from expectation, goal conduciveness, urgency), **coping potential** (control, power, adjustment) and **normative significance** (compatibility with internal and external standards). Checks are sequential and recursive. `[searched]`: [Penn State PSYCH 425 chapter](https://psu.pb.unizin.org/psych425/chapter/component-process-model-cpm/), [Coutinho et al. 2018 PLOS ONE](https://opus.bibliothek.uni-augsburg.de/opus4/files/71791/journal.pone.0189367.pdf). Primary sources to cite: Scherer 2001 (chapter) and 2009 CPM paper `[memory]`.
- Empirical caution: the novelty and pleasantness checks were not found in facial EMG in one test (Coutinho et al. 2018), so "well supported" means supported by **self-report of appraisal vs emotion**, not by physiology.

### 1.2 OCC model (Ortony, Clore, Collins 1988)
- Emotions are valenced reactions to three kinds of things: events (desirability against goals), agents (praiseworthiness against standards), objects (appealingness against tastes). 22 emotion types in groups such as prospect-based, well-being, attribution, fortunes-of-others. `[searched]`: [Ortony papers](https://users.cs.northwestern.edu/~ortony/Andrew_Ortony_files/2013%20-%20OCC%20and%20constructionism.pdf), [Cambridge 2nd ed.](https://www.cambridge.org/core/product/identifier/9781108934053/type/book). OCC is a cognitive account of emotion types, not of behaviour.
- Use for us: it justifies three of our `situation` fields: **goal_congruence** (desirability), **agent** (who is responsible), and a **prospect** flag (event already happened vs expected). Its "standards" branch justifies the guilt / shame / pride / reproach event types.

### 1.3 Dimensions best supported for inference from event text
Evidence that these can be read from text, ordered by strength (my judgement, labelled inference):

| Dimension | Our field | Support |
|---|---|---|
| Pleasantness / goal congruence | `goal_congruence` | Smith & Ellsworth 1985 dimensions; annotated in Hofmann 2020 and crowd-enVENT `[searched]` |
| Responsibility / agency (self, other, circumstance) | `agent` | Same; "Circumstances" is Hofmann's name for situational control `[searched]` |
| Control / coping potential | `controllability` | Same; also CPM coping potential |
| Certainty | `certainty` | Same |
| Anticipated effort | (optional) | Same |
| Attention | (optional) | Same |
| Novelty | (optional) | CPM; in crowd-enVENT's 21 variables per summary, not individually verified |
| Norm compatibility | (optional) | CPM normative significance; weakest in text datasets I found |

Inference: start with the first four (agent, goal congruence, controllability, certainty), which match the brief's schema. Add novelty and norm compatibility only if the evaluation shows they separate event types.

### 1.4 NLP work and datasets with appraisal annotations
- **Hofmann, Troiano, Sassenberg, Klinger 2020**, "Appraisal Theories for Emotion Classification in Text", COLING 2020, [ACL Anthology 2020.coling-main.11](https://aclanthology.org/2020.coling-main.11), [arXiv 2003.14155](https://arxiv.org/pdf/2003.14155v6). They re-annotated 1,001 English emotional event descriptions (from the Troiano et al. 2019 ISEAR-derived corpus) with seven dimensions: attention, certainty, effort, pleasantness, responsibility, control, circumstances. High-quality appraisal labels improved discrete-emotion classification. `[searched]`. The summary says the corpus is public; the licence was **not found**.
- **x-enVENT** (Troiano, Oberländer, Klinger, LREC 2022), [arXiv 2203.10909](https://arxiv.org/abs/2203.10909). Event descriptions with experiencer-specific emotion and appraisal annotations. An aggregator lists CC BY-NC-ND 4.0 (`single-source`, not verified). `[searched]`
- **crowd-enVENT** (Troiano, Oberländer, Klinger, "Dimensional Modeling of Emotions in Text with Appraisal Theories: Corpus Creation, Annotation Reliability, and Prediction", Computational Linguistics 2023, DOI 10.1162/coli_a_00461). Event reports in which each author self-reports 21 appraisal variables. The article is CC BY-NC-ND 4.0; the **dataset licence was not found**. Distributed from the Klinger group data page: [IMS Stuttgart appraisal data page](https://www.ims.uni-stuttgart.de/data/appraisalemotion). I could not confirm the corpus size. `[searched]`
- **CovidET-Appraisals** (Zhan, Ong, Li, EMNLP Findings 2023), [arXiv 2310.14389](https://arxiv.org/abs/2310.14389). 241 Reddit posts, 24 appraisal dimensions with rationales. In zero-shot tests ChatGPT matched human ratings better than open models, with a large gap to open-source models. Data: [honglizhan/CovidET-Appraisals-Public](https://github.com/honglizhan/CovidET-Appraisals-Public). Licence not checked. `[searched]`
- I did not find a clearly dated 2024-2026 benchmark of LLM appraisal extraction beyond CovidET-Appraisals. "Not found" after one search; worth a second pass with arxiv access.
- Implication for Engine E (inference): LLM zero-shot appraisal extraction is plausible but the human-agreement ceiling is low (appraisals are subjective). Report inter-annotator agreement next to model scores, and keep `certainty` and `controllability` as soft values with wide uncertainty.

---

## 2. Event-type taxonomy (MY PROPOSAL)

**Status: my proposal, not a published taxonomy.** It is built from (a) the 32 EmpatheticDialogues (ED) emotion labels (Rashkin et al. 2019, 25k conversations, 32 evenly distributed labels `[searched]`), (b) OCC (goals, standards, attitudes) and (c) the Smith-Ellsworth / Hofmann dimensions. The 32 ED labels, from my memory (the search confirmed 8 of them and the count, not the full list): surprised, excited, angry, proud, sad, annoyed, grateful, lonely, afraid, terrified, guilty, impressed, disgusted, hopeful, confident, furious, anxious, anticipating, joyful, nostalgic, disappointed, prepared, jealous, content, devastated, embarrassed, caring, sentimental, trusting, ashamed, apprehensive, faithful. Verify against the ED repo (`facebookresearch/EmpatheticDialogues`) before use. ED licence: I believe CC BY-NC 4.0 `[memory]`, **not confirmed**. Other agents' licence notes should be the reference.

Caveat: ED labels name the **emotion** a speaker was asked to write about, not the event. Mapping emotion to event type is many-to-many (anger can follow an insult or a failed device). Use the mapping only to seed an LLM re-annotation of the ED `situation` text, then hand-validate a sample. Treat headline accuracy against raw ED labels as an upper-noise benchmark, not ground truth.

| id | Definition | ED labels that mostly map here | Appraisal profile |
|---|---|---|---|
| `loss_or_failure` | Something valued is lost, damaged, missed or failed (death, breakup, job, exam, goal) | devastated, disappointed, sad | goal -, other/circumstance, low control |
| `conflict_or_mistreatment` | Another person wrongs, blocks, insults or irritates the person | angry, furious, annoyed, disgusted (partly) | goal -, agent other, medium control |
| `achievement_or_success` | The person (or someone close) reached a goal or is recognised | proud, joyful, impressed (partly), confident (partly) | goal +, agent self, high control |
| `threat_or_danger` | Present or recently experienced physical or serious personal danger | afraid, terrified | goal -, circumstance/other, low control |
| `uncertainty_about_future` | An upcoming event with unknown or doubtful outcome | anxious, apprehensive, anticipating (negative), prepared (partly) | goal -/0, circumstance, uncertain, medium control |
| `positive_anticipation` | An upcoming event expected to be good | excited, hopeful, anticipating (positive) | goal +, circumstance, uncertain, medium control |
| `own_transgression` | The person violated a norm or hurt someone and feels responsible | guilty, ashamed | goal -, agent self, norm violated |
| `social_exposure` | Public mistake or awkwardness, fear of judgement | embarrassed | goal -, agent self/other, norm violated (minor) |
| `isolation_or_absence` | Missing someone, being alone, not belonging | lonely | goal -, circumstance, low control |
| `kindness_or_support_received` | Someone helped, gave a gift, or was loyal or trustworthy | grateful, trusting, faithful, impressed (partly) | goal +, agent other, n/a |
| `caring_for_others` | The person is looking after or worried about someone else | caring | goal 0/-, agent self, medium control |
| `comparison_or_envy` | Someone else has what the person wants | jealous | goal -, agent other, low control |
| `reminiscence` | Recalling a past time or object with emotional weight | nostalgic, sentimental | goal 0/+, circumstance, low control |
| `unexpected_event` | A sudden, surprising event with unclear valence | surprised | goal ?, circumstance, certainty low |
| `routine_or_wellbeing` | No salient event; steady state or contentment | content, confident (partly), prepared (partly) | goal 0/+, circumstance, high control |
| `other_or_unclear` | Cannot be decided from the text | - | - |

That is 15 named types plus `other_or_unclear`. If 13 is preferred, merge `social_exposure` into `own_transgression` and `caring_for_others` into `kindness_or_support_received`. All 32 ED labels are covered at least once.

---

## 3. Need taxonomy

### 3.1 Theory
- **Social support theory** (House 1981; Cutrona & Russell 1990 "optimal matching") distinguishes **emotional**, **informational**, **instrumental** (tangible) and **esteem** (appraisal) support, and argues that support helps most when it matches the stressor's controllability. `[memory]`: I did not retrieve these sources. Verify before citing. Inference for us: the controllability dimension from appraisal is a principled predictor of "support vs solution".
- **ESConv** (Liu et al., ACL 2021, [arXiv 2106.01144](https://arxiv.org/abs/2106.01144); repo [thu-coai/Emotional-Support-Conversation](https://github.com/thu-coai/Emotional-Support-Conversation)) is based on Hill's Helping Skills Theory: three stages (exploration, comforting, action) and eight strategies. `[github-read]` size and strategy counts: 1,300 conversations, 10 problem topics, plus 196 negative samples (`FailedESConv.json`). Strategy counts from the README: Questions 3,801; Self-disclosure 1,713; Affirmation and Reassurance 2,827; Providing Suggestions 2,954; Reflection of feelings 1,436; Information 1,215; Restatement or Paraphrasing 1,089; Other (the README gives conflicting numbers, 3,661 and 3,341). A 2025 analysis found 2,525 of 13,767 supporter turns use more than one strategy ([arXiv 2505.15316](https://arxiv.org/html/2505.15316v1)).
- **ESConv licence (verbatim from the repo README, read):** "Copyright © 2021 CoAI Group, Tsinghua University. All rights reserved." and "Data and codes are for academic research use only." Source: https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/README.md. A Hugging Face PR revision of the dataset card lists `cc-by-nc-4.0` (`searched`, conflicting with the README; treat as **research-only** either way).
- **Important:** ESConv strategies describe what the **supporter** did, not what the **seeker** needed. A supporter's "Providing Suggestions" is evidence a solution was wanted only if the seeker responded well. Seeker feedback scores in ESConv (1-5 satisfaction) exist, but I did not confirm their fields.

### 3.2 Proposed mapping (MY PROPOSAL)

| Our need | Social-support type | ESConv strategies it corresponds to | Labelled data found |
|---|---|---|---|
| `emotional_support` | Emotional (plus esteem) | Affirmation and Reassurance, Reflection of feelings, Restatement | Weak: ESConv strategies (supporter side, research-only). ED responses likewise. |
| `solution` | Instrumental / informational-advice | Providing Suggestions | Weak: ESConv |
| `information` | Informational | Information | Weak: ESConv |
| `distraction` | none directly (coping literature: distraction as emotion regulation) | none | **Not found** |
| `rest` | none directly | none | **Not found** |
| `celebration` | Esteem / shared positive affect ("capitalization", Gable) `[memory]` | none (ESConv is about problems) | **Not found**; positive-event ED labels could seed it |

Also worth checking: a taxonomy of empathetic response intents for ED (arXiv 2012.04080, Welivita & Pu) appeared in search but I did not read it.
Recommendation (inference): since no labelled "seeker need" data exists, derive `need` with the LLM engine E plus a hand-labelled set of about 200 ED / personal-study messages, and treat `distraction`, `rest` and `celebration` as unvalidated until Phase 5 self-reports exist. Add a "needs" field to the Phase 5 experience-sampling form (one tap: "what do I want right now?").

---

## 4. Open alternatives to LIWC

| Resource | What | Licence | Notes |
|---|---|---|---|
| **Empath** (Fast, Chen, Bernstein, CHI 2016) | about 200 built-in word-list categories (the client example showed 194) generated from word embeddings, correlates about 0.90 with LIWC on shared categories `[searched]` | **MIT** `[github-read]`. Verbatim: "Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software..." Copyright (c) 2016 Ethan Fast. https://raw.githubusercontent.com/Ejhfast/empath-client/master/LICENSE.txt | This is the licence of the **client** repo. The core `empath` PyPI package listing shows only "Other" `[searched]`; confirm that the lexicon file (`categories.tsv`) in the PyPI package carries the same terms before shipping. Paper: [arXiv 1602.06979](https://arxiv.org/abs/1602.06979). Not updated since release. |
| **SEANCE** (Crossley et al. 2017, Behav Res Methods) | free tool bundling several open sentiment dictionaries `[searched]`: [doi 10.3758/s13428-016-0743-z](https://dx.doi.org/10.3758/s13428-016-0743-z) | Mixed (inherits component dictionaries) | Not checked per-dictionary. Probably includes research-only components such as GALC/GI/Lasswell. Treat as research-only until audited. |
| **Absolutist words** (Al-Mosaiwi & Johnstone 2018, Clinical Psychological Science) | 19-ish word dictionary (the search could not confirm the count); forums for anxiety, depression and suicidal ideation scored higher than controls; suicidal-ideation forums highest. Replicated in French, Spanish, German and Russian in a 2019 thesis `[searched]` | Paper: [Reading CentAUR 73630](https://centaur.reading.ac.uk/73630). Word list is in the supplement. **Not retrieved.** | Examples seen: always, nothing, totally, entire, everyone, completely. Authors caution that absolutism is a marker, not a cause. Build our own list from the paper's supplement (a word list is a fact, but check the supplement's terms). |
| **First-person singular pronouns** | Edwards & Holtzman 2017 meta-analysis: depression vs I-talk r = 0.13, 95% CI [0.10, 0.16], 21 correlations, N = 3,758. Tackman et al. 2019 (N = 4,754, 6 labs): the link holds across pronoun types except possessive and may reflect general negative emotionality, not depression specifically. Tølbøll 2019 meta-analysis: Cohen's d 0.44 for I-talk, 0.72 for negative emotion words, -0.38 for positive emotion words `[searched]` | Findings, not a lexicon; pronoun lists are trivial (I, me, my, mine, myself) | [Tackman et al. JPSP 2019](https://digitalcommons.georgiasouthern.edu/psych-facpubs/113), [Tølbøll 2019](https://tidsskrift.dk/lwo/article/view/117798). Inference: the effect is **small** (r about 0.1); use as a weak, low-weight feature and never as a standalone risk indicator. |
| **Hedging, certainty, tentative words** | No permissively licensed ready-made lexicon found. | - | "Not found" after one search. Build a small list from linguistics sources (hedges: maybe, perhaps, might, kind of, sort of, I guess, seems; boosters: definitely, certainly, obviously). Record as "own list, derived from public linguistic literature". |
| **NRC lexicons** (VAD, EmoLex) | Already covered by other researchers. NRC licences are non-commercial by default `[memory]`. | - | Out of my scope. |

Recommendation (inference): ship **Empath (after confirming the lexicon licence) plus our own small hand-built lists** (first-person singular, absolutist, hedge, booster, question words, past/future tense via spaCy). This avoids any LIWC dependency. For Russian, the absolutist finding was replicated in a Russian-language thesis; build RU/HY lists by hand later.

---

## 5. Crisis safety

What I found `[searched]` (evidence is thinner than I would like; there is no deployed-system false-negative data):
- **Classifier vs keyword.** Broadbent et al. 2023 (Frontiers in Psychiatry, [10.3389/fpsyt.2023.1110527](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2023.1110527/text)) used SafeUT crisis-line text. A neural model beat a TF-IDF baseline on the false-negative rate, and 75% of the neural model's false-negative encounters still contained some discussion of suicidality. A JMIR preprint (Williamson et al., not peer-reviewed) reports a hybrid: an SVM with symptom-vocabulary features at 91.8% accuracy, plus a keyword-informed layer that handles negation, temporal displacement and implicit risk, beating a base LLM (58.7%) and a fine-tuned LLM (74.5%) ([preprint](https://preprints.jmir.org/preprint/98925)). `single-source`.
- **General LLM behaviour.** A RAND study in Psychiatric Services (Aug 2025) found chatbots inconsistent on suicide-related queries, with ChatGPT pointing to an outdated hotline number rather than 988 ([PsyPost summary](https://www.psypost.org/ai-chatbots-give-inconsistent-responses-to-suicide-related-questions-study-finds/)). A 2026 independent clinical evaluation (Sentio / IASEAI) found a named crisis line in only 41% of responses ([Sentio](https://sentio.org/ai-research/llm-responses-to-suicide-risk-iaseai-study)). Inference: resource display must be **deterministic code**, not left to LLM wording.
- **Off-the-shelf guard models.** Llama Guard 3 has category S11 "Self-Harm" (suicide, self-injury, disordered eating) ([model card](https://llama.com/docs/model-cards-and-prompt-formats/llama-guard-3)); its licence is the Llama Community Licence (**not commercial-OK without reading it**; one third-party listing, `single-source`). OpenAI's moderation endpoint has `self-harm`, `self-harm/intent`, `self-harm/instructions` categories ([API reference](https://developers.openai.com/api/docs/api-reference/moderations)), but it sends user text to a third party, which conflicts with brief §1.8 unless you approve it. I found no permissively licensed, validated suicide-risk model with Armenian or Russian coverage. Not found.
- **Datasets** for training (Reddit SuicideWatch-style, C-SSRS Reddit) are mostly request-only or research-only `[memory]`; not verified. Do not train on them for a commercial product without a licence check.

Proposed design (inference, labelled as mine):
1. **Hard rule in the core:** `safety_flag` in the Emotional State Object, computed before and independent of fusion, so no adapter can suppress it. Three levels: `none`, `possible`, `acute`.
2. **Layer 1 (always on, local):** multilingual (en, ru, hy, plus transliterated) regex and phrase lists for explicit ideation, plan, means, and self-harm, with negation and quotation handling. Tuned for **recall**.
3. **Layer 2 (local):** a small classifier (fine-tune on a licensed or self-built set), or a zero-shot NLI/embedding check, to catch implicit ideation ("everyone would be better off").
4. **Layer 3 (optional, approved LLM call):** a judge for borderline cases only.
5. **Trade-off:** false negatives are the costly error, so set thresholds for high recall and accept false positives. Mitigate false-positive harm by showing a gentle, non-alarming resource card, not a lockout, and never telling the user they are in crisis. Log flag rates and review sample false positives.
6. **Product must show** (best practice, `[memory]` plus the evidence above): a visible human-help message with the user's local helpline and emergency number, no "solution" advice or recommendations in that turn, no method details, a clear statement that the system is not a clinician, and a way to continue talking to a human. Keep the list of numbers in a curated, dated table with an owner for yearly review (outdated numbers were a real failure in the RAND study).
7. **Directory choice:** [Find a Helpline](https://findahelpline.com/) is run by ThroughLine. Its terms (read via search summary) prohibit scraping, extracting content with bots or AI tools, and training AI on its content, and say to contact them for commercial use. ThroughLine has a commercial API with about 1,300 verified helplines worldwide and rate limits ([developer.throughlinecare.com](https://developer.throughlinecare.com)). OpenAI partners with ThroughLine for the same purpose ([OpenAI help article](https://help.openai.com/ta-in/articles/12677603-crisis-helpline-support-in-chatgpt)). For an MVP, **hand-curate** the three countries we care about and link to the directory page for the rest.

Country notes (numbers must be verified by the owner; the sources disagree):
- **Armenia:** Find A Helpline lists a free "Emotional Support Hotline" for adults, started in early 2025 per a GlobalGiving report ([Find a Helpline Armenia](https://findahelpline.com/countries/am), [organisation page](https://findahelpline.com/organizations/emotional-support-hotline)); the number was **not in my search results**. An older directory lists (2) 538194 / 538197 and may be out of date ([progress.guide](https://progress.guide/atlas/asia/armenia/)). TherapyRoute says emergency numbers are 112, 911 or 102. A WHO Europe story on an Armenian mental-health hotline exists ([WHO](https://www.who.int/europe/fr/news-room/feature-stories/item/meet-the-female-psychologists-taking-calls-on-an-armenian-mental-health-hotline)).
- **Russia:** the national child and teen line is 8-800-2000-122 (also short code 124 from mobiles, since Dec 2024, per one source). Adult options listed: EMERCOM psychological line +7 (495) 989-50-50, Moscow service 8-800-250-05-22. Sources disagree on formats; verify. Emergency: 112. ([TherapyRoute Russia](https://www.therapyroute.com/article/suicide-hotlines-and-crisis-lines-in-russia), [Find a Helpline RU](https://findahelpline.com/countries/ru/topics/suicidal-thoughts)).
- **US:** 988 Suicide and Crisis Lifeline `[memory]`.

---

## 6. Legal status (as of 2026-11-18) - NOT LEGAL ADVICE

**Source limits:** eur-lex.europa.eu and artificialintelligenceact.eu were unreachable, so the article text below is `[memory]` cross-checked with law-firm summaries `[searched]`. Re-verify against the Official Journal text of Regulation (EU) 2024/1689 ([eur-lex](https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689)).

### 6.1 Prohibition (Art. 5(1)(f))
- Text (from memory, paraphrase-safe): prohibits "the placing on the market, the putting into service for this specific purpose, or the use of AI systems to infer emotions of a natural person in the areas of workplace and education institutions, except where the use of the AI system is intended to be put in place or into the market for medical or safety reasons."
- **In force:** prohibitions apply from **2 February 2025** (Art. 113(a)); penalty provisions from 2 August 2025 `[memory]`. Commission guidelines on prohibited practices were published in early February 2025 (the CMS/DLA/Freshfields pieces describe them; one result gave a wrong year of 2024). The Commission stresses they are non-binding and the CJEU decides.
- **Scope per the guidelines** `[searched]` ([FPF analysis](https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/), [CMS](https://cms.law/en/gbr/legal-updates/eu-commission-issues-guidelines-on-prohibited-ai-practices-part-ii), [Freshfields](https://www.freshfields.com/en/our-thinking/blogs/technology-quotient/eu-ai-act-unpacked-25-european-commission-releases-critical-ai-act-implementati-102kfql)): covers individuals, not groups; also covers inference of intentions; "education institutions" includes public and private schools, universities, vocational and continuing education; examples: call-centre anger tracking by webcam or voice, inferring student interest or attention, emotion recognition during admission tests. Exceptions: medical and safety (pilot or driver fatigue). Detecting **burnout or depression** at work or school is *not* covered by the medical exception. Member States may impose stricter worker protections.
- **Not covered:** emotion recognition outside work and education (for example commercial settings) is not banned, though it may be high-risk. Systems inferring emotion from **written text** not based on biometric data are described as outside the definition ([CMS](https://cms.law/en/gbr/legal-updates/eu-commission-issues-guidelines-on-prohibited-ai-practices-part-ii), [HCA Mag](https://hcamag.com/au/specialisation/employment-law/eus-new-ai-act-restricts-emotion-recognition-systems-in-workplaces/524294)). Lewis Silkin says the written-text exception should be read narrowly ([Lewis Silkin](https://www.lewissilkin.com/insights/2025/02/17/understanding-the-eu-ai-acts-prohibited-practices-key-workplace-and-advertising-102k011)).
- **Keystrokes:** the guidelines treat keystroke patterns (typing style), body postures and movements as biometric inputs, so emotion inference from them **is** covered ([HCA Mag](https://hcamag.com/au/specialisation/employment-law/eus-new-ai-act-restricts-emotion-recognition-systems-in-workplaces/524294), `single-source` plus the education-focused summaries above).

### 6.2 Definition and the other obligations
- Art. 3(39) (from memory): "emotion recognition system" means an AI system for the purpose of identifying or inferring emotions or intentions of natural persons **on the basis of their biometric data**. Recital 18 says it does not cover physical states like pain or fatigue, nor the mere detection of readily apparent expressions (a smile) unless used to infer emotion `[memory]`.
- **High-risk:** Annex III point 1(c) lists AI systems intended for emotion recognition as high-risk (where permitted under EU or national law) `[memory]`. The Digital Omnibus on AI: provisional deal 7 May 2026, Parliament first-reading approval 16 June, Council approval 29 June, signed 8 July 2026 (per the Legislative Observatory as relayed by [Cuatrecasas](https://www.cuatrecasas.com/en/global/intellectual-property/art/council-eu-approves-digital-omnibus-ai)). It moves Annex III high-risk obligations to **2 December 2027** and Annex I product-embedded systems to 2 August 2028 ([White & Case](https://www.whitecase.com/insight-alert/eu-agrees-digital-omnibus-deal-simplify-ai-rules), [Gibson Dunn](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/)). One blog gives Official Journal publication on 24 July and entry into force 27 July 2026 as Regulation (EU) 2026/1744 ([Spicy Advisory](https://spicyadvisory.com/blog/digital-omnibus-ai-act-2026-what-changed)); `single-source`, **confirm on eur-lex**. The Omnibus adds two new prohibitions (non-consensual intimate content and CSAM) and reportedly leaves the emotion-recognition ban unchanged (not confirmed from the text).
- **Transparency:** Art. 50(3): deployers of an emotion recognition system must inform the exposed natural persons and process personal data per GDPR `[memory]`. Application date 2 August 2026, reported unchanged by the Omnibus (the same Cuatrecasas/Spicy Advisory results). Art. 50(2) watermarking moves to 2 Dec 2026.
- **Extraterritoriality:** the Act applies to providers placing systems on the EU market and to non-EU systems whose output is used in the EU (Art. 2) `[memory]`. Armenia-based development does not exempt EU-facing products.

### 6.3 GDPR
- **Special-category data (Art. 9).** The CJEU has held that data allowing sensitive characteristics to be inferred through "comparison or deduction" is special-category data ([Matheson summary of the 2022 ruling](https://publications.matheson.com/data-protection-technology-and-cyber-security-bulletin-october-2022/cjeu-expands-scope-of-special-category-data), case C-184/20 `[memory]`). EDPB analogy from video guidelines: footage is special-category data if processed to deduce health ([Baker McKenzie](https://connectontech.bakermckenzie.com/sensitive-data-under-the-cjeus-spotlight-practical-implications/)). **Inference (mine, not sourced):** a system that infers mood, distress or risk of self-harm is likely to produce health-related (mental-health) inferences and should be treated as Art. 9 processing. Plain valence/arousal on a single message is a grey area; the safety flag and the "pattern over time" outputs are not.
- **Keystroke dynamics:** Art. 9(1) lists "biometric data for the purpose of uniquely identifying a natural person". Typing rhythm used only to infer state, not identity, may not be Art. 9 biometric, though it is still personal data, and the AI Act definition of biometric data is broader (it does not require identification). `[searched]`: not covered by any EDPB text I found.
- **What follows:** explicit consent (Art. 9(2)(a)) or another Art. 9 ground, a DPIA (Art. 35), data minimisation (the brief already stores only key categories), right to erase (matches `DELETE /user/{id}/data`), parental consent for minors (Art. 8). EDPB guidelines on emotion inference specifically: not found.

### 6.4 Armenia
- Law on Protection of Personal Data (2015) remains the core framework; no replacement found. It defines biometric data and special-category data (race, ethnicity, political views, religion, trade-union membership, health, sex life per the summaries) and, per a Feb 2024 notice of the Personal Data Protection Agency, processors must notify the Agency **before** processing biometric or special-category data (Art. 23) ([Mondaq](https://www.mondaq.com/data-protection/1635302/personal-data-protection-in-armenia), [DLA Piper](https://www.dlapiperdataprotection.com/?c=AM&t=law), [GRATA](https://gratanet.com/publications/personal-data-protection-in-armenia-key-legal-overview), [English text on foi.am](https://foi.am/wp-content/uploads/2023/09/Personaldataprotectionlaw_ENG.pdf)). One summary says special-category processing without consent is prohibited except where law directly provides. A cybersecurity bill was pending in 2024; status unknown. Check arlis.am for the consolidated text. `[searched]`
- Russia: not researched (outside the assigned scope for this task). Russian data-localisation rules would matter for Russian users.

### 6.5 The owner's tutoring use case (flag)
Inference, stated as such:
1. **Text-only engines (A, B, C, E, F)**: per the Commission's guidelines as summarised, text-based inference is outside the AI Act emotion-recognition definition. Lewis Silkin says to read that narrowly. The Act's definition is tied to biometric data, so a pure-text tutor is **probably** not a prohibited or Annex III system. It is still GDPR personal data and probably special-category when it infers wellbeing.
2. **Keystroke engine D**: treated as biometric inference. If a tutoring product is deployed **in or for an educational institution** (school, university, vocational or continuing-education provider), using D to infer frustration or engagement is **within the Art. 5(1)(f) prohibition**, with fines up to the top penalty tier (EUR 35m or 7% of worldwide turnover for prohibited practices `[memory]`). The medical/safety exception does not rescue "student wellbeing" or "burnout" detection. The B2C case (an individual learner using an app on their own, no institution involved) is unclear: "in the areas of education institutions" may not reach it. The guidelines' online-language-course example is ambiguous (`[searched]`).
3. **Safest design (my recommendation):** make D a build-time and deploy-time switch, **off by default**, hard-disabled for any tenant flagged as education or workplace; use text-only signals in tutoring; obtain legal review before a pilot with a school.
4. The `safety_flag` for self-harm is not the same as emotion recognition for pedagogy, but flagging distress in a school-deployed text tutor may still be read as health or burnout detection; get counsel.

---

## Machine-readable

```json
{
  "event_types": [
    {"id": "loss_or_failure", "definition": "Something valued is lost, damaged, missed or failed (death, breakup, job, exam, goal).", "ed_labels": ["devastated", "disappointed", "sad"], "appraisal_profile": {"agency": "circumstance_or_other", "goal_congruence": "-", "controllability": "low"}},
    {"id": "conflict_or_mistreatment", "definition": "Another person wrongs, blocks, insults or irritates the person.", "ed_labels": ["angry", "furious", "annoyed", "disgusted"], "appraisal_profile": {"agency": "other", "goal_congruence": "-", "controllability": "medium"}},
    {"id": "achievement_or_success", "definition": "The person or someone close reached a goal or was recognised.", "ed_labels": ["proud", "joyful", "impressed", "confident"], "appraisal_profile": {"agency": "self", "goal_congruence": "+", "controllability": "high"}},
    {"id": "threat_or_danger", "definition": "Present or recent physical or serious personal danger.", "ed_labels": ["afraid", "terrified"], "appraisal_profile": {"agency": "circumstance_or_other", "goal_congruence": "-", "controllability": "low"}},
    {"id": "uncertainty_about_future", "definition": "An upcoming event with unknown or doubtful outcome.", "ed_labels": ["anxious", "apprehensive", "anticipating", "prepared"], "appraisal_profile": {"agency": "circumstance", "goal_congruence": "-", "controllability": "medium"}},
    {"id": "positive_anticipation", "definition": "An upcoming event expected to be good.", "ed_labels": ["excited", "hopeful", "anticipating"], "appraisal_profile": {"agency": "circumstance", "goal_congruence": "+", "controllability": "medium"}},
    {"id": "own_transgression", "definition": "The person violated a norm or hurt someone and feels responsible.", "ed_labels": ["guilty", "ashamed"], "appraisal_profile": {"agency": "self", "goal_congruence": "-", "controllability": "medium"}},
    {"id": "social_exposure", "definition": "Public mistake, awkwardness or fear of judgement.", "ed_labels": ["embarrassed"], "appraisal_profile": {"agency": "self_or_other", "goal_congruence": "-", "controllability": "low"}},
    {"id": "isolation_or_absence", "definition": "Missing someone, being alone, not belonging.", "ed_labels": ["lonely"], "appraisal_profile": {"agency": "circumstance", "goal_congruence": "-", "controllability": "low"}},
    {"id": "kindness_or_support_received", "definition": "Someone helped, gave a gift, or proved loyal and trustworthy.", "ed_labels": ["grateful", "trusting", "faithful", "impressed"], "appraisal_profile": {"agency": "other", "goal_congruence": "+", "controllability": "n/a"}},
    {"id": "caring_for_others", "definition": "The person is looking after or worried about someone else.", "ed_labels": ["caring"], "appraisal_profile": {"agency": "self", "goal_congruence": "0", "controllability": "medium"}},
    {"id": "comparison_or_envy", "definition": "Someone else has what the person wants.", "ed_labels": ["jealous"], "appraisal_profile": {"agency": "other", "goal_congruence": "-", "controllability": "low"}},
    {"id": "reminiscence", "definition": "Recalling a past time, person or object with emotional weight.", "ed_labels": ["nostalgic", "sentimental"], "appraisal_profile": {"agency": "circumstance", "goal_congruence": "0", "controllability": "low"}},
    {"id": "unexpected_event", "definition": "A sudden, surprising event with unclear valence.", "ed_labels": ["surprised"], "appraisal_profile": {"agency": "circumstance", "goal_congruence": "0", "controllability": "low"}},
    {"id": "routine_or_wellbeing", "definition": "No salient event; steady state or contentment.", "ed_labels": ["content", "confident", "prepared"], "appraisal_profile": {"agency": "circumstance", "goal_congruence": "+", "controllability": "high"}},
    {"id": "other_or_unclear", "definition": "Cannot be decided from the text.", "ed_labels": [], "appraisal_profile": {"agency": "unknown", "goal_congruence": "0", "controllability": "unknown"}}
  ],
  "needs": [
    {"id": "emotional_support", "definition": "The person wants to be heard, validated or comforted, not advised.", "theory_link": "Emotional and esteem support (House 1981; Cutrona & Russell 1990, from memory); ESConv strategies Affirmation and Reassurance, Reflection of feelings, Restatement", "data_available": "Weak: ESConv supporter-strategy labels (research-only, academic use). No seeker-need labels."},
    {"id": "solution", "definition": "The person wants concrete advice or a plan to fix the problem.", "theory_link": "Instrumental support; matches high-controllability stressors (optimal matching); ESConv strategy Providing Suggestions", "data_available": "Weak: ESConv Providing Suggestions (research-only). No seeker-need labels."},
    {"id": "information", "definition": "The person wants facts or explanation.", "theory_link": "Informational support; ESConv strategy Information", "data_available": "Weak: ESConv Information (research-only)."},
    {"id": "distraction", "definition": "The person wants a change of scene or topic; engaging with the problem is not wanted.", "theory_link": "Distraction as an emotion-regulation strategy (no source retrieved)", "data_available": "Not found. Collect in Phase 5 self-reports."},
    {"id": "rest", "definition": "The person is depleted and wants calm, low stimulation or sleep.", "theory_link": "None retrieved; links to low arousal and low dominance in the VAD output", "data_available": "Not found. Collect in Phase 5 self-reports."},
    {"id": "celebration", "definition": "The person wants to share a positive event and have it acknowledged.", "theory_link": "Esteem support / capitalization on positive events (Gable, from memory)", "data_available": "Not found. Positive ED situations (proud, joyful, excited) could seed it."}
  ]
}
```

---

## Open questions / things the user must do

1. **Allow hosts for a verification pass:** eur-lex.europa.eu, arxiv.org, aclanthology.org, huggingface.co, findahelpline.com, artificialintelligenceact.eu, fpf.org. Then re-run the legal and licence checks. Everything non-GitHub here is search-summary level.
2. **Get legal advice** on (a) keystroke engine D in any education or workplace deployment, (b) whether a consumer tutoring app outside a school counts as "in the areas of education institutions", (c) GDPR Art. 9 treatment of mood and risk inference, (d) Armenian Agency notification before processing biometric or special-category data. I am not a lawyer.
3. **Decide:** keystroke engine D off by default and hard-disabled for education and workplace tenants? (My recommendation: yes.)
4. **Decide:** helpline data approach for the MVP: hand-curated table for en/ru/hy countries (my recommendation) versus a ThroughLine API agreement. **You must verify the Armenian and Russian numbers by calling or checking official sites**; the sources disagree and some are outdated.
5. **Fetch:** the Al-Mosaiwi & Johnstone 2018 supplementary word list (via [CentAUR](https://centaur.reading.ac.uk/73630) or the journal), and check its terms. Confirm the Empath lexicon licence by inspecting the installed `empath` package's data files.
6. **Confirm:** the ED licence (my memory says CC BY-NC 4.0, unverified) and the crowd-enVENT / Hofmann dataset licences on the Klinger group page. If research-only, they cannot ship in a sold product.
7. **Approve:** adding a "what do I want right now?" tap (the six needs) to the Phase 5 self-report form, since no labelled need data exists.
8. **Decide:** whether safety Layer 3 may use an external LLM call (only the Claude call approved in §3.E exists today).

## Sources

1. Hofmann et al. 2020, Appraisal Theories for Emotion Classification in Text - https://aclanthology.org/2020.coling-main.11 - seven appraisal dimensions, 1,001 events; appraisal helps classification.
2. Hofmann et al. arXiv - https://arxiv.org/pdf/2003.14155v6 - same, preprint.
3. x-enVENT - https://arxiv.org/abs/2203.10909 - experiencer-specific emotion and appraisal corpus.
4. crowd-enVENT / Troiano et al. 2023 data page - https://www.ims.uni-stuttgart.de/data/appraisalemotion - 21 appraisal variables, dataset download.
5. TIB record for crowd-enVENT - https://service.tib.eu/ldmservice/dataset/bibtex/crowd-envent - dataset entry (2025).
6. CovidET-Appraisals - https://arxiv.org/abs/2310.14389 - 241 posts, 24 dimensions, LLM evaluation.
7. Penn State CPM chapter - https://psu.pb.unizin.org/psych425/chapter/component-process-model-cpm/ - SEC groups.
8. Coutinho et al. 2018 - https://opus.bibliothek.uni-augsburg.de/opus4/files/71791/journal.pone.0189367.pdf - tests of novelty and pleasantness checks.
9. Ortony OCC papers - https://users.cs.northwestern.edu/~ortony/Andrew_Ortony_files/2013%20-%20OCC%20and%20constructionism.pdf - OCC structure.
10. ESConv repo README (read) - https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/README.md - size, strategies, academic-use licence.
11. ESConv paper - https://arxiv.org/abs/2106.01144 - Helping Skills Theory basis.
12. Multi-strategy turns in ESConv - https://arxiv.org/html/2505.15316v1 - 2,525 of 13,767 turns multi-strategy.
13. Empath LICENSE (read) - https://raw.githubusercontent.com/Ejhfast/empath-client/master/LICENSE.txt - MIT text.
14. Empath paper - https://arxiv.org/abs/1602.06979 - categories, LIWC correlation.
15. Al-Mosaiwi & Johnstone 2018 - https://centaur.reading.ac.uk/73630 - absolutist words.
16. Tackman et al. 2019 - https://digitalcommons.georgiasouthern.edu/psych-facpubs/113 - I-talk and negative emotionality.
17. Tølbøll 2019 - https://tidsskrift.dk/lwo/article/view/117798 - effect sizes d = 0.44 / 0.72 / -0.38.
18. SEANCE - https://dx.doi.org/10.3758/s13428-016-0743-z - open sentiment tool bundle.
19. Broadbent et al. 2023 - https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2023.1110527/text - neural vs TF-IDF on crisis text.
20. JMIR preprint - https://preprints.jmir.org/preprint/98925 - hybrid keyword and probabilistic ensemble (not peer-reviewed).
21. RAND study summary - https://www.psypost.org/ai-chatbots-give-inconsistent-responses-to-suicide-related-questions-study-finds/ - outdated hotline, inconsistency.
22. Sentio / IASEAI 2026 - https://sentio.org/ai-research/llm-responses-to-suicide-risk-iaseai-study - named crisis line in 41% of responses.
23. Llama Guard 3 card - https://llama.com/docs/model-cards-and-prompt-formats/llama-guard-3 - S11 self-harm.
24. OpenAI moderation - https://developers.openai.com/api/docs/api-reference/moderations - self-harm categories.
25. ThroughLine developer docs - https://developer.throughlinecare.com - commercial helpline API.
26. Find a Helpline terms - https://findahelpline.com/terms - no scraping, no AI training.
27. OpenAI help on helpline support - https://help.openai.com/ta-in/articles/12677603-crisis-helpline-support-in-chatgpt - ThroughLine partnership precedent.
28. Find a Helpline Armenia - https://findahelpline.com/countries/am - Armenian hotlines list.
29. TherapyRoute Russia - https://www.therapyroute.com/article/suicide-hotlines-and-crisis-lines-in-russia - Russian lines.
30. FPF analysis of Art. 5(1)(f) - https://fpf.org/blog/red-lines-under-eu-ai-act-unpacking-the-prohibition-of-emotion-recognition-in-the-workplace-and-education-institutions/ - scope, exceptions.
31. CMS on Commission guidelines - https://cms.law/en/gbr/legal-updates/eu-commission-issues-guidelines-on-prohibited-ai-practices-part-ii - written text outside definition.
32. HCA Mag - https://hcamag.com/au/specialisation/employment-law/eus-new-ai-act-restricts-emotion-recognition-systems-in-workplaces/524294 - keystrokes treated as biometric.
33. Lewis Silkin - https://www.lewissilkin.com/insights/2025/02/17/understanding-the-eu-ai-acts-prohibited-practices-key-workplace-and-advertising-102k011 - narrow reading of text exception; guidelines non-binding.
34. Freshfields - https://www.freshfields.com/en/our-thinking/blogs/technology-quotient/eu-ai-act-unpacked-25-european-commission-releases-critical-ai-act-implementati-102kfql - guidelines summary.
35. Cuatrecasas on Omnibus approval - https://www.cuatrecasas.com/en/global/intellectual-property/art/council-eu-approves-digital-omnibus-ai - Council approval 29 June 2026.
36. White & Case on Omnibus deal - https://www.whitecase.com/insight-alert/eu-agrees-digital-omnibus-deal-simplify-ai-rules - Annex III to Dec 2027.
37. Spicy Advisory - https://spicyadvisory.com/blog/digital-omnibus-ai-act-2026-what-changed - OJ publication and Regulation number (single-source).
38. Matheson on CJEU special-category ruling - https://publications.matheson.com/data-protection-technology-and-cyber-security-bulletin-october-2022/cjeu-expands-scope-of-special-category-data - inference counts.
39. Baker McKenzie - https://connectontech.bakermckenzie.com/sensitive-data-under-the-cjeus-spotlight-practical-implications/ - sensitive data implications.
40. Armenia data protection: Mondaq https://www.mondaq.com/data-protection/1635302/personal-data-protection-in-armenia ; DLA Piper https://www.dlapiperdataprotection.com/?c=AM&t=law ; English law text https://foi.am/wp-content/uploads/2023/09/Personaldataprotectionlaw_ENG.pdf - 2015 law, special categories, Agency notification.
