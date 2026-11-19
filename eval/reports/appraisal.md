# Engine E (appraisal) report

ED test split, one row per conversation (2542 rows; zero-shot/combined on a stratified sample of 601). Labels: ED_TO_EVENT (lossy, 12 of 16 event types have ED support). 95% bootstrap CIs.

| System | n | Accuracy | Macro-F1 |
|---|---|---|---|
| Majority baseline | 601 | 0.145 [0.116, 0.173] | 0.021 [0.017, 0.025] |
| Zero-shot NLI | 601 | 0.291 [0.256, 0.328] | 0.249 [0.214, 0.279] |
| Fallback LR (same sample) | 601 | 0.616 [0.579, 0.652] | 0.620 [0.581, 0.655] |
| Combined 0.5/0.5 | 601 | 0.617 [0.582, 0.657] | 0.622 [0.584, 0.658] |
| Fallback LR (all test) | 2542 | 0.616 [0.597, 0.635] | 0.613 [0.594, 0.632] |

Fallback = frozen MiniLM embeddings + balanced LogisticRegression (replaces the DistilRoBERTa fine-tune for now: CPU cost). Zero-shot restricted to the event types present in ED labels for scoring (12); the engine itself ranks all 16.

Latency (CPU): full engine 217 ms/message (20 ED prompts); zero-shot event only 1212 ms.

Challenge sarcasm flag agreement: 48/48 (clauses.sarcasm_cues >= 0.5).

Verdict:
- Zero-shot NLI alone is weak on ED-derived labels (acc 0.29, macro-F1 0.25; ED labels name feelings, lossy mapping), but is the only path for non-English text.
- The embedding fallback carries English event typing (0.62 acc); the 0.5/0.5 blend is statistically tied with it (CIs overlap), so zero-shot adds no measurable accuracy on ED, only coverage of unsupported types/languages.
- Latency is noisy (CPU shared with parallel agents; 0.2-4 s/message observed), far above the 300 ms budget: engine E must stay async/optional. Sarcasm 48/48 only measures cue-heuristic agreement with the challenge set it was tuned on.
