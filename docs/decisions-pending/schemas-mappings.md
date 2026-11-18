# Pending decisions: schemas-mappings

For the orchestrator to merge into `docs/DECISIONS.md`. Each: decision, alternatives, why.

## Coarse aggregation by noisy-OR
- **Decision:** `to_ekman` / `to_plutchik` combine fine-label probabilities by noisy-OR,
  `1 - prod(1 - p_i)`. Missing labels count as 0. Unknown labels raise `UnknownLabelError`
  (a `ValueError`); out-of-range or NaN probabilities raise `ValueError`; booleans raise `TypeError`.
- **Alternatives:** sum and clip (not bounded in a principled way); max (ignores corroborating
  labels); mean (lowers the coarse score when unrelated members are 0).
- **Why:** GoEmotions is multi-label, so the fine scores are independent presence scores.
  Noisy-OR stays in [0, 1], is monotone in every input and is never below its strongest member
  (all three are Hypothesis-tested). Downside: it assumes independence and over-counts correlated
  labels (anger + annoyance); fusion calibration (§4.6) must correct for that.

## `neutral` is not mapped to any coarse category
- **Decision:** `neutral` contributes to no Ekman or Plutchik category, and the coarse dicts have
  no `neutral` key. Neutrality stays visible in the fine-grained `emotions` list.
- **Alternatives:** a seventh/ninth "neutral" coarse key (GoEmotions' Ekman-level experiments do
  use neutral as an extra class).
- **Why:** the official `ekman_mapping.json` has no neutral entry, and Plutchik's eight primaries
  have no neutral either. Adding one is easy later if the coarse view needs it.

## Plutchik mapping (our proposal)
- **Decision:** each GoEmotions label maps to one or two of Plutchik's primaries. Two when
  Plutchik names it as a dyad (love = joy + trust, optimism = anticipation + joy, remorse =
  sadness + disgust, curiosity = trust + surprise, disappointment = surprise + sadness, anxiety
  for nervousness = anticipation + fear). Twelve labels are marked approximate
  (`PLUTCHIK_APPROXIMATE`, and `# approximate` in code), each with a rationale in
  `PLUTCHIK_RATIONALE`. Two choices depart from Plutchik's dyad names on purpose:
  - `pride` maps to joy only, not anger + joy (GoEmotions' pride has no anger);
  - `disapproval` maps to disgust + anger (contempt), not surprise + sadness (Plutchik's
    "disapproval" means something else).
- **Alternatives:** one primary per label, copying the Ekman grouping where possible (loses
  trust and anticipation almost completely).
- **Why:** it uses Plutchik's own dyad structure where it exists and keeps trust and
  anticipation populated. **Needs validation**, e.g. against NRC EmoLex's Plutchik columns on
  GoEmotions text, before anyone relies on the coarse Plutchik view.

## Upstream pins without a commit SHA
- **Decision:** `GOEMOTIONS_LABELS` and `EKMAN_MAP` are pinned by URL, fetch date and sha256 of
  the fetched file (`emotions.txt` 45c3ef86…5a89, `ekman_mapping.json` d6b1fea3…a328).
- **Alternatives:** pin to a google-research commit SHA.
- **Why:** the GitHub API is not enabled for that repository in this session, so no commit SHA
  could be read. The sha256 values let anyone re-check the files.

## VAD scale ranges
- **Decision:** linear maps to [-1, 1], with `(2v - (low + high)) / (high - low)` and its inverse
  (exact at the ends and the midpoint, identity for -1..1). Ranges: Warriner 1..9, SAM 1..9,
  **EmoBank 1..5**, NRC VAD v1 0..1, **NRC VAD v2 -1..1**. Out-of-range values, NaN and infinities
  raise; nothing is clipped silently.
- **Checked here (2026-11-18):** EmoBank `corpus/emobank.csv` (10,062 rows; V 1.2–4.6,
  A 1.8–4.4, D 1.78–4.2), consistent with the 5-point SAM in Buechel & Hahn 2017. NRC VAD v2.1
  zip `README.txt` says scores range from -1 to 1, and the data file's min/max are exactly
  -1/1 on all three dimensions. The NRC VAD web page itself still describes v1's 0..1 range.
  The data were deleted after the check (no-redistribution terms).
- **Why:** out-of-range input means the wrong scale was chosen upstream, so it should fail loudly.

## Schema choices left open by the brief
- **Strict numbers:** probabilities, VAD values, keystroke times and `message_index` reject
  strings and booleans (datetimes still accept ISO-8601 strings, but they must carry a timezone).
  NaN and infinity are rejected everywhere (`allow_inf_nan=False`). `extra="forbid"` is set on
  every model, and `validate_assignment=True`, so a later `state.safety_flag = "yes"` fails.
- **Keystroke timestamps** are milliseconds on a monotonic clock (`performance.now()`), `>= 0`,
  with `up_ts >= down_ts`. `KeyCategory` adds `digit`, `modifier` and `other` to the brief's six.
- **`EngineInput.text` has `repr=False`**, so the raw message is left out of `repr()`, logs and
  tracebacks (§1.8).
- **`EngineOutput.features` is numeric only**, keyed by identifier-like names
  (`^[A-Za-z0-9_.:-]{1,64}$`), so free user text cannot be stored there. Text-derived
  explanations go in `EmotionalStateObject.explanations[].top_features` instead.
- **"Not estimated" is `None`:** `situation.goal_congruence`, `controllability` and `certainty`,
  and `pattern.baseline_deviation`, can be `None` (no appraisal engine run; no baseline yet).
  The brief's example always shows numbers, but filling in 0 would claim "neutral" where we
  actually know nothing (ground rule §1.1).
- **ESO `vad`, `situation`, `pattern` and `coarse` are required.** An unknown situation is
  `event_type="other_or_unclear"`, `agent="unknown"`.
- **Sorting:** `emotions` and `need` are re-sorted on validation (p descending, ties broken by
  canonical order), and duplicates are rejected. This means producers don't have to sort.
- **`safety_reason`** is a short code (max 200 chars), allowed only when `safety_flag` is true,
  and documented as never quoting user text. A flag with no reason is allowed.
- **Event taxonomy:** R3's 15 named types plus `other_or_unclear` (16), not the merged 13.
  **Need taxonomy:** the brief's six. Both are project proposals; ED labels are unverified (R3).
- **Language tags:** lowercase BCP-47-like (`en`, `en-GB`, `hy-Latn`). `SignalEngine.supports()`
  matches on the primary subtag and returns False for an unknown language.

## Safety rule enforcement in `Adapter`
- **Decision:** `Adapter.suggest` is sealed. It is a data descriptor (so an instance attribute
  cannot shadow it), and a metaclass refuses at class creation any subclass whose resolved
  `suggest` is not the core one (this also catches mixins earlier in the MRO) or that overrides
  `__getattribute__`. Assigning or deleting `suggest` on a class raises `TypeError`, and on an
  instance raises `AttributeError`. `suggest` treats any `safety_flag` value other than literal
  `False` as set (fail-safe for `model_construct`). It requires a real `EmotionalStateObject`
  and checks that `adapt` returned a `DomainSuggestion`.
- **Alternatives:** `typing.final` alone (static only, so it would not stop a runtime override);
  `__init_subclass__` checks (miss mixins and later patching).
- **Why:** brief §10 says this is a hard rule in the core. It cannot stop someone editing the
  core, or calling `adapt()` directly; callers must use `suggest()`.
- **Gap:** `SafetyResponse.resources` is empty for now. The helpline table must be curated and
  dated by hand, and its numbers checked by the owner (R3 §5: sources disagree, and Find a
  Helpline forbids scraping). `UserProfile.region` is the hook for choosing local resources.

## JSON Schemas
- **Decision:** `engine.schemas.json_schemas()` is the single source. `scripts/export_schemas.py`
  writes `docs/schemas/{engine_input,engine_output,emotional_state_object}.schema.json`
  (draft 2020-12, validation mode) and has a `--check` mode. A drift test compares the committed
  files with the models.
- **Gap:** cross-field rules (`up_ts >= down_ts`, sort order and uniqueness, `safety_reason`
  requiring the flag) are enforced by Pydantic only and are not expressible in the JSON Schema.
  Non-Python clients see a slightly looser contract.
