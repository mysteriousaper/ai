"""Pydantic v2 data contracts for the Engine.

* Input capture (brief §4.2): :class:`KeystrokeEvent`, :class:`MessageContext`,
  :class:`EngineInput`.
* Per-engine output (§4.4): :class:`VAD`, :class:`EngineOutput`.
* The universal, versioned API output (§4.8): :class:`EmotionalStateObject`.

Every model forbids unknown fields and rejects NaN / infinity. Numbers are strict (no
``"0.5"`` strings, no booleans for floats), so a wrong type fails loudly instead of being coerced.

Privacy (brief §1.8): the keystroke stream carries only :class:`KeyCategory` values, never
characters, and :class:`EngineInput` keeps the message text out of ``repr``/``str`` so it does
not end up in logs or tracebacks by accident.
"""

from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)

from engine.mappings.labels import GOEMOTIONS_LABELS

SCHEMA_VERSION: Literal["0.1"] = "0.1"

# --- shared types -----------------------------------------------------------------------------

Probability = Annotated[float, Field(ge=0.0, le=1.0, strict=True)]
Signed = Annotated[float, Field(ge=-1.0, le=1.0, strict=True)]
NonNegative = Annotated[float, Field(ge=0.0, strict=True)]
FiniteFloat = Annotated[float, Field(strict=True)]
NonEmptyStr = Annotated[str, Field(min_length=1)]
# BCP-47-like: lowercase ISO 639 primary subtag plus optional subtags ("en", "en-GB", "hy-Latn").
LanguageTag = Annotated[str, Field(pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})*$")]
# Feature names are identifiers, so free text cannot be smuggled in through the keys.
FeatureName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:\-]{1,64}$")]


class StrictModel(BaseModel):
    # hide_input_in_errors: a ValidationError's message would otherwise echo the offending input,
    # e.g. a missing field shows the whole dict including EngineInput.text (brief §1.8).
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_assignment=True,
        hide_input_in_errors=True,
    )


# --- enums ------------------------------------------------------------------------------------


class KeyCategory(StrEnum):
    """What kind of key was pressed. Holds categories only, never characters (brief §4.2)."""

    letter = "letter"
    digit = "digit"
    space = "space"
    backspace = "backspace"
    delete = "delete"
    punctuation = "punctuation"
    enter = "enter"
    modifier = "modifier"
    other = "other"


class EmotionLabel(StrEnum):
    """Canonical discrete labels: GoEmotions 27 + neutral, in the official emotions.txt order.

    Must equal :data:`engine.mappings.labels.GOEMOTIONS_LABELS` (checked at import and by tests).
    """

    admiration = "admiration"
    amusement = "amusement"
    anger = "anger"
    annoyance = "annoyance"
    approval = "approval"
    caring = "caring"
    confusion = "confusion"
    curiosity = "curiosity"
    desire = "desire"
    disappointment = "disappointment"
    disapproval = "disapproval"
    disgust = "disgust"
    embarrassment = "embarrassment"
    excitement = "excitement"
    fear = "fear"
    gratitude = "gratitude"
    grief = "grief"
    joy = "joy"
    love = "love"
    nervousness = "nervousness"
    optimism = "optimism"
    pride = "pride"
    realization = "realization"
    relief = "relief"
    remorse = "remorse"
    sadness = "sadness"
    surprise = "surprise"
    neutral = "neutral"


if tuple(EmotionLabel) != GOEMOTIONS_LABELS:  # pragma: no cover - guards a code edit
    raise RuntimeError("EmotionLabel is out of sync with GOEMOTIONS_LABELS")

_LABEL_ORDER: dict[str, int] = {label: i for i, label in enumerate(GOEMOTIONS_LABELS)}


class EkmanLabel(StrEnum):
    """Ekman's six, in the order of GoEmotions' ekman_mapping.json."""

    anger = "anger"
    disgust = "disgust"
    fear = "fear"
    joy = "joy"
    sadness = "sadness"
    surprise = "surprise"


class PlutchikLabel(StrEnum):
    """Plutchik's eight primaries, in wheel order (Plutchik 2001)."""

    joy = "joy"
    trust = "trust"
    fear = "fear"
    surprise = "surprise"
    sadness = "sadness"
    disgust = "disgust"
    anger = "anger"
    anticipation = "anticipation"


class EventType(StrEnum):
    """What probably happened (brief §0.2, §4.4 E).

    Project proposal, not a published taxonomy (docs/research/notes/R3, §2). Built from the
    EmpatheticDialogues situation labels (Rashkin et al. 2019, "Towards Empathetic Open-domain
    Conversation Models: a New Benchmark and Dataset", ACL 2019), OCC (Ortony, Clore & Collins
    1988, *The Cognitive Structure of Emotions*) and the appraisal dimensions of Smith & Ellsworth
    1985 (J. Pers. Soc. Psychol. 48(4):813-838) as annotated by Hofmann et al. 2020 (COLING).
    """

    loss_or_failure = "loss_or_failure"
    conflict_or_mistreatment = "conflict_or_mistreatment"
    achievement_or_success = "achievement_or_success"
    threat_or_danger = "threat_or_danger"
    uncertainty_about_future = "uncertainty_about_future"
    positive_anticipation = "positive_anticipation"
    own_transgression = "own_transgression"
    social_exposure = "social_exposure"
    isolation_or_absence = "isolation_or_absence"
    kindness_or_support_received = "kindness_or_support_received"
    caring_for_others = "caring_for_others"
    comparison_or_envy = "comparison_or_envy"
    reminiscence = "reminiscence"
    unexpected_event = "unexpected_event"
    routine_or_wellbeing = "routine_or_wellbeing"
    other_or_unclear = "other_or_unclear"


class Agent(StrEnum):
    """Who or what the person holds responsible (appraisal "responsibility"; Smith & Ellsworth
    1985; "circumstances" as in Hofmann et al. 2020)."""

    self_ = "self"
    other = "other"
    circumstance = "circumstance"
    unknown = "unknown"


class NeedType(StrEnum):
    """What the person probably needs (brief §0.3, §4.4 E).

    The first three follow social-support types (emotional, informational, instrumental:
    House 1981, *Work Stress and Social Support*; Cutrona & Russell 1990). ``distraction``,
    ``rest`` and ``celebration`` come from the brief and have no labelled data yet (R3 §3).
    """

    emotional_support = "emotional_support"
    solution = "solution"
    information = "information"
    distraction = "distraction"
    rest = "rest"
    celebration = "celebration"


class Trend(StrEnum):
    """Direction of the person's state over recent messages (brief §4.7)."""

    improving = "improving"
    worsening = "worsening"
    stable = "stable"
    unknown = "unknown"


# --- input capture (§4.2) ---------------------------------------------------------------------


class KeystrokeEvent(StrictModel):
    """One key press. Timestamps are milliseconds on a monotonic clock (e.g. ``performance.now()``)."""

    key_category: KeyCategory
    down_ts: NonNegative
    up_ts: NonNegative

    @model_validator(mode="after")
    def _up_after_down(self) -> Self:
        if self.up_ts < self.down_ts:
            raise ValueError("up_ts must be >= down_ts")
        return self


class MessageContext(StrictModel):
    session_id: NonEmptyStr
    user_id: NonEmptyStr | None = None
    message_index: Annotated[StrictInt, Field(ge=0)]
    timestamp: AwareDatetime
    language: LanguageTag | None = None


class EngineInput(StrictModel):
    # repr=False: raw user text never shows up in repr(), logs or tracebacks (brief §1.8).
    text: str = Field(repr=False)
    keystroke_events: list[KeystrokeEvent] | None = None
    context: MessageContext


# --- engine output (§4.4) ---------------------------------------------------------------------


class VAD(StrictModel):
    """Valence, arousal, dominance on the common [-1, 1] scale, with one std per dimension."""

    valence: Signed
    arousal: Signed
    dominance: Signed
    std: tuple[NonNegative, NonNegative, NonNegative]


class EngineOutput(StrictModel):
    engine: NonEmptyStr
    engine_version: NonEmptyStr
    vad: VAD | None = None
    emotions: dict[EmotionLabel, Probability]
    # Numeric only: raw values for explainability must not carry user text.
    features: dict[FeatureName, FiniteFloat] = Field(default_factory=dict)
    confidence: Probability
    coverage: Probability


# --- Emotional State Object (§4.8) ------------------------------------------------------------


class EmotionScore(StrictModel):
    label: EmotionLabel
    p: Probability


class NeedScore(StrictModel):
    type: NeedType
    p: Probability


class Coarse(StrictModel):
    plutchik: dict[PlutchikLabel, Probability]
    ekman: dict[EkmanLabel, Probability]


class Situation(StrictModel):
    """Appraisal of the event behind the feeling. ``None`` means "not estimated"."""

    event_type: EventType
    agent: Agent
    goal_congruence: Signed | None
    controllability: Signed | None
    certainty: Signed | None
    evidence_spans: list[NonEmptyStr]


class Pattern(StrictModel):
    """State over time. ``baseline_deviation`` is a z-score; ``None`` until a baseline exists."""

    baseline_deviation: FiniteFloat | None
    trend: Trend
    volatility: NonNegative


class Explanation(StrictModel):
    engine: NonEmptyStr
    top_features: list[NonEmptyStr]


def _sorted_unique(items: list[Any], key: str, order: dict[str, int]) -> list[Any]:
    seen: set[str] = set()
    for it in items:
        k = getattr(it, key)
        if k in seen:
            raise ValueError(f"duplicate {key} {str(k)!r}")
        seen.add(k)
    return sorted(items, key=lambda it: (-it.p, order[getattr(it, key)]))


_NEED_ORDER: dict[str, int] = {n.value: i for i, n in enumerate(NeedType)}


class EmotionalStateObject(StrictModel):
    """The universal, domain-neutral output (brief §4.8), versioned by ``schema_version``.

    ``emotions`` and ``need`` are always stored sorted by ``p``, descending (ties by canonical
    order). ``safety_flag`` is the core's crisis rule (brief §10): products must show human help
    resources instead of a suggestion when it is set; see ``engine.adapters.base.Adapter``.
    ``safety_reason`` is a short machine-readable reason and must never quote the user's text.
    """

    schema_version: Literal["0.1"] = SCHEMA_VERSION
    timestamp: AwareDatetime
    language: LanguageTag | None
    vad: VAD
    emotions: list[EmotionScore]
    coarse: Coarse
    intensity: Probability
    situation: Situation
    need: list[NeedScore]
    pattern: Pattern
    overall_confidence: Probability
    mixed_or_ambiguous: StrictBool
    explanations: list[Explanation]
    engine_versions: dict[NonEmptyStr, NonEmptyStr]
    safety_flag: StrictBool = False
    safety_reason: Annotated[str, Field(min_length=1, max_length=200)] | None = None

    @field_validator("emotions")
    @classmethod
    def _sort_emotions(cls, v: list[EmotionScore]) -> list[EmotionScore]:
        return _sorted_unique(v, "label", _LABEL_ORDER)

    @field_validator("need")
    @classmethod
    def _sort_need(cls, v: list[NeedScore]) -> list[NeedScore]:
        return _sorted_unique(v, "type", _NEED_ORDER)

    @model_validator(mode="after")
    def _reason_needs_flag(self) -> Self:
        if self.safety_reason is not None and not self.safety_flag:
            raise ValueError("safety_reason is only allowed when safety_flag is true")
        return self


# --- JSON Schema export -----------------------------------------------------------------------

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"

EXPORTED_MODELS: dict[str, type[BaseModel]] = {
    "engine_input": EngineInput,
    "engine_output": EngineOutput,
    "emotional_state_object": EmotionalStateObject,
}


def json_schemas() -> dict[str, dict[str, Any]]:
    """JSON Schemas (draft 2020-12, validation mode) for the exported models, keyed by file stem."""
    out: dict[str, dict[str, Any]] = {}
    for stem, model in EXPORTED_MODELS.items():
        schema = model.model_json_schema(mode="validation")
        out[stem] = {"$schema": JSON_SCHEMA_DIALECT, **schema}
    return out


__all__ = [
    "SCHEMA_VERSION",
    "VAD",
    "Agent",
    "Coarse",
    "EkmanLabel",
    "EmotionLabel",
    "EmotionScore",
    "EmotionalStateObject",
    "EngineInput",
    "EngineOutput",
    "EventType",
    "Explanation",
    "KeyCategory",
    "KeystrokeEvent",
    "MessageContext",
    "NeedScore",
    "NeedType",
    "Pattern",
    "PlutchikLabel",
    "Situation",
    "Trend",
    "json_schemas",
]
