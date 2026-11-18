"""Pydantic schemas: engine I/O and the universal Emotional State Object (ESO).

Versioned via SCHEMA_VERSION. Export JSON Schema with `uv run python scripts/export_schema.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from engine.mappings.labels import CANONICAL, EKMAN, PLUTCHIK, Agent, EventType, NeedType

SCHEMA_VERSION = "0.1"

Prob = Annotated[float, Field(ge=0.0, le=1.0)]
Signed = Annotated[float, Field(ge=-1.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# ------------------------------------------------------------------ input


class KeyCategory(StrEnum):
    """Keystroke stream stores categories only — never the actual character (privacy, §4.2)."""

    LETTER = "letter"
    DIGIT = "digit"
    SPACE = "space"
    BACKSPACE = "backspace"
    DELETE = "delete"
    PUNCTUATION = "punctuation"
    ENTER = "enter"
    MODIFIER = "modifier"
    NAVIGATION = "navigation"
    OTHER = "other"


class KeystrokeEvent(_Strict):
    key_category: KeyCategory
    down_ts: float = Field(ge=0, description="ms since message composition started")
    up_ts: float = Field(ge=0)

    @model_validator(mode="after")
    def _order(self) -> KeystrokeEvent:
        if self.up_ts < self.down_ts:
            raise ValueError("up_ts must be >= down_ts")
        return self


class MessageContext(_Strict):
    user_id: str | None = None
    session_id: str | None = None
    message_index: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    language: str | None = Field(default=None, description="ISO 639-1; detected if None")


class EngineInput(_Strict):
    text: str
    keystroke_events: list[KeystrokeEvent] | None = None
    context: MessageContext = Field(default_factory=MessageContext)


# ------------------------------------------------------------------ shared parts


class VAD(_Strict):
    valence: Signed
    arousal: Signed
    dominance: Signed
    std: tuple[float, float, float] = Field(default=(1.0, 1.0, 1.0), description="per-dimension uncertainty, >= 0")

    @field_validator("std")
    @classmethod
    def _nonneg(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(s < 0 for s in v):
            raise ValueError("std must be non-negative")
        return v


class Situation(_Strict):
    event_type: EventType
    event_type_p: Prob = 0.0
    agent: Agent = Agent.UNKNOWN
    goal_congruence: Signed = 0.0
    certainty: Prob = 0.5
    controllability: Signed = 0.0
    norm_violation: bool = False
    evidence_spans: list[str] = Field(default_factory=list)


class NeedScore(_Strict):
    type: NeedType
    p: Prob


class EmotionScore(_Strict):
    label: str
    p: Prob

    @field_validator("label")
    @classmethod
    def _canonical(cls, v: str) -> str:
        if v not in CANONICAL:
            raise ValueError(f"unknown emotion label {v!r}")
        return v


class SafetyLevel(StrEnum):
    NONE = "none"
    CONCERN = "concern"  # distress indicators; products should soften and offer support
    URGENT = "urgent"  # possible self-harm/danger; products MUST show human help resources


class SafetyFlag(_Strict):
    level: SafetyLevel = SafetyLevel.NONE
    reasons: list[str] = Field(default_factory=list)


# ------------------------------------------------------------------ engine output


class EngineOutput(_Strict):
    """What every signal engine (A–F) returns. Same shape for all engines."""

    engine: str
    version: str
    vad: VAD | None = None
    emotions: dict[str, Prob] = Field(default_factory=dict, description="canonical label -> probability")
    features: dict[str, Any] = Field(default_factory=dict, description="raw values for explainability")
    confidence: Prob = 0.0
    coverage: Prob = Field(default=0.0, description="share of the input the engine could actually read")
    situation: Situation | None = None
    needs: list[NeedScore] = Field(default_factory=list)
    safety: SafetyFlag | None = None

    @field_validator("emotions")
    @classmethod
    def _labels(cls, v: dict[str, float]) -> dict[str, float]:
        bad = set(v) - set(CANONICAL)
        if bad:
            raise ValueError(f"non-canonical labels: {sorted(bad)}")
        return v

    @classmethod
    def empty(cls, engine: str, version: str, reason: str) -> EngineOutput:
        """Engine could not read the input (missing modality, unsupported language...)."""
        return cls(engine=engine, version=version, confidence=0.0, coverage=0.0, features={"skipped": reason})


# ------------------------------------------------------------------ Emotional State Object


class Trend(StrEnum):
    IMPROVING = "improving"
    STABLE = "stable"
    WORSENING = "worsening"
    UNKNOWN = "unknown"


class Pattern(_Strict):
    baseline_deviation: float = Field(default=0.0, description="valence z-score vs personal baseline")
    trend: Trend = Trend.UNKNOWN
    volatility: float = Field(default=0.0, ge=0)
    n_messages: int = Field(default=0, ge=0)
    recurring_triggers: list[EventType] = Field(default_factory=list)


class Explanation(_Strict):
    engine: str
    top_features: list[str] = Field(default_factory=list)


class Coarse(_Strict):
    plutchik: dict[str, Prob]
    ekman: dict[str, Prob]

    @model_validator(mode="after")
    def _keys(self) -> Coarse:
        if set(self.plutchik) != set(PLUTCHIK) or set(self.ekman) != set(EKMAN):
            raise ValueError("coarse views must contain exactly the Plutchik-8 / Ekman-6 labels")
        return self


class EmotionalStateObject(_Strict):
    """Universal, domain-neutral output of the Engine (brief §4.8)."""

    schema_version: str = SCHEMA_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    language: str = "en"
    vad: VAD
    emotions: list[EmotionScore] = Field(description="sorted by p, descending")
    coarse: Coarse
    intensity: Prob
    situation: Situation | None = None
    need: list[NeedScore] = Field(default_factory=list)
    pattern: Pattern = Field(default_factory=Pattern)
    overall_confidence: Prob
    mixed_or_ambiguous: bool = False
    uncertain: bool = Field(default=False, description="true when evidence is too weak to name an emotion")
    safety_flag: SafetyFlag = Field(default_factory=SafetyFlag)
    explanations: list[Explanation] = Field(default_factory=list)
    engine_versions: dict[str, str] = Field(default_factory=dict)

    @field_validator("emotions")
    @classmethod
    def _sorted(cls, v: list[EmotionScore]) -> list[EmotionScore]:
        return sorted(v, key=lambda e: e.p, reverse=True)
