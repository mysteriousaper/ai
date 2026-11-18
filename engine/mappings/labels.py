"""Canonical label sets and the explicit mapping tables between them.

Canonical discrete set = GoEmotions 27 emotions + neutral (order = GoEmotions label ids).
Coarse views: Ekman 6 (+ neutral) and Plutchik 8. Mappings are *weighted*: a fine label can
spread its probability over several coarse labels (e.g. Plutchik defines "remorse" as
sadness + disgust). Every row sums to 1, so probability mass is conserved. See tests.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum

GOEMOTIONS: tuple[str, ...] = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral",
)
CANONICAL = GOEMOTIONS
NEUTRAL = "neutral"

EKMAN: tuple[str, ...] = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
PLUTCHIK: tuple[str, ...] = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")

# Source: GoEmotions repo `ekman_mapping.json` (Demszky et al. 2020). Single-target, weight 1.
_EKMAN_GROUPS: dict[str, tuple[str, ...]] = {
    "anger": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fear": ("fear", "nervousness"),
    "joy": ("joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring"),
    "sadness": ("sadness", "disappointment", "embarrassment", "grief", "remorse"),
    "surprise": ("surprise", "realization", "confusion", "curiosity"),
}
TO_EKMAN: dict[str, dict[str, float]] = {
    fine: {coarse: 1.0} for coarse, fines in _EKMAN_GROUPS.items() for fine in fines
}

# Our mapping, based on Plutchik's wheel and its primary dyads
# (e.g. remorse = sadness + disgust, disappointment = surprise + sadness,
# optimism = anticipation + joy, love = joy + trust, curiosity = trust + surprise).
# Documented in docs/DECISIONS.md ADR-007.
TO_PLUTCHIK: dict[str, dict[str, float]] = {
    "admiration": {"trust": 1.0},
    "amusement": {"joy": 1.0},
    "anger": {"anger": 1.0},
    "annoyance": {"anger": 1.0},
    "approval": {"trust": 1.0},
    "caring": {"trust": 0.5, "joy": 0.5},
    "confusion": {"surprise": 1.0},
    "curiosity": {"trust": 0.5, "surprise": 0.5},
    "desire": {"anticipation": 0.5, "joy": 0.5},
    "disappointment": {"surprise": 0.5, "sadness": 0.5},
    "disapproval": {"anger": 0.5, "disgust": 0.5},
    "disgust": {"disgust": 1.0},
    "embarrassment": {"fear": 0.5, "sadness": 0.5},
    "excitement": {"joy": 0.5, "anticipation": 0.5},
    "fear": {"fear": 1.0},
    "gratitude": {"joy": 0.5, "trust": 0.5},
    "grief": {"sadness": 1.0},
    "joy": {"joy": 1.0},
    "love": {"joy": 0.5, "trust": 0.5},
    "nervousness": {"fear": 1.0},
    "optimism": {"anticipation": 0.5, "joy": 0.5},
    "pride": {"joy": 1.0},
    "realization": {"surprise": 1.0},
    "relief": {"joy": 1.0},
    "remorse": {"sadness": 0.5, "disgust": 0.5},
    "sadness": {"sadness": 1.0},
    "surprise": {"surprise": 1.0},
}


def project(probs: Mapping[str, float], table: Mapping[str, Mapping[str, float]], targets: tuple[str, ...]) -> dict[str, float]:
    """Project fine-label probabilities onto a coarse label set.

    Multi-label probabilities are not a distribution, so we take, for each coarse label,
    the *max* weighted contribution (a noisy-OR would over-count correlated fine labels).
    Result values stay in [0, 1].
    """
    out = dict.fromkeys(targets, 0.0)
    for label, p in probs.items():
        for coarse, w in table.get(label, {}).items():
            out[coarse] = max(out[coarse], w * p)
    return out


def to_ekman(probs: Mapping[str, float]) -> dict[str, float]:
    return project(probs, TO_EKMAN, EKMAN)


def to_plutchik(probs: Mapping[str, float]) -> dict[str, float]:
    return project(probs, TO_PLUTCHIK, PLUTCHIK)


# ---------------------------------------------------------------- other datasets → canonical

# DailyDialog integer emotion ids (dataset card): 0 no emotion, 1 anger, 2 disgust, 3 fear,
# 4 happiness, 5 sadness, 6 surprise.
DAILYDIALOG_TO_CANONICAL: dict[int, str] = {
    0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "joy", 5: "sadness", 6: "surprise",
}

# SemEval-2018 E-c (11 labels) → canonical. "pessimism" and "trust" have no GoEmotions twin;
# mapped to the closest label and flagged in ADR-007.
SEMEVAL_EC_TO_CANONICAL: dict[str, str] = {
    "anger": "anger", "anticipation": "desire", "disgust": "disgust", "fear": "fear", "joy": "joy",
    "love": "love", "optimism": "optimism", "pessimism": "sadness", "sadness": "sadness",
    "surprise": "surprise", "trust": "approval",
}

# EmoSurv keystroke classes → canonical.
EMOSURV_TO_CANONICAL: dict[str, str] = {
    "anger": "anger", "happiness": "joy", "calmness": "neutral", "sadness": "sadness", "neutral": "neutral",
}


# ---------------------------------------------------------------- situation ("what happened")


class EventType(StrEnum):
    """Event taxonomy v0.1. Built from OCC branches + EmpatheticDialogues situations."""

    LOSS_OR_FAILURE = "loss_or_failure"
    OWN_MISTAKE_OR_WRONGDOING = "own_mistake_or_wrongdoing"
    CONFLICT_OR_MISTREATMENT = "conflict_or_mistreatment"
    UNFAIRNESS = "unfairness"
    THREAT_OR_DANGER = "threat_or_danger"
    UNCERTAINTY_ABOUT_FUTURE = "uncertainty_about_future"
    ACHIEVEMENT_OR_SUCCESS = "achievement_or_success"
    POSITIVE_ANTICIPATION = "positive_anticipation"
    PLEASANT_EVENT = "pleasant_event"
    SOCIAL_CONNECTION = "social_connection"
    REJECTION_OR_LONELINESS = "rejection_or_loneliness"
    REMINISCENCE = "reminiscence"
    SURPRISE_EVENT = "surprise_event"
    ROUTINE_OR_BOREDOM = "routine_or_boredom"
    HEALTH_OR_BODY = "health_or_body"
    OTHER = "other"


class Agent(StrEnum):
    SELF = "self"
    OTHER = "other"
    CIRCUMSTANCE = "circumstance"
    UNKNOWN = "unknown"


class NeedType(StrEnum):
    EMOTIONAL_SUPPORT = "emotional_support"
    PRACTICAL_SOLUTION = "practical_solution"
    INFORMATION = "information"
    CHANGE_OF_SCENE = "change_of_scene"
    REST = "rest"
    CELEBRATION = "celebration"
    CONNECTION = "connection"


# EmpatheticDialogues' 32 situation-emotion labels → event type. Used to train/evaluate the
# local fallback of engine E. Lossy by design: ED labels name the *feeling*; the event type is
# the most typical cause. UNFAIRNESS, ROUTINE_OR_BOREDOM, HEALTH_OR_BODY have no ED source and
# come only from the LLM path.
ED_TO_EVENT: dict[str, EventType] = {
    "sad": EventType.LOSS_OR_FAILURE,
    "devastated": EventType.LOSS_OR_FAILURE,
    "disappointed": EventType.LOSS_OR_FAILURE,
    "guilty": EventType.OWN_MISTAKE_OR_WRONGDOING,
    "ashamed": EventType.OWN_MISTAKE_OR_WRONGDOING,
    "embarrassed": EventType.OWN_MISTAKE_OR_WRONGDOING,
    "angry": EventType.CONFLICT_OR_MISTREATMENT,
    "furious": EventType.CONFLICT_OR_MISTREATMENT,
    "annoyed": EventType.CONFLICT_OR_MISTREATMENT,
    "disgusted": EventType.CONFLICT_OR_MISTREATMENT,
    "afraid": EventType.THREAT_OR_DANGER,
    "terrified": EventType.THREAT_OR_DANGER,
    "anxious": EventType.UNCERTAINTY_ABOUT_FUTURE,
    "apprehensive": EventType.UNCERTAINTY_ABOUT_FUTURE,
    "proud": EventType.ACHIEVEMENT_OR_SUCCESS,
    "confident": EventType.ACHIEVEMENT_OR_SUCCESS,
    "excited": EventType.POSITIVE_ANTICIPATION,
    "anticipating": EventType.POSITIVE_ANTICIPATION,
    "hopeful": EventType.POSITIVE_ANTICIPATION,
    "prepared": EventType.POSITIVE_ANTICIPATION,
    "joyful": EventType.PLEASANT_EVENT,
    "content": EventType.PLEASANT_EVENT,
    "grateful": EventType.SOCIAL_CONNECTION,
    "caring": EventType.SOCIAL_CONNECTION,
    "trusting": EventType.SOCIAL_CONNECTION,
    "faithful": EventType.SOCIAL_CONNECTION,
    "impressed": EventType.SOCIAL_CONNECTION,
    "lonely": EventType.REJECTION_OR_LONELINESS,
    "jealous": EventType.REJECTION_OR_LONELINESS,
    "nostalgic": EventType.REMINISCENCE,
    "sentimental": EventType.REMINISCENCE,
    "surprised": EventType.SURPRISE_EVENT,
}

# Typical agent per ED label (weak prior for the fallback; the LLM decides per message).
ED_TO_AGENT: dict[str, Agent] = {
    **dict.fromkeys(["guilty", "ashamed", "embarrassed", "proud", "confident", "prepared"], Agent.SELF),
    **dict.fromkeys(["angry", "furious", "annoyed", "disgusted", "grateful", "impressed", "jealous", "trusting",
                     "faithful", "caring"], Agent.OTHER),
}
