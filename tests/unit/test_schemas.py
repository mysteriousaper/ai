"""Pydantic schemas: input capture, engine output and the Emotional State Object (§4.2, §4.4, §4.8)."""

import math
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from engine import schemas as S
from engine.mappings.labels import EKMAN_LABELS, GOEMOTIONS_LABELS, PLUTCHIK_LABELS

TS = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)


def ctx(**kw: Any) -> S.MessageContext:
    base: dict[str, Any] = {"session_id": "s1", "message_index": 0, "timestamp": TS}
    base.update(kw)
    return S.MessageContext(**base)


def vad(**kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "valence": -0.62,
        "arousal": 0.41,
        "dominance": -0.30,
        "std": [0.1, 0.15, 0.2],
    }
    base.update(kw)
    return base


def eso_dict(**kw: Any) -> dict[str, Any]:
    """The brief's §4.8 example, with the ``...`` placeholders filled in."""
    base: dict[str, Any] = {
        "schema_version": "0.1",
        "timestamp": "2026-11-18T12:00:00Z",
        "language": "en",
        "vad": vad(),
        "emotions": [{"label": "sadness", "p": 0.71}, {"label": "disappointment", "p": 0.44}],
        "coarse": {
            "plutchik": {"sadness": 0.8, "surprise": 0.44},
            "ekman": {"sadness": 0.84},
        },
        "intensity": 0.66,
        "situation": {
            "event_type": "loss_or_failure",
            "agent": "circumstance",
            "goal_congruence": -0.8,
            "controllability": -0.5,
            "certainty": 0.2,
            "evidence_spans": ["missed the last train"],
        },
        "need": [{"type": "emotional_support", "p": 0.6}, {"type": "solution", "p": 0.3}],
        "pattern": {"baseline_deviation": -1.4, "trend": "worsening", "volatility": 0.2},
        "overall_confidence": 0.58,
        "mixed_or_ambiguous": False,
        "explanations": [{"engine": "lexicon", "top_features": ["missed"]}],
        "engine_versions": {"lexicon": "0.1.0"},
    }
    base.update(kw)
    return base


# --- enums ------------------------------------------------------------------------------------


def test_key_category_members() -> None:
    assert [k.value for k in S.KeyCategory] == [
        "letter", "digit", "space", "backspace", "delete", "punctuation", "enter",
        "modifier", "other",
    ]


def test_label_enums_match_mapping_tables() -> None:
    assert tuple(e.value for e in S.EmotionLabel) == GOEMOTIONS_LABELS
    assert tuple(e.value for e in S.EkmanLabel) == EKMAN_LABELS
    assert tuple(e.value for e in S.PlutchikLabel) == PLUTCHIK_LABELS


def test_event_and_need_taxonomies() -> None:
    assert len(S.EventType) == 16
    assert "other_or_unclear" in {e.value for e in S.EventType}
    assert [n.value for n in S.NeedType] == [
        "emotional_support", "solution", "information", "distraction", "rest", "celebration",
    ]
    assert [a.value for a in S.Agent] == ["self", "other", "circumstance", "unknown"]
    assert [t.value for t in S.Trend] == ["improving", "worsening", "stable", "unknown"]


# --- keystrokes and context -------------------------------------------------------------------


def test_keystroke_event_valid() -> None:
    ev = S.KeystrokeEvent(key_category="letter", down_ts=10.0, up_ts=95.5)
    assert ev.key_category is S.KeyCategory.letter


@pytest.mark.parametrize("cat", ["a", "A", " ", "Letter", "", 1])
def test_keystroke_rejects_characters_and_unknown_categories(cat: object) -> None:
    with pytest.raises(ValidationError):
        S.KeystrokeEvent(key_category=cat, down_ts=0.0, up_ts=1.0)


def test_keystroke_rejects_extra_fields_such_as_the_character() -> None:
    with pytest.raises(ValidationError):
        S.KeystrokeEvent(key_category="letter", down_ts=0.0, up_ts=1.0, key="a")


@pytest.mark.parametrize(
    ("down", "up"), [(5.0, 4.9), (-1.0, 1.0), (math.nan, 1.0), (0.0, math.inf), (True, 2.0)]
)
def test_keystroke_rejects_bad_times(down: float, up: float) -> None:
    with pytest.raises(ValidationError):
        S.KeystrokeEvent(key_category="space", down_ts=down, up_ts=up)


def test_keystroke_equal_times_allowed() -> None:
    S.KeystrokeEvent(key_category="enter", down_ts=3.0, up_ts=3.0)


def test_context_requires_aware_timestamp() -> None:
    with pytest.raises(ValidationError):
        ctx(timestamp=datetime(2026, 11, 18, 12, 0))


@pytest.mark.parametrize("idx", [-1, 1.5, "2", True])
def test_context_message_index(idx: object) -> None:
    with pytest.raises(ValidationError):
        ctx(message_index=idx)


@pytest.mark.parametrize("lang", ["en", "hy", "ru", "en-GB", "hy-Latn", None])
def test_context_language_ok(lang: str | None) -> None:
    assert ctx(language=lang).language == lang


@pytest.mark.parametrize("lang", ["english", "EN", "", "e"])
def test_context_language_bad(lang: str) -> None:
    with pytest.raises(ValidationError):
        ctx(language=lang)


def test_context_session_id_non_empty() -> None:
    with pytest.raises(ValidationError):
        ctx(session_id="")


def test_engine_input_and_repr_hides_text() -> None:
    secret = "I feel terrible today"
    ei = S.EngineInput(
        text=secret,
        keystroke_events=[{"key_category": "letter", "down_ts": 0.0, "up_ts": 50.0}],
        context=ctx(user_id="u1"),
    )
    assert ei.text == secret
    assert secret not in repr(ei)
    assert secret not in str(ei)
    assert S.EngineInput(text="hi", context=ctx()).keystroke_events is None


# --- VAD and EngineOutput ---------------------------------------------------------------------


def test_vad_valid_and_bounds() -> None:
    v = S.VAD(**vad(valence=1.0, arousal=-1.0))
    assert v.std == (0.1, 0.15, 0.2)


@pytest.mark.parametrize(
    "bad",
    [
        {"valence": 1.01},
        {"arousal": -1.5},
        {"dominance": math.nan},
        {"std": [0.1, 0.1]},
        {"std": [0.1, 0.1, 0.1, 0.1]},
        {"std": [0.1, -0.01, 0.1]},
        {"std": [0.1, math.inf, 0.1]},
        {"extra": 1},
    ],
)
def test_vad_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        S.VAD(**vad(**bad))


def engine_output(**kw: Any) -> S.EngineOutput:
    base: dict[str, Any] = {
        "engine": "lexicon",
        "engine_version": "0.1.0",
        "vad": vad(),
        "emotions": {"sadness": 0.7, "neutral": 0.1},
        "features": {"neg_count": 2, "valence_mean": -0.4},
        "confidence": 0.5,
        "coverage": 0.8,
    }
    base.update(kw)
    return S.EngineOutput(**base)


def test_engine_output_valid() -> None:
    out = engine_output(vad=None)
    assert out.vad is None
    assert out.emotions["sadness"] == 0.7


@pytest.mark.parametrize(
    "bad",
    [
        {"emotions": {"happiness": 0.5}},
        {"emotions": {"Sadness": 0.5}},
        {"emotions": {"joy": 1.2}},
        {"emotions": {"joy": -0.1}},
        {"emotions": {"joy": math.nan}},
        {"confidence": 1.01},
        {"coverage": -0.01},
        {"features": {"word": "sad"}},
        {"features": {"x": math.nan}},
        {"features": {"bad key with spaces": 1.0}},
        {"engine": ""},
        {"extra": 1},
    ],
)
def test_engine_output_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        engine_output(**bad)


# --- Emotional State Object -------------------------------------------------------------------


def test_eso_brief_example_validates_and_round_trips() -> None:
    eso = S.EmotionalStateObject.model_validate(eso_dict())
    assert eso.schema_version == "0.1"
    assert eso.safety_flag is False and eso.safety_reason is None
    again = S.EmotionalStateObject.model_validate_json(eso.model_dump_json())
    assert again == eso


def test_eso_emotions_and_needs_sorted_descending() -> None:
    eso = S.EmotionalStateObject.model_validate(
        eso_dict(
            emotions=[
                {"label": "joy", "p": 0.1},
                {"label": "sadness", "p": 0.9},
                {"label": "anger", "p": 0.1},
                {"label": "fear", "p": 0.5},
            ],
            need=[{"type": "rest", "p": 0.2}, {"type": "solution", "p": 0.7}],
        )
    )
    assert [e.label for e in eso.emotions] == ["sadness", "fear", "anger", "joy"]
    assert [n.type for n in eso.need] == ["solution", "rest"]


@pytest.mark.parametrize(
    "bad",
    [
        {"schema_version": "0.2"},
        {"timestamp": "2026-11-18T12:00:00"},
        {"emotions": [{"label": "happiness", "p": 0.5}]},
        {"emotions": [{"label": "joy", "p": 0.5}, {"label": "joy", "p": 0.4}]},
        {"need": [{"type": "rest", "p": 0.5}, {"type": "rest", "p": 0.4}]},
        {"need": [{"type": "hug", "p": 0.5}]},
        {"coarse": {"plutchik": {"love": 0.5}, "ekman": {}}},
        {"coarse": {"plutchik": {}, "ekman": {"trust": 0.5}}},
        {"intensity": 1.5},
        {"overall_confidence": -0.1},
        {"mixed_or_ambiguous": "no"},
        {"safety_reason": "self_harm_explicit"},
        {"unexpected": True},
    ],
)
def test_eso_rejects(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        S.EmotionalStateObject.model_validate(eso_dict(**bad))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_type", "lost_keys"),
        ("agent", "god"),
        ("goal_congruence", 1.1),
        ("controllability", -1.1),
        ("certainty", math.nan),
        ("evidence_spans", [""]),
        ("extra", 1),
    ],
)
def test_situation_rejects(field: str, value: object) -> None:
    sit = eso_dict()["situation"] | {field: value}
    with pytest.raises(ValidationError):
        S.EmotionalStateObject.model_validate(eso_dict(situation=sit))


def test_situation_appraisals_may_be_unknown() -> None:
    sit = {
        "event_type": "other_or_unclear",
        "agent": "unknown",
        "goal_congruence": None,
        "controllability": None,
        "certainty": None,
        "evidence_spans": [],
    }
    eso = S.EmotionalStateObject.model_validate(eso_dict(situation=sit))
    assert eso.situation.certainty is None


@pytest.mark.parametrize(
    "bad", [{"trend": "up"}, {"volatility": -0.1}, {"baseline_deviation": math.inf}]
)
def test_pattern_rejects(bad: dict[str, Any]) -> None:
    pat = eso_dict()["pattern"] | bad
    with pytest.raises(ValidationError):
        S.EmotionalStateObject.model_validate(eso_dict(pattern=pat))


def test_pattern_without_baseline() -> None:
    pat = {"baseline_deviation": None, "trend": "unknown", "volatility": 0.0}
    assert S.EmotionalStateObject.model_validate(eso_dict(pattern=pat)).pattern.trend == "unknown"


def test_safety_flag_with_reason() -> None:
    eso = S.EmotionalStateObject.model_validate(
        eso_dict(safety_flag=True, safety_reason="self_harm_explicit")
    )
    assert eso.safety_flag is True
    flagged_no_reason = S.EmotionalStateObject.model_validate(eso_dict(safety_flag=True))
    assert flagged_no_reason.safety_reason is None


def test_safety_flag_must_be_a_real_bool() -> None:
    with pytest.raises(ValidationError):
        S.EmotionalStateObject.model_validate(eso_dict(safety_flag="yes"))


@pytest.mark.parametrize(
    "bad",
    [
        {"text": "SECRET user message"},  # missing context: error used to echo the whole dict
        "SECRET user message",  # not a dict at all
    ],
)
def test_validation_errors_do_not_echo_raw_text(bad: Any) -> None:
    with pytest.raises(ValidationError) as info:
        S.EngineInput.model_validate(bad)
    assert "SECRET" not in str(info.value)
    assert "SECRET" not in repr(info.value)


def test_json_validation_errors_do_not_echo_raw_text() -> None:
    with pytest.raises(ValidationError) as info:
        S.EngineInput.model_validate_json('{"text": "SECRET user message"}')
    assert "SECRET" not in str(info.value)
