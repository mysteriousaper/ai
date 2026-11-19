import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from engine import safety
from engine.adapters.base import Adapter, run_adapter
from engine.adapters.travel_v0 import TravelV0
from engine.engines.appraisal import AppraisalEngine
from engine.engines.keystroke import KeystrokeEngine
from engine.engines.lexicon import LexiconEngine
from engine.mappings.labels import EKMAN, PLUTCHIK, NeedType
from engine.schemas import (
    SCHEMA_VERSION,
    VAD,
    Attribution,
    Coarse,
    DesiredGoal,
    DesiredState,
    EmotionalStateObject,
    EmotionScore,
    EngineInput,
    EngineOutput,
    KeystrokeEvent,
    MessageContext,
    NeedScore,
    SafetyFlag,
    SafetyLevel,
    TemporalFocus,
)


def make_state(v=-0.6, a=-0.4, needs=(), safety_level=SafetyLevel.NONE, **kw):
    return EmotionalStateObject(
        vad=VAD(valence=v, arousal=a, dominance=-0.2, std=(0.1, 0.1, 0.2)),
        emotions=[EmotionScore(label="neutral", p=0.1), EmotionScore(label="sadness", p=0.7)],
        coarse=Coarse(plutchik=dict.fromkeys(PLUTCHIK, 0.0), ekman=dict.fromkeys(EKMAN, 0.0)),
        intensity=0.6,
        need=[NeedScore(type=t, p=p) for t, p in needs],
        overall_confidence=0.55,
        safety_flag=SafetyFlag(level=safety_level),
        **kw,
    )


# ---------------------------------------------------------------- schemas


def test_eso_roundtrips_json_and_sorts_emotions():
    s = make_state()
    assert s.schema_version == SCHEMA_VERSION
    assert [e.label for e in s.emotions] == ["sadness", "neutral"]
    again = EmotionalStateObject.model_validate_json(s.model_dump_json())
    assert again == s


@pytest.mark.parametrize("bad", [
    lambda: VAD(valence=1.5, arousal=0, dominance=0),
    lambda: VAD(valence=0, arousal=0, dominance=0, std=(-0.1, 0, 0)),
    lambda: EmotionScore(label="happiness", p=0.5),
    lambda: EmotionScore(label="joy", p=1.2),
    lambda: EngineOutput(engine="x", version="0", emotions={"happy": 0.5}),
    lambda: KeystrokeEvent(key_category="letter", down_ts=10, up_ts=5),
    lambda: KeystrokeEvent(key_category="the_letter_a", down_ts=0, up_ts=5),
    lambda: Coarse(plutchik={"joy": 1.0}, ekman=dict.fromkeys(EKMAN, 0.0)),
])
def test_invalid_values_rejected(bad):
    with pytest.raises(ValidationError):
        bad()


def test_exported_json_schema_is_current():
    path = Path(f"docs/schema/emotional_state_object.v{SCHEMA_VERSION}.json")
    assert json.loads(path.read_text()) == EmotionalStateObject.model_json_schema(), "run scripts/export_schema.py"


def test_schema_version_is_02():
    assert SCHEMA_VERSION == "0.2"


def test_v02_defaults_are_backwards_compatible():
    s = make_state()
    assert s.desired_state is None and s.attributions == []
    assert s.temporal_focus is TemporalFocus.UNKNOWN and s.sarcasm_suspected is False
    o = EngineOutput(engine="x", version="0")
    assert o.attributions == [] and o.desired_state is None and o.sarcasm_suspected is False


def test_desired_state_and_attribution_roundtrip():
    s = make_state(
        desired_state=DesiredState(goals=["calm", "solitude"], vad=VAD(valence=0.4, arousal=-0.5, dominance=0.1),
                                   evidence_spans=["I want some quiet"]),
        attributions=[Attribution(experiencer="other", who="my brother", label="nervousness", p=0.7,
                                  time="present", span="My brother is nervous")],
        temporal_focus="mixed", sarcasm_suspected=True,
    )
    again = EmotionalStateObject.model_validate_json(s.model_dump_json())
    assert again == s
    assert again.desired_state.goals == [DesiredGoal.CALM, DesiredGoal.SOLITUDE]
    assert again.attributions[0].who == "my brother"


@pytest.mark.parametrize("bad", [
    lambda: DesiredState(goals=["world_peace"]),
    lambda: Attribution(label="not_an_emotion"),
    lambda: Attribution(label="joy", experiencer="nobody"),
    lambda: Attribution(label="joy", time="yesterday"),
    lambda: Attribution(label="joy", p=1.5),
    lambda: EngineOutput(engine="x", version="0", temporal_focus="past"),
])
def test_v02_invalid_values_rejected(bad):
    with pytest.raises(ValidationError):
        bad()


# ---------------------------------------------------------------- engine interface


def test_missing_modality_gives_empty_output_not_error():
    out = KeystrokeEngine()(EngineInput(text="hello"))
    assert out.confidence == 0 and out.coverage == 0
    assert out.features["skipped"] == "no_keystrokes"


def test_unsupported_language_skips_english_only_engine():
    inp = EngineInput(text="Ես տխուր եմ", context=MessageContext(language="hy"))
    assert LexiconEngine()(inp).features["skipped"] == "unsupported_language:hy"


def test_language_agnostic_engine_reaches_analyze():
    with pytest.raises(NotImplementedError):
        AppraisalEngine()(EngineInput(text="Ես տխուր եմ", context=MessageContext(language="hy")))


# ---------------------------------------------------------------- safety


@pytest.mark.parametrize("text", [
    "I want to kill myself", "honestly I just want to die", "I've been cutting myself again",
    "thinking about suicide a lot", "there is no reason to live", "I don't want to be alive anymore",
])
def test_urgent_phrases(text):
    assert safety.screen(text).level == SafetyLevel.URGENT


@pytest.mark.parametrize("text", ["I feel hopeless", "I can't take it anymore", "I'm a burden to everyone"])
def test_concern_phrases(text):
    assert safety.screen(text).level == SafetyLevel.CONCERN


@pytest.mark.parametrize("text", [
    "This joke killed me", "I'm dying to see that movie", "Had a great day at the beach", "my phone died",
])
def test_no_false_alarm_on_everyday_idioms(text):
    assert safety.screen(text).level == SafetyLevel.NONE


def test_merge_takes_most_severe():
    m = safety.merge(SafetyFlag(), SafetyFlag(level=SafetyLevel.URGENT, reasons=["llm"]), None)
    assert m.level == SafetyLevel.URGENT and m.reasons == ["llm"]
    assert safety.merge().level == SafetyLevel.NONE


# ---------------------------------------------------------------- adapters


def test_travel_low_valence_low_arousal_rest_goes_calm():
    s = make_state(v=-0.6, a=-0.5, needs=[(NeedType.REST, 0.8)])
    out = run_adapter(TravelV0(), s)
    assert out.items[0]["archetype"] == "calm_nature"
    assert not out.safety_override


def test_travel_positive_celebration():
    s = make_state(v=0.7, a=0.5, needs=[(NeedType.CELEBRATION, 0.9)])
    assert run_adapter(TravelV0(), s).items[0]["archetype"] in {"celebration", "adventure"}


def test_urgent_safety_overrides_any_adapter():
    class Exploding(Adapter):
        domain = "x"

        def adapt(self, state, user_profile=None):
            raise AssertionError("adapter must not run when safety is urgent")

    out = run_adapter(Exploding(), make_state(safety_level=SafetyLevel.URGENT))
    assert out.safety_override
    assert any("findahelpline" in i["text"] for i in out.items)
