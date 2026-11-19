import time

import pytest

from engine.engines import appraisal as A
from engine.mappings.labels import EventType
from engine.schemas import DesiredGoal, EngineInput, SafetyLevel

pytestmark = pytest.mark.skipif(not A.models_available(), reason="NLI model missing")


@pytest.fixture(scope="module")
def eng():
    return A.AppraisalEngine()


def _top(o, k=2):
    return [e for e, _ in o.features["event_ranking"][:k]]


def test_event_type_valid_and_needs(eng):
    o = eng(EngineInput(text="I got the job today!"))
    assert o.situation is not None and EventType(o.situation.event_type)
    assert len(o.needs) == 7 and o.coverage == 1.0
    assert 0 <= o.confidence <= 1 and o.situation.evidence_spans


def test_failed_test(eng):
    o = eng(EngineInput(text="I failed my driving test again"))
    assert set(_top(o)) & {"loss_or_failure", "own_mistake_or_wrongdoing"}


def test_boss_yelled(eng):
    o = eng(EngineInput(text="My boss yelled at me in front of everyone"))
    assert "conflict_or_mistreatment" in _top(o)
    assert o.situation.agent.value == "other"


def test_desired_calm(eng):
    o = eng(EngineInput(text="I feel anxious but want to feel calm"))
    assert o.desired_state is not None and DesiredGoal.CALM in o.desired_state.goals


def test_sarcasm(eng):
    assert eng(EngineInput(text="Fantastic, another delayed flight.")).sarcasm_suspected


def test_safety_urgent(eng):
    o = eng(EngineInput(text="I want to kill myself"))
    assert o.safety is not None and o.safety.level == SafetyLevel.URGENT


def test_armenian_runs(eng):
    o = eng(EngineInput(text="Ես կորցրեցի աշխատանքս"))
    assert o.situation is not None and o.features["fallback_used"] is False


def test_claude_stub_raises():
    with pytest.raises(NotImplementedError):
        A._claude_appraisal("x")


def test_latency(eng):
    msgs = ["I failed my driving test again", "My boss yelled at me", "Excited for the trip tomorrow!"] * 7
    eng(EngineInput(text=msgs[0]))
    t = time.perf_counter()
    for m in msgs[:20]:
        eng(EngineInput(text=m))
    print(f"appraisal mean latency: {(time.perf_counter() - t) / 20 * 1000:.0f} ms")
