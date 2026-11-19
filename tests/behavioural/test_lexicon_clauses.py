from engine.engines.lexicon import LexiconEngine
from engine.schemas import EngineInput

E = LexiconEngine()


def v(text: str) -> float:
    return E(EngineInput(text=text)).vad.valence


def test_past_state_does_not_describe_current_self():
    assert v("I was happy last week, but now I'm anxious.") < 0


def test_other_persons_feeling_is_discounted():
    assert v("My brother is devastated, but I am happy.") > 0


def test_desire_is_not_a_felt_state():
    assert v("I feel sad but want to feel joyful.") < 0
