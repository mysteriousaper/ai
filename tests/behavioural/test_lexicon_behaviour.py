import pytest

from engine.engines.lexicon import LexiconEngine
from engine.schemas import EngineInput

E = LexiconEngine()


def out(text: str):
    return E(EngineInput(text=text))


def v(text: str) -> float:
    return out(text).vad.valence


def test_negation_flips_valence_sign():
    assert v("I am happy") > 0 > v("I am not happy")
    assert v("I am not sad") > v("I am sad")


def test_intensity_monotonic_on_valence():
    # more intense negative emotion = more negative valence
    assert v("I am sad") > v("I am very sad") >= v("I am devastated")


def test_name_and_city_invariance():
    a = out("Anna was angry at me in Paris")
    b = out("Mark was angry at me in Tokyo")
    va, vb = a.vad.model_dump(), b.vad.model_dump()
    assert [*va['std'], va['valence'], va['arousal'], va['dominance']] == pytest.approx(
        [*vb['std'], vb['valence'], vb['arousal'], vb['dominance']], abs=1e-6
    )
    assert a.emotions == pytest.approx(b.emotions, abs=1e-6)
    assert a.confidence == pytest.approx(b.confidence, abs=1e-6)


@pytest.mark.parametrize("variant", [
    "i am extremely sad about the news.",
    "I am extremely sad about the news",
    "I am extremely sad about the nwes.",
])
def test_robustness_to_case_punctuation_typo(variant):
    assert abs(v(variant) - v("I am extremely sad about the news.")) < 0.25


@pytest.mark.xfail(reason="lexicon cannot read a typo inside the emotion word itself (no fuzzy matching)", strict=True)
def test_typo_in_emotion_word():
    assert abs(v("I am sdad") - v("I am sad")) < 0.25
