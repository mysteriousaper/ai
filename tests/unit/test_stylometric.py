import pytest

from engine.engines.stylometric import MODEL_PATH, StylometricEngine, extract_features
from engine.schemas import EngineInput


def test_absolutist_and_first_person():
    f = extract_features("I always fail, nothing ever works")
    assert f["absolutist_rate"] > 0 and f["first_person_singular_rate"] > 0


def test_plural_second_person():
    assert extract_features("We love our team")["first_person_plural_rate"] > 0
    assert extract_features("You did your best")["second_person_rate"] > 0


def test_tense():
    assert extract_features("I walked home yesterday")["past_tense_ratio"] > 0
    assert extract_features("I will go tomorrow")["future_ratio"] > 0
    assert extract_features("I am going to leave")["future_ratio"] > 0


def test_hedge_certainty_question():
    f = extract_features("Maybe I think it might work?")
    assert f["hedge_rate"] > 0 and f["question_rate"] > 0
    assert extract_features("I am definitely sure")["certainty_rate"] > 0


def test_length_ttr_and_surface_stats():
    f = extract_features("NO!!! Sooo bad... not good :(")
    assert f["n_words"] > 0 and 0 < f["type_token_ratio"] <= 1
    assert f["exclaims"] >= 3 and f["repeated_punct"] > 0 and f["elongations"] > 0 and f["negations"] > 0
    assert f["ellipses"] > 0 and f["caps_ratio"] > 0


def test_empath_category():
    assert extract_features("I am so sad and miserable and I cry")["empath_sadness"] > 0


@pytest.mark.parametrize("text", ["", "   ", "🙂", "!!!"])
def test_no_crash(text):
    f = extract_features(text)
    assert all(isinstance(v, float) for v in f.values())
    out = StylometricEngine().analyze(EngineInput(text=text))
    assert out.engine == "stylometric" and out.emotions == {}


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="model not trained")
def test_vad_in_range():
    out = StylometricEngine().analyze(EngineInput(text="I always fail and nothing ever works for me."))
    assert out.vad is not None
    assert all(-1 <= x <= 1 for x in (out.vad.valence, out.vad.arousal, out.vad.dominance))
    assert 0 < out.confidence <= 0.2
