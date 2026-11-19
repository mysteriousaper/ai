import pytest

from engine.engines import transformer as T
from engine.schemas import EngineInput

pytestmark = [
    pytest.mark.skipif(not T.models_available(), reason="transformer model files missing"),
    pytest.mark.skipif(not T.RIDGE_PATH.exists(), reason="VAD ridge not fitted (scripts/tune_transformer.py)"),
]


@pytest.fixture(scope="module")
def E():
    return T.TransformerEngine()


@pytest.fixture(scope="module")
def out(E):
    return lambda text: E(EngineInput(text=text))


def test_negation_lowers_joy_and_valence(out):
    a, b = out("I am happy"), out("I am not happy")
    assert a.emotions["joy"] > b.emotions["joy"]
    assert a.vad.valence > b.vad.valence


def test_intensity_monotonic_on_negative_valence(out):
    v = [out(t).vad.valence for t in ("I am sad", "I am very sad", "I am devastated")]
    assert v[0] > v[1] > v[2], v


@pytest.mark.xfail(
    reason="genuine: the pretrained model is name/place sensitive (anger prob differs ~0.17, valence ~0.08)",
    strict=True,
)
def test_name_and_city_invariance(out):
    a = out("Anna was angry at me in Paris")
    b = out("Mark was angry at me in Tokyo")
    assert abs(a.vad.valence - b.vad.valence) < 0.05
    assert abs(a.emotions["anger"] - b.emotions["anger"]) < 0.05
    assert abs(a.emotions["annoyance"] - b.emotions["annoyance"]) < 0.05


@pytest.mark.parametrize("variant", [
    "i am extremely sad about the news.",
    "I am extremely sad about the news",
    "I am extremely sad about the nwes.",
])
def test_robustness_to_case_punctuation_typo(out, variant):
    assert abs(out(variant).vad.valence - out("I am extremely sad about the news.").vad.valence) < 0.25
