import pandas as pd

from engine.engines.lexicon import LEXICON_PATH, LexiconEngine, _load
from engine.schemas import EngineInput, MessageContext

E = LexiconEngine()


def run(text: str, lang: str | None = None):
    return E(EngineInput(text=text, context=MessageContext(language=lang)))


def test_parquet_schema():
    _load()
    df = pd.read_parquet(LEXICON_PATH)
    assert {"term", "v", "a", "d", "anger", "trust"} <= set(df.columns)
    assert df["term"].is_unique
    assert df["v"].dropna().between(-1, 1).all()


def test_single_match_and_features():
    out = run("I am happy")
    assert out.engine == "lexicon" and out.version == "0.1.0"
    assert out.vad.valence > 0.5
    assert out.features["n_matched"] == 1 and out.features["matched"][0]["term"] == "happy"
    assert out.emotions["joy"] > 0
    assert all(0 <= p <= 1 for p in out.emotions.values())


def test_multiword_longest_match():
    assert "a bit" in _load() or "give up" in _load()
    out = run("I want to give up")
    terms = [f["term"] for f in out.features["matched"]]
    assert any(" " in t for t in terms)


def test_no_match_is_neutral():
    out = run("zzqx blorp")
    assert out.emotions == {"neutral": 1.0}
    assert out.vad.valence == 0 and out.vad.std == (1.0, 1.0, 1.0)
    assert out.confidence == 0.05


def test_negation_removes_emotion_votes():
    out = run("I am not happy")
    assert out.features["n_negated"] == 1
    assert "joy" not in out.emotions


def test_empty_text_skipped_by_base_class():
    out = run("   ")
    assert out.features["skipped"] == "empty_text" and out.confidence == 0 and out.vad is None


def test_non_english_skipped():
    out = run("Ես երջանիկ եմ", "hy")
    assert out.features["skipped"].startswith("unsupported_language") and out.vad is None


def test_exclamations_raise_arousal():
    assert run("I am happy!!!").vad.arousal > run("I am happy").vad.arousal


def test_contrast_weights_later_clause():
    assert run("It was awful but I am happy").vad.valence > run("I am happy but it was awful").vad.valence
