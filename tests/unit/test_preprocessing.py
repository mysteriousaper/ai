import pytest

from engine.preprocessing.pipeline import detect_language, preprocess


def tok(pp, word):
    return next(t for t in pp.tokens if t.text.lower() == word)


def test_negation_scope():
    assert not tok(preprocess("I am happy"), "happy").negated
    assert tok(preprocess("I am not happy"), "happy").negated
    assert tok(preprocess("I'm not really that happy"), "happy").negated
    assert tok(preprocess("I don't feel good"), "good").negated


def test_negation_stops_at_contrast_and_punctuation():
    pp = preprocess("It was not bad, but the ending was great")
    assert tok(pp, "bad").negated
    assert not tok(pp, "great").negated
    assert tok(pp, "great").after_contrast
    assert not tok(pp, "bad").after_contrast


def test_intensifiers_and_diminishers():
    assert tok(preprocess("I am sad"), "sad").boost == 0
    assert tok(preprocess("I am very sad"), "sad").boost > 0
    assert tok(preprocess("I am kind of sad"), "sad").boost < 0
    assert tok(preprocess("I am really very sad"), "sad").boost > tok(preprocess("I am very sad"), "sad").boost


def test_surface_stats():
    s = preprocess("I am SOOOO HAPPY!!! 😭😭").stats
    assert s["elongations"] == 1
    assert s["repeated_punct"] == 1
    assert s["exclaims"] == 3
    assert s["emoji_count"] == 2
    assert s["caps_words"] == 2
    assert s["caps_ratio"] > 0.5


def test_emoji_and_elongation_normalised():
    pp = preprocess("soooo tired 😴")
    assert "sleeping_face" in pp.normalized
    assert "sooo" not in pp.normalized


@pytest.mark.parametrize(("text", "lang"), [
    ("I feel really tired today and I don't know why", "en"),
    ("Ես շատ հոգնած եմ այսօր", "hy"),
    ("Я сегодня очень устал", "ru"),
    ("ok", "en"),
    ("🙂", "und"),
])
def test_language_detection(text, lang):
    assert detect_language(text) == lang


def test_mixed_script_detected():
    assert detect_language("Сегодня был good day, но устал") == "mixed"


def test_non_english_skips_spacy_but_tokenises():
    pp = preprocess("Ես չեմ ուզում։ Վատ եմ")
    assert pp.language == "hy"
    assert len(pp.tokens) >= 4
    assert len(pp.sentences) == 2


def test_empty_text_does_not_crash():
    pp = preprocess("")
    assert pp.tokens == [] and pp.stats["n_words"] == 0


def test_short_english_not_misdetected():
    assert detect_language("I don't feel good") == "en"
    assert detect_language("Je suis très fatigué aujourd'hui et je ne sais pas pourquoi") == "fr"
