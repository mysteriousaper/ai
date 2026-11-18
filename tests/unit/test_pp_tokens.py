"""Tokenisation, negation scope, intensifiers/diminishers, contrast markers (brief §4.3, §4.4)."""

import pytest

from engine.preprocessing.tokens import Analysis, analyze_english, analyze_regex


def _tok(a: Analysis, text: str, nth: int = 0):
    matches = [t for t in a.tokens if t.text.lower() == text.lower()]
    assert len(matches) > nth, f"token {text!r} not found"
    return matches[nth]


def _negated_words(a: Analysis) -> list[str]:
    return [t.text for t in a.tokens if t.negated]


class TestEnglishTokens:
    def test_lemma_and_pos(self) -> None:
        a = analyze_english("The cats were running.")
        cats = _tok(a, "cats")
        assert cats.lemma == "cat"
        assert cats.pos == "NOUN"
        assert _tok(a, "running").lemma == "run"
        assert _tok(a, ".").is_punct

    def test_offsets_slice_the_text(self) -> None:
        text = "I can't   believe it :) wow"
        a = analyze_english(text)
        for t in a.tokens:
            assert text[t.start : t.end] == t.text

    def test_sentences(self) -> None:
        a = analyze_english("I am fine. You are not! Are we?")
        assert [s.text for s in a.sentences] == ["I am fine.", "You are not!", "Are we?"]
        assert [t.sentence_index for t in a.tokens if t.text == "are"] == [1]

    def test_all_caps_words_are_tagged_like_lowercase(self) -> None:
        a = analyze_english("I AM SO HAPPY")
        assert _tok(a, "HAPPY").text == "HAPPY"
        assert _tok(a, "HAPPY").pos == "ADJ"
        assert _tok(a, "HAPPY").lemma == "happy"


class TestNegation:
    def test_not_happy(self) -> None:
        a = analyze_english("I am not happy")
        assert _negated_words(a) == ["happy"]
        assert _tok(a, "not").is_negator
        assert not _tok(a, "not").negated

    def test_no_negation(self) -> None:
        a = analyze_english("I am happy")
        assert _negated_words(a) == []

    def test_contraction(self) -> None:
        a = analyze_english("I don't like it")
        assert _negated_words(a) == ["like", "it"]

    def test_unapostrophised_contraction(self) -> None:
        a = analyze_english("i dont like it")
        assert "like" in _negated_words(a)

    def test_but_resets_scope(self) -> None:
        a = analyze_english("not bad but terrible")
        assert _tok(a, "bad").negated
        assert not _tok(a, "terrible").negated
        assert not _tok(a, "but").negated

    def test_punctuation_ends_scope(self) -> None:
        a = analyze_english("Not great, happy now")
        assert _tok(a, "great").negated
        assert not _tok(a, "happy").negated

    def test_sentence_boundary_ends_scope(self) -> None:
        a = analyze_english("I am not sad. I am happy")
        assert not _tok(a, "happy").negated

    @pytest.mark.parametrize(
        ("text", "word"),
        [
            ("never happy", "happy"),
            ("no fun at all", "fun"),
            ("nobody cares", "cares"),
            ("nothing works", "works"),
            ("without hope", "hope"),
            ("I hardly slept", "slept"),
            ("I cannot sleep", "sleep"),
        ],
    )
    def test_negators(self, text: str, word: str) -> None:
        assert _tok(analyze_english(text), word).negated

    def test_quotes_do_not_end_scope(self) -> None:
        a = analyze_english('not "happy" at all')
        assert _tok(a, "happy").negated

    def test_negation_available_flag_on_every_token(self) -> None:
        a = analyze_english("fine, thanks")
        assert all(t.negated is not None for t in a.tokens)


class TestModifiers:
    def test_very_sad(self) -> None:
        a = analyze_english("I am very sad")
        sad = _tok(a, "sad")
        assert sad.multiplier > 1.0
        (mod,) = a.modifiers
        assert mod.kind == "intensifier"
        assert mod.phrase == "very"
        assert a.tokens[mod.target_index].text == "sad"

    def test_kind_of_diminishes(self) -> None:
        a = analyze_english("I feel kind of tired")
        assert _tok(a, "tired").multiplier < 1.0
        assert a.modifiers[0].phrase == "kind of"
        assert a.modifiers[0].kind == "diminisher"

    def test_a_little_bit(self) -> None:
        a = analyze_english("a little bit worried")
        assert [m.phrase for m in a.modifiers] == ["a little bit"]
        assert _tok(a, "worried").multiplier < 1.0

    def test_stacked_intensifiers_multiply(self) -> None:
        a = analyze_english("very very happy")
        assert _tok(a, "happy").multiplier == pytest.approx(1.3 * 1.3)

    def test_so_as_conjunction_is_not_intensifier(self) -> None:
        a = analyze_english("I was tired so I left")
        assert a.modifiers == ()

    def test_so_happy(self) -> None:
        a = analyze_english("SO HAPPY")
        assert _tok(a, "HAPPY").multiplier > 1.0

    def test_what_kind_of_is_not_diminisher(self) -> None:
        a = analyze_english("What kind of person does that?")
        assert a.modifiers == ()

    def test_intensifier_passes_over_negation(self) -> None:
        a = analyze_english("I really don't like it")
        like = _tok(a, "like")
        assert like.multiplier > 1.0
        assert like.negated

    def test_no_target(self) -> None:
        a = analyze_english("really?")
        assert all(t.multiplier == 1.0 for t in a.tokens)
        assert all(m.target_index is None for m in a.modifiers)

    def test_slightly(self) -> None:
        assert _tok(analyze_english("slightly annoyed"), "annoyed").multiplier == pytest.approx(0.6)


class TestContrast:
    def test_but_however_although(self) -> None:
        a = analyze_english("Good but slow. However, fine although late")
        assert [(c.phrase, a.tokens[c.token_index].text) for c in a.contrast_markers] == [
            ("but", "but"),
            ("however", "However"),
            ("although", "although"),
        ]

    def test_even_though(self) -> None:
        a = analyze_english("I went even though I was tired")
        assert [c.phrase for c in a.contrast_markers] == ["even though"]

    def test_not_yet_is_not_contrast(self) -> None:
        a = analyze_english("not yet")
        assert a.contrast_markers == ()


class TestRegexTokens:
    def test_russian(self) -> None:
        text = "Мне грустно, но я держусь. А ты?"
        a = analyze_regex(text, "ru")
        words = [t.text for t in a.tokens if t.is_word]
        assert words == ["Мне", "грустно", "но", "я", "держусь", "А", "ты"]
        assert all(t.lemma is None and t.pos is None and t.negated is None for t in a.tokens)
        assert [c.phrase for c in a.contrast_markers] == ["но"]
        assert len(a.sentences) == 2
        assert a.modifiers == ()

    def test_armenian(self) -> None:
        text = "Բարև, ինչպե՞ս ես։ Լավ եմ, բայց հոգնած"
        a = analyze_regex(text, "hy")
        words = [t.text for t in a.tokens if t.is_word]
        assert "ինչպե՞ս" in words
        assert [c.phrase for c in a.contrast_markers] == ["բայց"]
        assert len(a.sentences) == 2

    def test_offsets_and_emoji(self) -> None:
        text = "ок 👍🏽 :) !!!"
        a = analyze_regex(text, "ru")
        assert [t.text for t in a.tokens] == ["ок", "👍🏽", ":)", "!!!"]
        for t in a.tokens:
            assert text[t.start : t.end] == t.text

    def test_empty(self) -> None:
        a = analyze_regex("", "other")
        assert a.tokens == () and a.sentences == ()
