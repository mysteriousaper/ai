"""Style cues: CAPS ratio, elongations, repeated punctuation, counts (brief §4.3)."""

import pytest

from engine.preprocessing.style import (
    caps_ratio,
    collapse_elongations,
    punctuation_counts,
    style_cues,
    word_count,
)


class TestCapsRatio:
    def test_all_caps(self) -> None:
        assert caps_ratio("SO HAPPY!!!") == 1.0

    def test_no_caps(self) -> None:
        assert caps_ratio("so happy") == 0.0

    def test_partial(self) -> None:
        # "HATE" is 4 of 1+4+4+4 = 13 letters
        assert caps_ratio("I HATE this sooo") == pytest.approx(4 / 13)

    def test_single_letter_words_do_not_count(self) -> None:
        assert caps_ratio("I am A ok") == 0.0

    def test_title_case_is_not_caps(self) -> None:
        assert caps_ratio("Hello World") == 0.0

    def test_cyrillic_caps(self) -> None:
        assert caps_ratio("ОЧЕНЬ плохо") == pytest.approx(5 / 10)

    def test_no_letters(self) -> None:
        assert caps_ratio("123 !!!") == 0.0
        assert caps_ratio("") == 0.0


class TestElongation:
    def test_so_collapses_to_single(self) -> None:
        collapsed, found = collapse_elongations("soooo happy")
        assert collapsed == "so happy"
        assert len(found) == 1
        assert found[0].original == "soooo"
        assert found[0].collapsed == "so"
        assert found[0].extra_chars == 3
        assert found[0].start == 0

    def test_good_keeps_double(self) -> None:
        collapsed, found = collapse_elongations("goooood")
        assert collapsed == "good"
        assert found[0].extra_chars == 3

    def test_case_preserved(self) -> None:
        collapsed, _ = collapse_elongations("NOOOO way")
        assert collapsed == "NO way"

    def test_two_letters_not_elongation(self) -> None:
        collapsed, found = collapse_elongations("book good")
        assert collapsed == "book good"
        assert found == ()

    def test_multiple_runs_in_one_word(self) -> None:
        collapsed, found = collapse_elongations("yesssss pleeeease")
        assert collapsed == "yes please"
        assert [e.collapsed for e in found] == ["yes", "please"]

    def test_cyrillic(self) -> None:
        collapsed, _ = collapse_elongations("оооочень")
        assert collapsed == "очень"

    def test_digits_and_punct_untouched(self) -> None:
        collapsed, found = collapse_elongations("1000 !!!")
        assert collapsed == "1000 !!!"
        assert found == ()


class TestPunctuation:
    def test_runs(self) -> None:
        p = punctuation_counts("What?? No!!! Well... ok?! fine… !")
        assert p.question_runs == 1
        assert p.exclamation_runs == 1
        assert p.mixed_runs == 1
        assert p.ellipsis_runs == 2
        assert p.exclamation_marks == 5
        assert p.question_marks == 3

    def test_single_marks_are_not_runs(self) -> None:
        p = punctuation_counts("Hi! Why? ok.")
        assert (p.exclamation_runs, p.question_runs, p.mixed_runs, p.ellipsis_runs) == (0, 0, 0, 0)

    def test_fullwidth_and_armenian_question(self) -> None:
        p = punctuation_counts("！！")
        assert p.exclamation_runs == 1


class TestCounts:
    def test_word_count(self) -> None:
        assert word_count("I don't know, 2 cats!") == 5
        assert word_count("") == 0
        assert word_count("😀 !!!") == 0

    def test_armenian_word_with_question_mark_is_one_word(self) -> None:
        assert word_count("ինչպե՞ս ես") == 2

    def test_style_cues_bundle(self) -> None:
        s = style_cues("SO HAPPY!!!")
        assert s.caps_ratio == 1.0
        assert s.char_count == 11
        assert s.word_count == 2
        assert s.punctuation.exclamation_runs == 1
        assert s.elongation_count == 0
