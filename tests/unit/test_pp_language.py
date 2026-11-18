"""Script mix, code-switching and language identification (brief §4.3, §9; ADR-0005)."""

import pytest

from engine.preprocessing.language import (
    detect_language,
    looks_like_latin_armenian,
    script_mix,
)


class TestScriptMix:
    def test_pure_latin(self) -> None:
        mix = script_mix("Hello there")
        assert mix.latin == 1.0
        assert mix.cyrillic == 0.0
        assert mix.armenian == 0.0
        assert mix.other == 0.0
        assert mix.letter_count == 10

    def test_pure_cyrillic(self) -> None:
        mix = script_mix("Привет, как дела?")
        assert mix.cyrillic == 1.0
        assert mix.latin == 0.0

    def test_pure_armenian(self) -> None:
        mix = script_mix("Բարև, ինչպե՞ս ես")
        assert mix.armenian == 1.0

    def test_mixed_latin_cyrillic(self) -> None:
        mix = script_mix("ok привет")
        assert mix.latin == pytest.approx(2 / 8)
        assert mix.cyrillic == pytest.approx(6 / 8)

    def test_other_script(self) -> None:
        mix = script_mix("你好")
        assert mix.other == 1.0

    def test_no_letters_gives_zero_ratios(self) -> None:
        mix = script_mix("123 !!! 😀")
        assert mix.letter_count == 0
        assert (mix.latin, mix.cyrillic, mix.armenian, mix.other) == (0.0, 0.0, 0.0, 0.0)

    def test_dominant_and_scripts_present(self) -> None:
        mix = script_mix("это мой API")
        assert mix.dominant == "cyrillic"
        assert set(mix.scripts_present()) == {"cyrillic", "latin"}


class TestLatinArmenian:
    def test_greeting_is_detected(self) -> None:
        assert looks_like_latin_armenian("Barev, inchpes es?")

    def test_plain_english_is_not(self) -> None:
        assert not looks_like_latin_armenian("I am two inches taller than Jan.")
        assert not looks_like_latin_armenian("Hello, how are you?")

    def test_empty_is_not(self) -> None:
        assert not looks_like_latin_armenian("")


class TestDetectLanguage:
    def test_english(self) -> None:
        result = detect_language("I am really happy today, thanks for asking.")
        assert result.language == "en"
        assert 0.5 < result.confidence <= 1.0
        assert not result.code_switched

    def test_russian(self) -> None:
        result = detect_language("Привет, как у тебя дела сегодня?")
        assert result.language == "ru"
        assert not result.code_switched

    def test_armenian(self) -> None:
        result = detect_language("Բարև, ինչպե՞ս ես այսօր")
        assert result.language == "hy"
        assert not result.code_switched

    def test_latin_armenian_flagged(self) -> None:
        result = detect_language("Barev, inchpes es?")
        assert result.code_switched
        assert result.latin_armenian
        assert result.language in {"hy", "other"}

    def test_mixed_scripts_flagged(self) -> None:
        result = detect_language("Сегодня был ужасный день, honestly so tired")
        assert result.code_switched

    def test_other_script(self) -> None:
        result = detect_language("今天我很开心")
        assert result.language == "other"

    def test_no_letters_is_unknown(self) -> None:
        for text in ["", "   ", "😀😀", "123 !!!"]:
            result = detect_language(text)
            assert result.language == "unknown"
            assert result.confidence == 0.0

    def test_hint_used_when_detection_unknown(self) -> None:
        result = detect_language("😀😀", language_hint="ru")
        assert result.language == "ru"
        assert result.source == "hint"

    def test_hint_does_not_override_confident_detection(self) -> None:
        result = detect_language("Привет, как у тебя дела сегодня?", language_hint="en")
        assert result.language == "ru"
        assert result.source == "detected"

    def test_invalid_hint_ignored(self) -> None:
        result = detect_language("😀", language_hint="klingon")
        assert result.language == "unknown"
