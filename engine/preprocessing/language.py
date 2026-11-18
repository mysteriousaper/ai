"""Script mix, code-switching and language identification (brief §4.3, §9; ADR-0005).

Lingua is restricted to English, Russian and Armenian. Because lingua filters candidates by
alphabet first, any Latin-script text comes back as English with confidence about 1.0, so
two extra rules sit around it:

* text whose letters are mostly in another script (CJK, Arabic, Greek, ...) is ``other``;
* Latin-script text with transliterated Armenian marker words is ``hy`` and
  ``code_switched`` (it must go to the multilingual model, brief §9).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import TYPE_CHECKING

from engine.preprocessing.lexicons import LATIN_ARMENIAN_STRONG, LATIN_ARMENIAN_WEAK
from engine.preprocessing.types import LanguageCode, LanguageResult, Script, ScriptMix

if TYPE_CHECKING:
    from lingua import LanguageDetector

SUPPORTED_HINTS: frozenset[str] = frozenset({"en", "ru", "hy"})
# Below this detection confidence a valid language hint wins.
HINT_THRESHOLD = 0.5

_WORD_RE = re.compile(r"[^\W\d_]+")


def _script_of(ch: str) -> Script:
    cp = ord(ch)
    if cp < 0x0250 or 0x1E00 <= cp <= 0x1EFF or 0xFF21 <= cp <= 0xFF5A:
        return "latin"
    if 0x0400 <= cp <= 0x052F or 0x1C80 <= cp <= 0x1C8F or 0x2DE0 <= cp <= 0x2DFF or (
        0xA640 <= cp <= 0xA69F
    ):
        return "cyrillic"
    if 0x0530 <= cp <= 0x058F or 0xFB13 <= cp <= 0xFB17:
        return "armenian"
    # Latin letters outside the main blocks (e.g. IPA, Latin Extended-C/D/E).
    name = unicodedata.name(ch, "")
    if name.startswith("LATIN "):
        return "latin"
    return "other"


def script_mix(text: str) -> ScriptMix:
    """Ratio of alphabetic characters per script."""
    counts: dict[Script, int] = {"latin": 0, "cyrillic": 0, "armenian": 0, "other": 0}
    for ch in text:
        if ch.isalpha():
            counts[_script_of(ch)] += 1
    total = sum(counts.values())
    if total == 0:
        return ScriptMix(0.0, 0.0, 0.0, 0.0, 0)
    return ScriptMix(
        latin=counts["latin"] / total,
        cyrillic=counts["cyrillic"] / total,
        armenian=counts["armenian"] / total,
        other=counts["other"] / total,
        letter_count=total,
    )


def looks_like_latin_armenian(text: str) -> bool:
    """Heuristic: does Latin-script text contain transliterated Armenian?

    Needs two strong marker words, or one strong marker with markers making up at least a
    quarter of the words. Weak markers are words that are also common in English or other
    Latin languages ("es", "lav", "inch", "jan").
    """
    words = [w.lower() for w in _WORD_RE.findall(text) if _script_of(w[0]) == "latin"]
    if not words:
        return False
    strong = sum(1 for w in words if w in LATIN_ARMENIAN_STRONG)
    weak = sum(1 for w in words if w in LATIN_ARMENIAN_WEAK)
    if strong >= 2:
        return True
    return strong >= 1 and (strong + weak) / len(words) >= 0.25


@lru_cache(maxsize=1)
def _detector() -> LanguageDetector:
    from lingua import Language, LanguageDetectorBuilder

    return (
        LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.RUSSIAN, Language.ARMENIAN)
        .with_preloaded_language_models()
        .build()
    )


def _lingua(text: str) -> tuple[LanguageCode, float]:
    from lingua import Language

    values = _detector().compute_language_confidence_values(text)
    if not values:
        return "unknown", 0.0
    top = values[0]
    code: dict[Language, LanguageCode] = {
        Language.ENGLISH: "en",
        Language.RUSSIAN: "ru",
        Language.ARMENIAN: "hy",
    }
    return code.get(top.language, "other"), float(top.value)


def detect_language(text: str, language_hint: str | None = None) -> LanguageResult:
    """Identify the language of ``text``, restricted to en/ru/hy (else other/unknown)."""
    mix = script_mix(text)
    latin_hy = mix.latin > 0.5 and looks_like_latin_armenian(text)
    code_switched = latin_hy or len(mix.scripts_present()) >= 2

    language: LanguageCode
    if mix.letter_count == 0:
        language, confidence = "unknown", 0.0
    elif mix.other > 0.5:
        language, confidence = "other", mix.other
    elif latin_hy:
        # Lingua cannot tell transliterated Armenian from English; the marker rule decides.
        language, confidence = "hy", 0.6
    else:
        language, confidence = _lingua(text)

    source = "detected" if language != "unknown" else "none"
    hint = language_hint.lower().strip() if language_hint else None
    if hint in SUPPORTED_HINTS and (language == "unknown" or confidence < HINT_THRESHOLD):
        language, source = hint, "hint"  # type: ignore[assignment]

    return LanguageResult(
        language=language,
        confidence=min(1.0, max(0.0, confidence)),
        script_mix=mix,
        code_switched=code_switched,
        latin_armenian=latin_hy,
        source=source,  # type: ignore[arg-type]
    )
