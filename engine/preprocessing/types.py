"""Value types produced by preprocessing.

All types are frozen, slotted dataclasses holding tuples, so a ``PreprocessedText`` is
immutable and hashable-safe to share between engines. Offsets are Python string indices
(code points): token, sentence, modifier and contrast offsets refer to
``PreprocessedText.normalized_text``; emoji, emoticon and elongation offsets refer to
``PreprocessedText.text`` (the original).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

LanguageCode = Literal["en", "ru", "hy", "other", "unknown"]
Script = Literal["latin", "cyrillic", "armenian", "other"]
LanguageSource = Literal["detected", "hint", "none"]
ModifierKind = Literal["intensifier", "diminisher"]


@dataclass(frozen=True, slots=True)
class ScriptMix:
    """Share of letters in each script. Ratios are in [0, 1] and sum to 1 (or all 0)."""

    latin: float
    cyrillic: float
    armenian: float
    other: float
    letter_count: int

    @property
    def dominant(self) -> Script | None:
        """Script with the most letters, or None when there are no letters."""
        if self.letter_count == 0:
            return None
        ratios: dict[Script, float] = {
            "latin": self.latin,
            "cyrillic": self.cyrillic,
            "armenian": self.armenian,
            "other": self.other,
        }
        return max(ratios, key=lambda s: ratios[s])

    def scripts_present(self, min_ratio: float = 0.1, min_letters: int = 2) -> tuple[Script, ...]:
        """Scripts that hold at least ``min_ratio`` of letters and ``min_letters`` letters."""
        out: list[Script] = []
        for name, ratio in (
            ("latin", self.latin),
            ("cyrillic", self.cyrillic),
            ("armenian", self.armenian),
            ("other", self.other),
        ):
            if ratio >= min_ratio and round(ratio * self.letter_count) >= min_letters:
                out.append(name)  # type: ignore[arg-type]
        return tuple(out)


@dataclass(frozen=True, slots=True)
class LanguageResult:
    language: LanguageCode
    confidence: float
    script_mix: ScriptMix
    code_switched: bool
    latin_armenian: bool
    source: LanguageSource


@dataclass(frozen=True, slots=True)
class Sentence:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Token:
    """One token of the normalised text.

    ``lemma``, ``pos`` and ``negated`` are None when the language has no analyser yet
    (anything but English): unavailable, not guessed.
    """

    text: str
    start: int
    end: int
    sentence_index: int
    is_punct: bool
    is_word: bool
    lemma: str | None = None
    pos: str | None = None
    negated: bool | None = None
    is_negator: bool = False
    # Product of all intensifier/diminisher multipliers aimed at this token (1.0 = none).
    multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class Modifier:
    """An intensifier or diminisher phrase and the content word it scales."""

    kind: ModifierKind
    phrase: str
    multiplier: float
    token_start: int  # first token index of the phrase
    token_end: int  # one past the last token index of the phrase
    target_index: int | None  # token index of the scaled word, None if nothing follows


@dataclass(frozen=True, slots=True)
class ContrastMarker:
    phrase: str
    token_index: int  # index of the first token of the marker


@dataclass(frozen=True, slots=True)
class EmojiHit:
    char: str
    name: str
    start: int


@dataclass(frozen=True, slots=True)
class EmoticonHit:
    text: str
    name: str
    start: int


@dataclass(frozen=True, slots=True)
class Elongation:
    original: str  # e.g. "soooo"
    collapsed: str  # e.g. "so"
    extra_chars: int  # characters removed by collapsing, e.g. 3
    start: int


@dataclass(frozen=True, slots=True)
class PunctuationCounts:
    exclamation_marks: int
    question_marks: int
    exclamation_runs: int  # runs of 2+ "!" (no "?")
    question_runs: int  # runs of 2+ "?" (no "!")
    mixed_runs: int  # runs mixing "!" and "?", e.g. "?!", "!!?"
    ellipsis_runs: int  # "..." (3+ dots) or "…"


@dataclass(frozen=True, slots=True)
class StyleCues:
    caps_ratio: float
    char_count: int
    word_count: int
    elongations: tuple[Elongation, ...]
    punctuation: PunctuationCounts

    @property
    def elongation_count(self) -> int:
        return len(self.elongations)


@dataclass(frozen=True, slots=True)
class PreprocessedText:
    text: str
    normalized_text: str
    language: LanguageCode
    language_confidence: float
    language_source: LanguageSource
    script_mix: ScriptMix
    code_switched: bool
    latin_armenian: bool
    sentences: tuple[Sentence, ...]
    tokens: tuple[Token, ...]
    modifiers: tuple[Modifier, ...]
    contrast_markers: tuple[ContrastMarker, ...]
    emojis: tuple[EmojiHit, ...]
    emoticons: tuple[EmoticonHit, ...]
    style: StyleCues
    lemma_available: bool
    negation_available: bool
    # Free-form notes for explainability (never contains user text).
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __repr__(self) -> str:  # keep user text out of reprs that may end up in logs
        return (
            f"PreprocessedText(language={self.language!r}, "
            f"confidence={self.language_confidence:.2f}, tokens={len(self.tokens)}, "
            f"chars={self.style.char_count})"
        )
