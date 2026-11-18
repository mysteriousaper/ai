"""Surface style cues: CAPS ratio, elongations, repeated punctuation, counts (brief §4.3)."""

from __future__ import annotations

import itertools
import re

from engine.preprocessing.lexicons import DOUBLE_LETTER_WORDS
from engine.preprocessing.types import Elongation, PunctuationCounts, StyleCues

# Armenian in-word marks (emphasis, exclamation, question: U+055B..U+055E) stay in the word.
_WORD_CHAR = r"[^\W_]|[՛-՞]"
WORD_RE = re.compile(rf"(?:{_WORD_CHAR})+(?:['’\-](?:{_WORD_CHAR})+)*")
_LETTER_WORD_RE = re.compile(r"[^\W\d_]+")
_RUN_RE = re.compile(r"([^\W\d_])\1{2,}", re.IGNORECASE)
_PUNCT_RUN_RE = re.compile(r"[!?！？]+")
_ELLIPSIS_RE = re.compile(r"\.{3,}|…+")
# Beyond this many elongated runs in one word, skip the dictionary search and collapse all.
_MAX_RUNS_SEARCHED = 4


def word_count(text: str) -> int:
    """Number of words (letters/digits, with inner apostrophes or hyphens)."""
    return sum(1 for _ in WORD_RE.finditer(text))


def caps_ratio(text: str) -> float:
    """Letters inside ALL-CAPS words longer than one letter, divided by all letters."""
    total = sum(1 for ch in text if ch.isalpha())
    if total == 0:
        return 0.0
    caps = 0
    for match in WORD_RE.finditer(text):
        word = match.group()
        letters = sum(1 for ch in word if ch.isalpha())
        if letters > 1 and word.isupper():
            caps += letters
    return min(1.0, caps / total)


def _collapse_word(word: str) -> str:
    runs = list(_RUN_RE.finditer(word))
    if not runs:
        return word

    def build(keep_double: tuple[bool, ...]) -> str:
        parts: list[str] = []
        pos = 0
        for run, double in zip(runs, keep_double, strict=True):
            parts.append(word[pos : run.start()])
            parts.append(run.group()[:2] if double else run.group()[0])
            pos = run.end()
        parts.append(word[pos:])
        return "".join(parts)

    if len(runs) <= _MAX_RUNS_SEARCHED:
        # Fewest doubles first, so "soooo" -> "so" beats "soo" when both were listed.
        choices = sorted(itertools.product((False, True), repeat=len(runs)), key=sum)
        for choice in choices:
            if any(choice):
                candidate = build(choice)
                if candidate.lower() in DOUBLE_LETTER_WORDS:
                    return candidate
    return build((False,) * len(runs))


def collapse_elongations(text: str) -> tuple[str, tuple[Elongation, ...]]:
    """Collapse letters repeated 3+ times ("soooo" -> "so", "goooood" -> "good")."""
    found: list[Elongation] = []
    parts: list[str] = []
    pos = 0
    for match in _LETTER_WORD_RE.finditer(text):
        word = match.group()
        collapsed = _collapse_word(word)
        if collapsed != word:
            found.append(
                Elongation(
                    original=word,
                    collapsed=collapsed,
                    extra_chars=len(word) - len(collapsed),
                    start=match.start(),
                )
            )
            parts.append(text[pos : match.start()])
            parts.append(collapsed)
            pos = match.end()
    if not found:
        return text, ()
    parts.append(text[pos:])
    return "".join(parts), tuple(found)


def punctuation_counts(text: str) -> PunctuationCounts:
    excl = q = excl_runs = q_runs = mixed = 0
    for match in _PUNCT_RUN_RE.finditer(text):
        run = match.group()
        n_excl = sum(1 for ch in run if ch in "!！")
        n_q = len(run) - n_excl
        excl += n_excl
        q += n_q
        if n_excl and n_q:
            mixed += 1
        elif n_excl >= 2:
            excl_runs += 1
        elif n_q >= 2:
            q_runs += 1
    return PunctuationCounts(
        exclamation_marks=excl,
        question_marks=q,
        exclamation_runs=excl_runs,
        question_runs=q_runs,
        mixed_runs=mixed,
        ellipsis_runs=sum(1 for _ in _ELLIPSIS_RE.finditer(text)),
    )


def style_cues(text: str, elongations: tuple[Elongation, ...] | None = None) -> StyleCues:
    """All style cues of the original ``text``."""
    if elongations is None:
        _, elongations = collapse_elongations(text)
    return StyleCues(
        caps_ratio=caps_ratio(text),
        char_count=len(text),
        word_count=word_count(text),
        elongations=elongations,
        punctuation=punctuation_counts(text),
    )
