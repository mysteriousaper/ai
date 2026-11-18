"""Tokenisation plus English negation scope, intensifiers and contrast markers.

English goes through spaCy ``en_core_web_sm`` (loaded once). Every other language gets a
regex tokeniser and sentence splitter; lemma, POS and negation are left as None there
("unavailable") rather than guessed.

Negation scope (VADER-style, but clause-bounded instead of a fixed 3-word window): after a
negator, every token is ``negated`` until scope-ending punctuation, a clause boundary word
("but", "and", "because", ...), or the end of the sentence.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING

import emoji

from engine.preprocessing.emoji_map import EMOTICON_RE
from engine.preprocessing.lexicons import (
    CONTENT_POS,
    CONTRAST_BY_LANGUAGE,
    CONTRAST_EN,
    DIMINISHERS,
    INTENSIFIERS,
    MAX_MODIFIER_LEN,
    NEGATORS,
    SCOPE_BREAKERS,
    STRICT_MODIFIERS,
)
from engine.preprocessing.style import WORD_RE
from engine.preprocessing.types import ContrastMarker, Modifier, Sentence, Token

if TYPE_CHECKING:
    from spacy.language import Language

SPACY_MODEL = "en_core_web_sm"
# Punctuation that closes a negation scope. Quotes, brackets, apostrophes and hyphens do not.
_SCOPE_END_CHARS = frozenset(".,;:!?…—–！？。，；：։՝")
_SENTENCE_END_CHARS = frozenset(".!?…！？。։")
# Phrases that only modify a predicate (adjective, adverb or verb), never a noun.
_PREDICATE_ONLY: frozenset[tuple[str, ...]] = frozenset(
    {("kind", "of"), ("sort", "of"), ("a", "bit"), ("a", "little"), ("a", "little", "bit")}
)
_PREDICATE_POS = frozenset({"ADJ", "ADV", "VERB"})
_DETERMINERS_BEFORE_KIND = frozenset(
    {"what", "a", "an", "the", "this", "that", "these", "those", "some", "any", "every", "which",
     "one", "no", "same", "my", "your", "his", "her", "our", "their", "its"}
)  # fmt: skip
# How many tokens after a modifier phrase we look for its target.
_TARGET_WINDOW = 4
_MODIFIERS: dict[tuple[str, ...], tuple[str, float]] = {
    **{k: ("intensifier", v) for k, v in INTENSIFIERS.items()},
    **{k: ("diminisher", v) for k, v in DIMINISHERS.items()},
}


@dataclass(frozen=True, slots=True)
class Analysis:
    sentences: tuple[Sentence, ...]
    tokens: tuple[Token, ...]
    modifiers: tuple[Modifier, ...]
    contrast_markers: tuple[ContrastMarker, ...]


@lru_cache(maxsize=1)
def load_spacy() -> Language:
    """Load the English pipeline once (NER is not needed)."""
    import spacy

    return spacy.load(SPACY_MODEL, exclude=["ner"])


def _is_punct_text(text: str) -> bool:
    return bool(text) and all(unicodedata.category(ch).startswith("P") for ch in text)


def _ends_scope(token: Token) -> bool:
    return token.is_punct and any(ch in _SCOPE_END_CHARS for ch in token.text)


def _spacy_input(text: str) -> str:
    """Lowercase ALL-CAPS words (same length only) so the tagger sees 'so happy', not 'SO HAPPY'."""

    def lower(match: re.Match[str]) -> str:
        word = match.group()
        if len(word) > 1 and word.isupper():
            low = word.lower()
            if len(low) == len(word):
                return low
        return word

    return WORD_RE.sub(lower, text)


# --- negation ---------------------------------------------------------------------------


def _is_negator(token: Token) -> bool:
    return token.text.lower() in NEGATORS or token.lemma == "not"


def apply_negation(tokens: list[Token]) -> list[Token]:
    out: list[Token] = []
    in_scope = False
    sentence = -1
    for token in tokens:
        if token.sentence_index != sentence:
            sentence, in_scope = token.sentence_index, False
        lower = token.text.lower()
        if _ends_scope(token) or lower in SCOPE_BREAKERS:
            in_scope = False
        negator = _is_negator(token)
        out.append(replace(token, negated=in_scope and not token.is_punct, is_negator=negator))
        if negator:
            in_scope = True
    return out


# --- intensifiers / diminishers ---------------------------------------------------------


def _match_modifier(tokens: list[Token], i: int) -> tuple[tuple[str, ...], str, float] | None:
    lowers = [t.text.lower() for t in tokens[i : i + MAX_MODIFIER_LEN]]
    for n in range(len(lowers), 0, -1):
        key = tuple(lowers[:n])
        if key not in _MODIFIERS:
            continue
        if any(tokens[i + k].sentence_index != tokens[i].sentence_index for k in range(n)):
            continue
        kind, mult = _MODIFIERS[key]
        nxt = tokens[i + n] if i + n < len(tokens) else None
        if n == 1 and key[0] in STRICT_MODIFIERS:
            if nxt is None or nxt.pos not in {"ADJ", "ADV"}:
                continue
        if key in {("kind", "of"), ("sort", "of")} and i > 0:
            if tokens[i - 1].text.lower() in _DETERMINERS_BEFORE_KIND:
                continue
        return key, kind, mult
    return None


def apply_modifiers(tokens: list[Token]) -> tuple[list[Token], tuple[Modifier, ...]]:
    spans: list[tuple[int, int, tuple[str, ...], str, float]] = []
    i = 0
    while i < len(tokens):
        found = _match_modifier(tokens, i)
        if found is None:
            i += 1
            continue
        key, kind, mult = found
        spans.append((i, i + len(key), key, kind, mult))
        i += len(key)

    in_span = {k for start, end, *_ in spans for k in range(start, end)}
    multipliers = [1.0] * len(tokens)
    modifiers: list[Modifier] = []
    for start, end, key, kind, mult in spans:
        allowed = _PREDICATE_POS if key in _PREDICATE_ONLY else CONTENT_POS
        target: int | None = None
        for j in range(end, min(len(tokens), end + _TARGET_WINDOW)):
            cand = tokens[j]
            if cand.sentence_index != tokens[start].sentence_index or _ends_scope(cand):
                break
            if j in in_span or cand.is_negator or cand.pos not in CONTENT_POS:
                continue
            if cand.pos in allowed:
                target = j
            break
        if target is None and key in _PREDICATE_ONLY:
            continue  # "a bit of cake", "a little dog": not a diminisher
        if target is not None:
            multipliers[target] *= mult
        modifiers.append(
            Modifier(
                kind=kind,  # type: ignore[arg-type]
                phrase=" ".join(key),
                multiplier=mult,
                token_start=start,
                token_end=end,
                target_index=target,
            )
        )
    out = [replace(t, multiplier=m) if m != 1.0 else t for t, m in zip(tokens, multipliers, strict=True)]
    return out, tuple(modifiers)


# --- contrast -----------------------------------------------------------------------------


def find_contrast(tokens: list[Token], language: str) -> tuple[ContrastMarker, ...]:
    markers = CONTRAST_BY_LANGUAGE.get(language, frozenset())
    out: list[ContrastMarker] = []
    skip: int | None = None
    for i, token in enumerate(tokens):
        if i == skip:
            continue
        lower = token.text.lower()
        if language == "en" and lower == "even" and i + 1 < len(tokens):
            if tokens[i + 1].text.lower() == "though":
                out.append(ContrastMarker(phrase="even though", token_index=i))
                skip = i + 1
                continue
        if lower not in markers:
            continue
        if language == "en" and lower == "yet" and token.pos != "CCONJ":
            continue  # "not yet", "yet another"
        out.append(ContrastMarker(phrase=lower, token_index=i))
    return tuple(out)


# --- English (spaCy) ----------------------------------------------------------------------


def analyze_english(text: str) -> Analysis:
    doc = load_spacy()(_spacy_input(text))
    sentences: list[Sentence] = []
    tokens: list[Token] = []
    for s_idx, sent in enumerate(doc.sents):
        sentences.append(
            Sentence(text=text[sent.start_char : sent.end_char], start=sent.start_char, end=sent.end_char)
        )
        for tok in sent:
            if tok.is_space:
                continue
            start, end = tok.idx, tok.idx + len(tok.text)
            surface = text[start:end]
            tokens.append(
                Token(
                    text=surface,
                    start=start,
                    end=end,
                    sentence_index=s_idx,
                    is_punct=tok.is_punct or _is_punct_text(surface),
                    is_word=any(ch.isalnum() for ch in surface),
                    lemma=tok.lemma_,
                    pos=tok.pos_,
                    negated=False,
                )
            )
    tokens = apply_negation(tokens)
    tokens, modifiers = apply_modifiers(tokens)
    contrast = find_contrast(tokens, "en")
    return Analysis(tuple(sentences), tuple(tokens), modifiers, contrast)


# --- other languages (regex) --------------------------------------------------------------

_PUNCT_RUN_RE = re.compile(r"([^\w\s])\1*")


def _regex_spans(text: str) -> list[tuple[int, int]]:
    emoji_spans = {hit["match_start"]: hit["match_end"] for hit in emoji.emoji_list(text)}
    spans: list[tuple[int, int]] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isspace():
            pos += 1
            continue
        end = emoji_spans.get(pos)
        if end is None:
            for pattern in (EMOTICON_RE, WORD_RE, _PUNCT_RUN_RE):
                match = pattern.match(text, pos)
                if match and match.end() > pos:
                    end = match.end()
                    break
        if end is None or end <= pos:
            end = pos + 1
        spans.append((pos, end))
        pos = end
    return spans


def analyze_regex(text: str, language: str) -> Analysis:
    tokens: list[Token] = []
    sentences: list[Sentence] = []
    s_idx = 0
    sent_start: int | None = None
    sent_end = 0
    for start, end in _regex_spans(text):
        surface = text[start:end]
        if sent_start is None:
            sent_start = start
        is_punct = _is_punct_text(surface)
        tokens.append(
            Token(
                text=surface,
                start=start,
                end=end,
                sentence_index=s_idx,
                is_punct=is_punct,
                is_word=any(ch.isalnum() for ch in surface),
            )
        )
        sent_end = end
        if is_punct and any(ch in _SENTENCE_END_CHARS for ch in surface):
            sentences.append(Sentence(text[sent_start:end], sent_start, end))
            s_idx += 1
            sent_start = None
    if sent_start is not None:
        sentences.append(Sentence(text[sent_start:sent_end], sent_start, sent_end))
    return Analysis(tuple(sentences), tuple(tokens), (), find_contrast(tokens, language))
