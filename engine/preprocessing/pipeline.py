"""Preprocessing shared by all text engines (brief §4.3).

Language ID → normalisation (emoji, elongation) → spaCy parse (en only) → per-token negation
scope and intensifier boost → surface stats (CAPS, repeated punctuation, elongation...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import emoji
from vaderSentiment.vaderSentiment import BOOSTER_DICT, NEGATE

NEGATORS = {w.replace("'", "") for w in NEGATE} | {"n't", "no", "never", "nothing", "nobody", "none", "without", "cannot"}
SCOPE_BREAK = {"but", "however", "although", "though", "yet", ".", ",", ";", ":", "!", "?"}
NEG_WINDOW = 4
CONTRAST = {"but", "however", "although", "though", "yet"}

_ELONG = re.compile(r"(\w)\1{2,}")
_REPEAT_PUNCT = re.compile(r"([!?.])\1+")
_ARMENIAN = re.compile(r"[԰-֏]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
_LATIN = re.compile(r"[A-Za-z]")


@dataclass
class Token:
    text: str
    lemma: str
    pos: str = ""
    negated: bool = False
    boost: float = 0.0  # VADER booster sum from up to 2 preceding modifiers (+ intensifier, - diminisher)
    after_contrast: bool = False  # token follows "but"/"however" in its sentence


@dataclass
class Preprocessed:
    text: str
    normalized: str
    language: str
    sentences: list[str]
    tokens: list[Token]
    stats: dict[str, float] = field(default_factory=dict)


def detect_language(text: str) -> str:
    """Script first (reliable for hy/ru), langdetect only for Latin text. Mixed scripts → 'mixed'."""
    hy, ru, lat = (len(r.findall(text)) for r in (_ARMENIAN, _CYRILLIC, _LATIN))
    total = hy + ru + lat
    if total == 0:
        return "und"
    shares = {"hy": hy / total, "ru": ru / total, "latin": lat / total}
    top, share = max(shares.items(), key=lambda kv: kv[1])
    if share < 0.8:
        return "mixed"
    if top != "latin":
        return top
    if len(text.split()) < 4:
        return "en"  # langdetect is unreliable on very short text; English-first MVP
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    try:
        best = detect_langs(text)[0]
    except Exception:  # noqa: BLE001 — langdetect raises a bare LangDetectException on odd input
        return "en"
    # English-first: only trust another Latin-script language when langdetect is very sure.
    return best.lang if best.prob >= 0.95 else "en"


def normalize(text: str) -> str:
    t = emoji.demojize(text, delimiters=(" :", ": "))
    t = _ELONG.sub(r"\1\1", t)  # "soooo" -> "soo" (keeps a trace; lexicons match after lemmatising)
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["ner"])


def _boosts(words: list[str]) -> list[float]:
    """Per token: sum of booster values of up to 2 preceding modifiers (handles 'kind of', 'sort of')."""
    lw = [w.lower() for w in words]
    out = [0.0] * len(words)
    for i in range(len(words)):
        total, j = 0.0, i - 1
        for _ in range(2):
            if j >= 1 and f"{lw[j - 1]} {lw[j]}" in BOOSTER_DICT:
                total += BOOSTER_DICT[f"{lw[j - 1]} {lw[j]}"]
                j -= 2
            elif j >= 0 and lw[j] in BOOSTER_DICT:
                total += BOOSTER_DICT[lw[j]]
                j -= 1
            else:
                break
        out[i] = total
    return out


def _mark_scopes(tokens: list[Token]) -> None:
    remaining, contrast = 0, False
    for t in tokens:
        low = t.text.lower()
        if low in SCOPE_BREAK:
            remaining = 0
            if low in CONTRAST:
                contrast = True
            if low in {".", "!", "?"}:
                contrast = False
            continue
        t.after_contrast = contrast
        if low.replace("'", "") in NEGATORS or low == "n't":
            remaining = NEG_WINDOW
            continue
        if remaining:
            t.negated = True
            remaining -= 1


def preprocess(text: str, language: str | None = None) -> Preprocessed:
    lang = language or detect_language(text)
    norm = normalize(text)
    if lang == "en":
        doc = _nlp()(norm)
        sentences = [s.text for s in doc.sents]
        tokens = [Token(t.text, t.lemma_.lower(), t.pos_) for t in doc if not t.is_space]
    else:
        sentences = [s for s in re.split(r"(?<=[.!?։])\s+", norm) if s]
        tokens = [Token(w, w.lower()) for w in re.findall(r"\w+(?:'\w+)?|[^\w\s]", norm)]
    for t, b in zip(tokens, _boosts([t.text for t in tokens]), strict=True):
        t.boost = b
    _mark_scopes(tokens)

    letters = [c for c in text if c.isalpha()]
    words = re.findall(r"\b\w+\b", text)
    stats = {
        "n_chars": float(len(text)),
        "n_words": float(len(words)),
        "caps_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "caps_words": float(sum(1 for w in words if len(w) > 1 and w.isupper())),
        "elongations": float(len(_ELONG.findall(text))),
        "exclaims": float(text.count("!")),
        "questions": float(text.count("?")),
        "repeated_punct": float(len(_REPEAT_PUNCT.findall(text))),
        "ellipses": float(text.count("...") + text.count("…")),
        "emoji_count": float(emoji.emoji_count(text)),
        "negations": float(sum(1 for t in tokens if t.text.lower().replace("'", "") in NEGATORS or t.text.lower() == "n't")),
    }
    return Preprocessed(text, norm, lang, sentences, tokens, stats)
