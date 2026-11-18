"""Emoji (via the ``emoji`` package) and ASCII emoticons (small table) (brief §4.3)."""

from __future__ import annotations

import re

import emoji

from engine.preprocessing.lexicons import EMOTICONS
from engine.preprocessing.types import EmojiHit, EmoticonHit

_NOT_ALNUM_BEFORE = r"(?<![^\W_])"
_NOT_ALNUM_AFTER = r"(?![^\W_])"
_SLASHY = {":/", ":-/", ":\\"}


def _emoticon_pattern(emoticon: str) -> str:
    # Allow the last character to repeat (":)))", ":DDD").
    body = re.escape(emoticon) + f"(?:{re.escape(emoticon[-1])})*"
    if emoticon[0].isalnum() or emoticon in _SLASHY:
        body = _NOT_ALNUM_BEFORE + body
    if emoticon[-1].isalnum():
        body = body + _NOT_ALNUM_AFTER
    if emoticon in _SLASHY:
        body = body + r"(?![/\\])"
    return body


# Longest first so ":-)" wins over ":)" and "</3" over "<3".
_ORDERED = sorted(EMOTICONS, key=len, reverse=True)
EMOTICON_RE = re.compile("|".join(f"(?P<e{i}>{_emoticon_pattern(e)})" for i, e in enumerate(_ORDERED)))


def _emoji_name(char: str) -> str:
    data = emoji.EMOJI_DATA.get(char)
    if data is None:
        name = emoji.demojize(char)
        if name == char:
            return "unknown"
    else:
        name = str(data["en"])
    return name.strip(":").split(":")[0] or "unknown"


def find_emojis(text: str) -> tuple[EmojiHit, ...]:
    return tuple(
        EmojiHit(char=hit["emoji"], name=_emoji_name(hit["emoji"]), start=hit["match_start"])
        for hit in emoji.emoji_list(text)
    )


def find_emoticons(text: str) -> tuple[EmoticonHit, ...]:
    hits: list[EmoticonHit] = []
    for match in EMOTICON_RE.finditer(text):
        assert match.lastgroup is not None
        base = _ORDERED[int(match.lastgroup[1:])]
        hits.append(EmoticonHit(text=match.group(), name=EMOTICONS[base], start=match.start()))
    return tuple(hits)


def space_emoticons(text: str) -> str:
    """Put spaces around emoticons glued to neighbouring characters."""

    def pad(match: re.Match[str]) -> str:
        start, end = match.span()
        before = " " if start > 0 and not text[start - 1].isspace() else ""
        after = " " if end < len(text) and not text[end].isspace() else ""
        return f"{before}{match.group()}{after}"

    return EMOTICON_RE.sub(pad, text)
