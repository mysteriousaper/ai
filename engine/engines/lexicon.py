"""Engine A: lexicon (NRC VAD v2.1 + EmoLex) with negation, boosters, contrast and arousal cues."""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import ClassVar

import numpy as np

from engine.engines.base import Engine
from engine.preprocessing.pipeline import Preprocessed, Token, preprocess
from engine.schemas import VAD, EngineInput, EngineOutput

LEXICON_PATH = Path(__file__).resolve().parents[2] / "data/processed/lexicon_en.parquet"
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
EMO_TO_CANON = {
    "anger": "anger", "anticipation": "desire", "disgust": "disgust", "fear": "fear",
    "joy": "joy", "sadness": "sadness", "surprise": "surprise", "trust": "approval",
}
NEG_FACTOR = -0.74
NEG_AROUSAL = 0.8
MAX_NGRAM = 3
_FALLBACK_STOP = frozenset(
    ["a", "an", "the", "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "them", "his", "her", "its", "their", "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "have", "has", "had", "of", "to", "in", "on", "at", "by", "for", "with", "from", "and", "or", "but", "if", "as", "that", "this", "these", "those", "so", "than", "not", "no", "nor", "too", "very", "just", "can", "will", "would", "could", "should", "may", "might", "must"]
)


@lru_cache(maxsize=1)
def _load() -> dict[str, tuple[float, float, float, tuple[int, ...]]]:
    import pandas as pd

    if not LEXICON_PATH.exists():
        scripts = str(Path(__file__).resolve().parents[2] / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from build_lexicons import build

        build(LEXICON_PATH)
    df = pd.read_parquet(LEXICON_PATH)
    emo = df[list(EMOTIONS)].to_numpy()
    return {
        t: (v, a, d, tuple(int(x) for x in e))
        for t, v, a, d, e in zip(df["term"], df["v"], df["a"], df["d"], emo, strict=True)
    }


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    try:
        from engine.preprocessing.pipeline import _nlp

        return frozenset(w.lower() for w in _nlp().Defaults.stop_words)
    except Exception:  # noqa: BLE001 — fall back to the small list
        return _FALLBACK_STOP


def _is_punct(t: Token) -> bool:
    return not any(c.isalnum() for c in t.text)


def _weights(tokens: list[Token]) -> list[float]:
    """Contrast weighting: after 'but' 1.5, earlier tokens in the same sentence 0.5."""
    w = [1.0] * len(tokens)
    start = 0
    for end in range(len(tokens) + 1):
        if end == len(tokens) or tokens[end].text in {".", "!", "?"}:
            seg = range(start, end)
            but = next((i for i in seg if tokens[i].text.lower() == "but"), None)
            if but is not None:
                for i in seg:
                    if i < but:
                        w[i] = 0.5
                    elif tokens[i].after_contrast:
                        w[i] = 1.5
            start = end + 1
    return w


class LexiconEngine(Engine):
    name: ClassVar[str] = "lexicon"
    version: ClassVar[str] = "0.1.0"

    def _match(self, toks: list[Token]) -> list[tuple[int, int, tuple]]:
        lex, stop = _load(), _stopwords()
        found: list[tuple[int, int, tuple]] = []  # (start, end_exclusive, entry)
        i = 0
        while i < len(toks):
            hit = None
            if not _is_punct(toks[i]) and toks[i].pos != "PROPN":
                for n in range(min(MAX_NGRAM, len(toks) - i), 0, -1):
                    span = toks[i : i + n]
                    if any(_is_punct(t) for t in span):
                        continue
                    if n == 1 and (span[0].text.lower() in stop and span[0].lemma not in lex):
                        continue
                    if n == 1 and span[0].text.lower() in stop:
                        continue
                    if n > 1 and all(t.text.lower() in stop for t in span):
                        continue
                    for key in (" ".join(t.lemma for t in span), " ".join(t.text.lower() for t in span)):
                        if key in lex:
                            hit = (i, i + n, lex[key], key)
                            break
                    if hit:
                        break
            if hit:
                found.append(hit)
                i = hit[1]
            else:
                i += 1
        return found

    def analyze(self, inp: EngineInput) -> EngineOutput:
        pre: Preprocessed = preprocess(inp.text, inp.context.language)
        toks = pre.tokens
        stop = _stopwords()
        weights = _weights(toks)
        matches = self._match(toks)

        # content units: matched spans count once; other tokens count if not punctuation / stopword
        covered = {j for s, e, *_ in matches for j in range(s, e)}
        n_content = len(matches) + sum(
            1 for j, t in enumerate(toks) if j not in covered and not _is_punct(t) and t.text.lower() not in stop
        )

        rows = []  # (weight, v, a, d) for tokens with VAD
        hits = dict.fromkeys(EMOTIONS, 0)
        n_negated = 0
        feats = []
        for s, e, (v, a, d, emo), key in matches:
            span = toks[s:e]
            neg = any(t.negated for t in span)
            boost = max(-0.9, span[0].boost)
            n_negated += neg
            w = weights[s]
            if not math.isnan(v):
                if neg:
                    v, d, a = v * NEG_FACTOR, d * NEG_FACTOR, a * NEG_AROUSAL
                if boost:
                    v = float(np.sign(v) * abs(v) * (1 + boost))
                    a = a * (1 + boost)
                v, a, d = (float(np.clip(x, -1, 1)) for x in (v, a, d))
                rows.append((w, v, a, d))
                feats.append({"term": key, "v": round(v, 3), "a": round(a, 3), "d": round(d, 3), "negated": neg})
            if not neg:
                for emo_name, flag in zip(EMOTIONS, emo, strict=True):
                    hits[emo_name] += flag

        n_matched = len(matches)
        if not n_matched:
            return EngineOutput(
                engine=self.name, version=self.version,
                vad=VAD(valence=0.0, arousal=0.0, dominance=0.0, std=(1.0, 1.0, 1.0)),
                emotions={"neutral": 1.0}, confidence=0.05, coverage=0.0,
                features={"n_matched": 0, "n_negated": 0, "matched": []},
            )

        if rows:
            arr = np.array(rows)
            w, vals = arr[:, 0], arr[:, 1:]
            mean = (w[:, None] * vals).sum(0) / w.sum()
            var = (w[:, None] * (vals - mean) ** 2).sum(0) / w.sum()
            widen = 0.5 / math.sqrt(len(rows))
            std = np.clip(np.sqrt(np.maximum(np.sqrt(var), 0.05) ** 2 + widen**2), 0.05, 1.0)
            stats = pre.stats
            bump = 0.05 * min(stats["exclaims"] + stats["caps_words"] + stats["elongations"], 6)
            mean[1] = mean[1] + bump
            mean = np.clip(mean, -1, 1)
            vad = VAD(valence=float(mean[0]), arousal=float(mean[1]), dominance=float(mean[2]),
                      std=tuple(float(x) for x in std))
        else:
            vad = VAD(valence=0.0, arousal=0.0, dominance=0.0, std=(1.0, 1.0, 1.0))

        emotions: dict[str, float] = {}
        for emo_name, n in hits.items():
            if n:
                emotions[EMO_TO_CANON[emo_name]] = n / (n + 2)
        emotions["neutral"] = 1.0 if not emotions else float(max(0.0, 1.0 - sum(emotions.values())))

        coverage = min(1.0, n_matched / n_content) if n_content else 0.0
        confidence = min(1.0, n_matched / 4) * (0.5 + 0.5 * coverage)
        top = sorted(feats, key=lambda f: abs(f["v"]), reverse=True)[:5]
        return EngineOutput(
            engine=self.name, version=self.version, vad=vad, emotions=emotions,
            confidence=float(min(1.0, max(0.05, confidence))), coverage=float(coverage),
            features={"matched": top, "n_matched": n_matched, "n_negated": int(n_negated)},
        )
