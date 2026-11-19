"""Engine C. Pronouns, absolutist words, tense, hedges, punctuation, Empath categories.

Output = raw features (explainability) + a weak VAD from a small ridge model trained on the
EmoBank train split (scripts/train_stylometric.py) + deliberately low confidence.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.engines.base import Engine
from engine.mappings.scales import emobank  # noqa: F401  (documents the target scale used in training)
from engine.preprocessing.pipeline import _nlp, preprocess
from engine.schemas import VAD, EngineInput, EngineOutput

MODEL_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "stylometric_ridge.joblib"

FIRST_SING = {"i", "me", "my", "mine", "myself"}
FIRST_PLUR = {"we", "us", "our", "ours", "ourselves"}
SECOND = {"you", "your", "yours", "yourself", "yourselves"}
ABSOLUTIST = {
    "absolutely", "all", "always", "complete", "completely", "constant", "constantly", "definitely",
    "entire", "ever", "every", "everyone", "everything", "full", "must", "never", "nothing", "totally", "whole",
}  # fmt: skip
HEDGE_WORDS = {"maybe", "perhaps", "possibly", "probably", "might", "seems"}
HEDGE_PHRASES = ("kind of", "sort of", "i guess", "i think")
CERTAINTY = {"definitely", "certainly", "sure", "clearly", "obviously", "know"}
FUTURE_WORDS = {"will", "shall", "gonna"}
EMPATH_CATEGORIES = [
    "negative_emotion", "positive_emotion", "sadness", "anger", "fear", "joy", "love",
    "nervousness", "pain", "shame", "achievement", "family",
]  # fmt: skip
STAT_KEYS = [
    "caps_ratio", "exclaims", "elongations", "emoji_count", "repeated_punct", "ellipses", "negations",
]  # fmt: skip
FEATURE_NAMES = [
    "first_person_singular_rate", "first_person_plural_rate", "second_person_rate", "absolutist_rate",
    "past_tense_ratio", "future_ratio", "hedge_rate", "certainty_rate", "question_rate", "n_words",
    "type_token_ratio", *STAT_KEYS, *(f"empath_{c}" for c in EMPATH_CATEGORIES),
]  # fmt: skip
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


@lru_cache(maxsize=1)
def _empath():
    from empath import Empath

    return Empath()


def _zeros() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


def extract_features(text: str) -> dict[str, float]:
    feats = _zeros()
    if not text or not text.strip():
        return feats
    pre = preprocess(text, language="en")
    feats.update({k: float(pre.stats.get(k, 0.0)) for k in STAT_KEYS})
    doc = _nlp()(pre.normalized)
    words = [t.text.lower() for t in doc if t.is_alpha]
    n = len(words)
    feats["n_words"] = math.log1p(n)
    n_q = pre.stats.get("questions", 0.0)
    n_sent = max(1, len(pre.sentences))
    feats["question_rate"] = n_q / n_sent
    if n == 0:
        return feats
    cnt = lambda vocab: sum(w in vocab for w in words) / n
    feats["first_person_singular_rate"] = cnt(FIRST_SING)
    feats["first_person_plural_rate"] = cnt(FIRST_PLUR)
    feats["second_person_rate"] = cnt(SECOND)
    feats["absolutist_rate"] = cnt(ABSOLUTIST)
    feats["certainty_rate"] = cnt(CERTAINTY)
    joined = " ".join(_WORD.findall(pre.normalized.lower()))
    hedges = sum(w in HEDGE_WORDS for w in words) + sum(joined.count(p) for p in HEDGE_PHRASES)
    feats["hedge_rate"] = hedges / n
    feats["type_token_ratio"] = len(set(words)) / n
    verbs = [t for t in doc if t.pos_ in {"VERB", "AUX"}]
    if verbs:
        past = sum("Past" in t.morph.get("Tense") for t in verbs)
        fut = sum(t.text.lower() in FUTURE_WORDS for t in verbs) + len(re.findall(r"\bgoing to\b", joined))
        feats["past_tense_ratio"] = past / len(verbs)
        feats["future_ratio"] = min(1.0, fut / len(verbs))
    try:
        emp = _empath().analyze(pre.normalized, categories=EMPATH_CATEGORIES, normalize=True) or {}
    except Exception:  # noqa: BLE001 — Empath must never break the engine
        emp = {}
    for c in EMPATH_CATEGORIES:
        feats[f"empath_{c}"] = float(emp.get(c, 0.0) or 0.0)
    return feats


@lru_cache(maxsize=1)
def _load_model(path: str = str(MODEL_PATH)) -> dict[str, Any] | None:
    try:
        import joblib

        return joblib.load(path)
    except Exception:  # noqa: BLE001 — missing/corrupt model => features-only mode
        return None


class StylometricEngine(Engine):
    name = "stylometric"
    version = "0.1.0"
    requires = frozenset({"text"})
    languages = frozenset({"en"})

    def analyze(self, inp: EngineInput) -> EngineOutput:
        feats = extract_features(inp.text)
        n_words = math.expm1(feats["n_words"])
        coverage = min(1.0, n_words / 8.0)
        model = _load_model()
        if model is None:
            return EngineOutput(engine=self.name, version=self.version, features=feats, confidence=0.0, coverage=coverage)
        x = [[feats[k] for k in model["feature_names"]]]
        v, a, d = (max(-1.0, min(1.0, float(p))) for p in model["pipeline"].predict(x)[0])
        std = tuple(float(s) for s in model["dev_rmse"])
        return EngineOutput(
            engine=self.name,
            version=self.version,
            vad=VAD(valence=v, arousal=a, dominance=d, std=std),
            emotions={},
            features=feats,
            confidence=0.2 * coverage,
            coverage=coverage,
        )
