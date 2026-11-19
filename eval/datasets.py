"""Loaders for the evaluation splits under data/raw (research data only).

Each loader returns plain Python / numpy objects:
- GoEmotions: (texts, Y) with Y multi-hot (n, 28) in engine.mappings.labels.CANONICAL order.
- EmoBank: (texts, V, A, D) with values mapped to [-1, 1] by engine.mappings.scales.emobank.
- SemEval 2018 EI-reg: (texts, intensity) with intensity in [0, 1].
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from engine.mappings.labels import CANONICAL
from engine.mappings.scales import emobank as emobank_to_common

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"

GOEMOTIONS_DIR = DATA / "go_emotions"
EMOBANK_CSV = DATA / "emobank" / "emobank.csv"
SEMEVAL_EIREG_DIR = (
    DATA / "semeval2018" / "SemEval2018-Task1-all-data" / "English" / "EI-reg"
)

_GO_SPLITS = {"train": "train", "validation": "validation", "dev": "validation", "test": "test"}
_EMOBANK_SPLITS = {"train": "train", "dev": "dev", "validation": "dev", "test": "test"}
_SEMEVAL_SPLITS = {
    "train": ("training", "train"),
    "training": ("training", "train"),
    "dev": ("development", "dev"),
    "development": ("development", "dev"),
    "test": ("test-gold", "test-gold"),
    "test-gold": ("test-gold", "test-gold"),
}
SEMEVAL_EMOTIONS = ("anger", "fear", "joy", "sadness")


def load_goemotions(split: str) -> tuple[list[str], np.ndarray]:
    """GoEmotions split as (texts, multi-hot Y of shape (n, len(CANONICAL)))."""
    if split not in _GO_SPLITS:
        raise ValueError(f"unknown GoEmotions split: {split!r}")
    df = pd.read_parquet(GOEMOTIONS_DIR / f"{_GO_SPLITS[split]}.parquet", columns=["text", "labels"])
    texts = df["text"].astype(str).tolist()
    y = np.zeros((len(df), len(CANONICAL)), dtype=np.int8)
    for i, ids in enumerate(df["labels"]):
        y[i, np.asarray(ids, dtype=int)] = 1
    return texts, y


def load_emobank(split: str) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """EmoBank split as (texts, V, A, D), VAD mapped from 1-5 to [-1, 1]. Rows without text dropped."""
    if split not in _EMOBANK_SPLITS:
        raise ValueError(f"unknown EmoBank split: {split!r}")
    df = pd.read_csv(EMOBANK_CSV)
    df = df[df["split"] == _EMOBANK_SPLITS[split]].dropna(subset=["text"])
    texts = df["text"].astype(str).tolist()

    def conv(col: str) -> np.ndarray:
        return np.array([emobank_to_common(float(v)) for v in df[col]], dtype=float)

    return texts, conv("V"), conv("A"), conv("D")


def load_semeval_eireg(emotion: str, split: str) -> tuple[list[str], np.ndarray]:
    """SemEval 2018 EI-reg (English) for one emotion: (tweets, intensity in [0, 1])."""
    if emotion not in SEMEVAL_EMOTIONS:
        raise ValueError(f"unknown emotion: {emotion!r}")
    if split not in _SEMEVAL_SPLITS:
        raise ValueError(f"unknown SemEval split: {split!r}")
    folder, suffix = _SEMEVAL_SPLITS[split]
    matches = sorted((SEMEVAL_EIREG_DIR / folder).glob(f"*EI-reg-En-{emotion}-{suffix}.txt"))
    if not matches:
        raise FileNotFoundError(f"no EI-reg file for {emotion}/{split} in {SEMEVAL_EIREG_DIR / folder}")

    texts: list[str] = []
    scores: list[float] = []
    with matches[0].open(encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)  # header: ID, Tweet, Affect Dimension, Intensity Score
        for row in reader:
            if not row:
                continue
            texts.append(row[1])
            scores.append(float(row[-1]))
    return texts, np.asarray(scores, dtype=float)
