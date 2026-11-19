"""Build data/processed/lexicon_en.parquet from NRC VAD v2.1 + NRC EmoLex (word level).

One row per lowercase term: v, a, d (NaN if absent, range [-1, 1]) + 8 EmoLex 0/1 columns.
Run: `uv run python scripts/build_lexicons.py`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VAD_PATH = ROOT / "data/raw/nrc/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
EMOLEX_PATH = ROOT / "data/raw/nrc/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
OUT_PATH = ROOT / "data/processed/lexicon_en.parquet"
EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")


def build(out_path: Path = OUT_PATH) -> Path:
    vad = pd.read_csv(VAD_PATH, sep="\t", keep_default_na=False, na_values=[""])
    vad.columns = ["term", "v", "a", "d"]
    vad = vad.dropna(subset=["term"]).assign(term=lambda df: df["term"].astype(str).str.lower().str.strip())
    vad = vad.groupby("term", as_index=False).mean(numeric_only=True)

    emo = pd.read_csv(EMOLEX_PATH, sep="\t", header=None, names=["term", "emotion", "flag"], keep_default_na=False)
    emo = emo[emo["emotion"].isin(EMOTIONS)].assign(term=lambda df: df["term"].astype(str).str.lower().str.strip())
    wide = emo.pivot_table(index="term", columns="emotion", values="flag", aggfunc="max").reindex(columns=list(EMOTIONS))
    wide = wide.fillna(0).astype("int8").reset_index()

    df = vad.merge(wide, on="term", how="outer")
    df[list(EMOTIONS)] = df[list(EMOTIONS)].fillna(0).astype("int8")
    df = df.sort_values("term").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    p = build()
    df = pd.read_parquet(p)
    print(f"wrote {p} rows={len(df)} with_vad={int(df['v'].notna().sum())} with_emo={int((df[list(EMOTIONS)].sum(axis=1) > 0).sum())}")
