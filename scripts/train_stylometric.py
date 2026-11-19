"""Train the weak stylometric VAD ridge on EmoBank (train split, alpha tuned on dev)."""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.engines.stylometric import FEATURE_NAMES, MODEL_PATH, extract_features
from engine.mappings.scales import emobank

CACHE = ROOT / "data" / "processed" / "stylometric_features.parquet"


def load_features() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    df = pd.read_csv(ROOT / "data" / "raw" / "emobank" / "emobank.csv")
    feats = pd.DataFrame([extract_features(str(t)) for t in df["text"]])[FEATURE_NAMES]
    out = pd.concat([df[["id", "split", "V", "A", "D"]].reset_index(drop=True), feats], axis=1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE)
    return out


def main() -> None:
    df = load_features()
    y = np.column_stack([df[c].map(emobank) for c in ["V", "A", "D"]])
    tr, dv, te = (df["split"] == s for s in ("train", "dev", "test"))
    X = df[FEATURE_NAMES].to_numpy()
    rmse = lambda p, t: np.sqrt(((p - t) ** 2).mean(axis=0))
    best = None
    for alpha in [0.1, 1, 10, 100]:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], y[tr])
        r = rmse(np.clip(pipe.predict(X[dv]), -1, 1), y[dv])
        print(f"alpha={alpha}: dev RMSE V/A/D = {np.round(r, 4)} mean={r.mean():.4f}")
        if best is None or r.mean() < best[0]:
            best = (r.mean(), alpha, pipe, r)
    _, alpha, pipe, dev_rmse = best
    base = rmse(np.tile(y[tr].mean(axis=0), (dv.sum(), 1)), y[dv])
    print(f"best alpha={alpha}; dev RMSE={np.round(dev_rmse, 4)}; mean-baseline dev RMSE={np.round(base, 4)}")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "feature_names": FEATURE_NAMES, "dev_rmse": [float(x) for x in dev_rmse], "alpha": alpha},
        MODEL_PATH,
    )
    print("saved", MODEL_PATH, "| n train/dev/test:", int(tr.sum()), int(dv.sum()), int(te.sum()))


if __name__ == "__main__":
    main()
