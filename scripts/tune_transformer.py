"""Tune Engine B: per-label thresholds on GoEmotions validation, ridge VAD head on EmoBank.

Run: `uv run python scripts/tune_transformer.py` (log: data/processed/tune_transformer.log).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.engines.transformer import RIDGE_PATH, THRESHOLDS_PATH, TransformerEngine
from engine.mappings.labels import CANONICAL
from eval.datasets import load_emobank, load_goemotions

PROC = ROOT / "data/processed"


def cached(path: Path, texts: list[str], eng: TransformerEngine) -> np.ndarray:
    if path.exists():
        return np.load(path)
    p = eng.predict_probs(texts, batch_size=32)
    np.save(path, p)
    return p


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.05):
        pred = p >= t
        tp = float((pred & (y == 1)).sum())
        denom = 2 * tp + float((pred & (y == 0)).sum()) + float((~pred & (y == 1)).sum())
        f = 2 * tp / denom if denom else 0.0
        if f > best_f + 1e-12:
            best, best_f = float(round(t, 2)), f
    return best


def main() -> None:
    eng = TransformerEngine()
    print("backend", eng.backend_name, flush=True)
    texts, Y = load_goemotions("validation")
    p = cached(PROC / "goemotions_val_probs.npy", texts, eng)
    thr = {c: best_threshold(Y[:, j], p[:, j]) for j, c in enumerate(CANONICAL)}
    THRESHOLDS_PATH.write_text(json.dumps(thr, indent=1))
    print("thresholds saved", flush=True)

    tr_t, *tr_y = load_emobank("train")
    dv_t, *dv_y = load_emobank("dev")
    ptr = cached(PROC / "emobank_train_probs.npy", tr_t, eng)
    pdv = cached(PROC / "emobank_dev_probs.npy", dv_t, eng)
    ytr, ydv = np.column_stack(tr_y), np.column_stack(dv_y)
    best = None
    for alpha in (0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30):
        m = Ridge(alpha=alpha).fit(ptr, ytr)
        rmse = np.sqrt(((np.clip(m.predict(pdv), -1, 1) - ydv) ** 2).mean(0))
        print("alpha", alpha, "dev rmse", rmse.round(4), flush=True)
        if best is None or rmse.mean() < best[0]:
            best = (rmse.mean(), alpha, rmse)
    _, alpha, rmse = best
    model = Ridge(alpha=alpha).fit(ptr, ytr)
    joblib.dump({"model": model, "std": [float(x) for x in rmse], "alpha": alpha}, RIDGE_PATH)
    print("ridge saved alpha", alpha, "std", rmse.round(4), flush=True)


if __name__ == "__main__":
    main()
