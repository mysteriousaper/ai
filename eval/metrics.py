"""Evaluation metrics for multi-label / single-label emotion classification and VAD regression.

All functions take plain numpy arrays. Correlation and CCC use population (ddof=0) moments, so
CCC <= |Pearson| holds exactly. Undefined cases (zero variance, no positives) return NaN.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def multilabel_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Sequence[str],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Macro/micro F1, per-label F1 and macro ROC-AUC for multi-hot targets.

    y_true: (n, L) binary; y_prob: (n, L) probabilities; labels: L names.
    ROC-AUC is averaged over labels that have both classes in y_true (others are skipped).
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    per_label = f1_score(y_true, y_pred, average=None, zero_division=0)
    aucs = [
        roc_auc_score(y_true[:, j], y_prob[:, j])
        for j in range(y_true.shape[1])
        if 0 < y_true[:, j].sum() < y_true.shape[0]
    ]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "per_label_f1": {lab: float(v) for lab, v in zip(labels, per_label, strict=True)},
        "roc_auc_macro": float(np.mean(aucs)) if aucs else float("nan"),
        "threshold": threshold,
    }


def f1_single(y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any]) -> float:
    """Macro F1 for single-label classification over an explicit label set."""
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc**2).sum() * (yc**2).sum())
    if denom == 0:
        return float("nan")
    return float(np.clip((xc * yc).sum() / denom, -1.0, 1.0))


def ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient: 2*cov / (var_x + var_y + (mean_x - mean_y)^2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    if denom == 0:
        return float("nan")
    return float(2 * cov / denom)


def mae(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))))


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: Any,
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of metric_fn(*arrays), resampling rows jointly.

    Returns (point, lo, hi). The point estimate is always inside [lo, hi]: if the percentile
    interval happens to exclude it (skewed metrics on small samples), the bound is widened to it.
    NaN replicates (e.g. a resample with zero variance) are dropped.
    """
    if not arrays:
        raise ValueError("bootstrap_ci needs at least one array")
    prepared = [np.asarray(a) if not isinstance(a, list) else np.asarray(a, dtype=object) for a in arrays]
    size = len(prepared[0])
    point = float(metric_fn(*prepared))

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        val = float(metric_fn(*(a[idx] for a in prepared)))
        if not np.isnan(val):
            stats.append(val)
    if not stats:
        return point, float("nan"), float("nan")

    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(min(lo, point)), float(max(hi, point))
