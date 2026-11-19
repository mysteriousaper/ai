"""Evaluate Engine C (stylometric weak VAD) on EmoBank test vs a mean-of-train baseline.

Run: `uv run python eval/eval_stylometric.py` -> eval/reports/stylometric.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.engines.stylometric import StylometricEngine
from engine.schemas import EngineInput
from eval.datasets import load_emobank
from eval.metrics import bootstrap_ci, ccc, mae, pearson


def fmt(f, *arrs) -> str:
    p, lo, hi = bootstrap_ci(f, *arrs, n=500)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def main() -> None:
    eng = StylometricEngine()
    _, *train = load_emobank("train")
    texts, *gold = load_emobank("test")
    outs = [eng(EngineInput(text=t)) for t in texts]
    if any(o.vad is None for o in outs):
        raise SystemExit("model missing or engine skipped rows: run scripts/train_stylometric.py")
    lines = [
        "# Engine C (stylometric) evaluation",
        "",
        (f"EmoBank test (n={len(texts)}), 95% bootstrap CI (500 resamples). Baseline = constant train mean "
        "(Pearson r is undefined for a constant; effectively 0)."),
        "",
        "| Dim | Model | Pearson r | CCC | MAE |",
        "|---|---|---|---|---|",
    ]
    res = {}
    for name, key, y, ytr in zip(("valence", "arousal", "dominance"), ("valence", "arousal", "dominance"), gold, train, strict=True):
        p = np.array([getattr(o.vad, key) for o in outs])
        b = np.full_like(y, ytr.mean())
        res[name] = (pearson(p, y), ccc(p, y), ccc(b, y), mae(p, y), mae(b, y))
        lines.append(f"| {name} | Stylometric ridge | {fmt(pearson, p, y)} | {fmt(ccc, p, y)} | {fmt(mae, p, y)} |")
        lines.append(f"| {name} | Mean-of-train baseline | undefined (0) | {fmt(ccc, b, y)} | {fmt(mae, b, y)} |")
    r = {k: v[0] for k, v in res.items()}
    lines += [
        "",
        (f"Verdict: weak (r V/A/D = {r['valence']:.2f}/{r['arousal']:.2f}/{r['dominance']:.2f}), "
        "small MAE/CCC gain over the baseline: a low-confidence prior only."),
        "Value is in the explainable raw features, best read relative to a personal baseline.",
        "",
    ]
    out = ROOT / "eval" / "reports" / "stylometric.md"
    out.write_text("\n".join(lines))
    print("\n".join(lines[5:]))


if __name__ == "__main__":
    main()
