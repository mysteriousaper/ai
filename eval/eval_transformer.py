"""Evaluate Engine B (transformer): GoEmotions test, EmoBank test, latency.

Run: `uv run python eval/eval_transformer.py` -> eval/reports/transformer.md
Requires scripts/tune_transformer.py to have run (thresholds + VAD ridge).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.engines.transformer import ONNX_FILE, THRESHOLDS_PATH, TransformerEngine
from engine.mappings.labels import CANONICAL, EKMAN, to_ekman
from engine.schemas import EngineInput
from eval.datasets import load_emobank, load_goemotions
from eval.metrics import bootstrap_ci, ccc, mae, pearson

PROC = ROOT / "data/processed"
N_BOOT = 200
N_EKMAN = 2000
EK_THRESH = 0.3


def fmt(p, lo, hi) -> str:
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def boot(f, *arrs, n=N_BOOT) -> str:
    return fmt(*bootstrap_ci(f, *arrs, n=n))


def cached(path: Path, texts, eng) -> np.ndarray:
    if path.exists():
        return np.load(path)
    p = eng.predict_probs(texts)
    np.save(path, p)
    return p


def pred_matrix(prob: np.ndarray, thr: np.ndarray) -> np.ndarray:
    return (prob >= thr).astype(int)


def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="macro", zero_division=0))


def micro_f1(y, pred) -> float:
    return float(f1_score(y, pred, average="micro", zero_division=0))


def macro_auc(y, prob) -> float:
    a = [roc_auc_score(y[:, j], prob[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(a))


def ekman_f1(Y: np.ndarray, probs: np.ndarray) -> float:
    def proj(rows):
        return np.array([[to_ekman(dict(zip(CANONICAL, map(float, r), strict=True)))[e] for e in EKMAN] for r in rows])

    truth = (proj(Y) >= EK_THRESH).astype(int)
    pred = (proj(probs) >= EK_THRESH).astype(int)
    return macro_f1(truth, pred)


def latency(backend: str, msgs: list[str]) -> tuple[float, float]:
    eng = TransformerEngine(backend)
    eng(EngineInput(text="warm up"))
    ts = []
    for m in msgs:
        t = time.perf_counter()
        eng(EngineInput(text=m))
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.mean(ts)), float(np.percentile(ts, 95))


def main() -> None:
    eng = TransformerEngine()
    thr_default = np.full(len(CANONICAL), 0.5)
    thr_tuned = np.array([json.loads(THRESHOLDS_PATH.read_text())[c] for c in CANONICAL])

    texts, Y = load_goemotions("test")
    Y = Y.astype(int)
    P = cached(PROC / "goemotions_test_probs.npy", texts, eng)
    neu = CANONICAL.index("neutral")
    base = np.zeros_like(Y)
    base[:, neu] = 1
    prior = np.zeros_like(P)
    prior[:, neu] = 1.0

    lines = [
        "# Engine B (transformer v0: SamLowe/roberta-base-go_emotions) evaluation",
        "",
        f"## GoEmotions test (n={len(Y)}, 28 labels), 95% bootstrap CI ({N_BOOT} resamples), backend={eng.backend_name}",
        "",
        "| Model | Macro F1 | Micro F1 | ROC-AUC macro |",
        "|---|---|---|---|",
    ]
    rows = {}
    for name, thr in (("Transformer @0.5", thr_default), ("Transformer tuned thresholds", thr_tuned)):
        pr = pred_matrix(P, thr)
        # ROC-AUC is threshold-free: same value for both rows
        lines.append(
            f"| {name} | {boot(macro_f1, Y, pr)} | {boot(micro_f1, Y, pr)} | "
            f"{boot(macro_auc, Y, P, n=100)} |"
        )
        rows[name] = macro_f1(Y, pr)
    lines.append(
        f"| Majority ('neutral' always) | {boot(macro_f1, Y, base)} | {boot(micro_f1, Y, base)} | 0.500 (constant) |"
    )
    f_major = macro_f1(Y, base)

    # Ekman-6 coarse comparison with the lexicon engine (same recipe as eval_lexicon.py)
    ek_t = ekman_f1(Y[:N_EKMAN], P[:N_EKMAN])
    lines += [
        "",
        f"## Ekman-6 macro F1 (first {N_EKMAN} test rows, project with to_ekman, threshold {EK_THRESH})",
        "",
        "| Model | Macro F1 |",
        "|---|---|",
        f"| Transformer | {ek_t:.3f} |",
        "| Lexicon (from eval/reports/lexicon.md) | 0.245 [0.229, 0.259] |",
        "| Majority class (lexicon report) | 0.093 |",
    ]

    # EmoBank
    et, V, A, D = load_emobank("test")
    Pe = cached(PROC / "emobank_test_probs.npy", et, eng)
    vad = eng.vad_from_probs(Pe)
    lines += [
        "",
        f"## EmoBank test (n={len(et)}), VAD from ridge on 28 probabilities",
        "",
        "| Dim | Pearson r | CCC | MAE |",
        "|---|---|---|---|",
    ]
    rs = {}
    for k, (dim, y) in enumerate((("valence", V), ("arousal", A), ("dominance", D))):
        p = vad[:, k]
        rs[dim] = pearson(p, y)
        lines.append(f"| {dim} | {boot(pearson, p, y)} | {boot(ccc, p, y)} | {boot(mae, p, y)} |")
    lines.append("| (lexicon valence, for reference) | 0.523 [0.471, 0.578] | 0.364 | 0.231 |")

    # latency
    rng = np.random.default_rng(0)
    msgs = [texts[i] for i in rng.choice(len(texts), 100, replace=False)]
    lat = {}
    backends = (["onnx"] if ONNX_FILE.exists() else []) + ["torch"]
    lines += ["", "## Latency per single message, CPU, 100 GoEmotions test messages", "", "| Backend | mean ms | p95 ms |", "|---|---|---|"]
    for b in backends:
        lat[b] = latency(b, msgs)
        lines.append(f"| {b} | {lat[b][0]:.1f} | {lat[b][1]:.1f} |")

    tuned = rows["Transformer tuned thresholds"]
    lines += [
        "",
        "## Verdict",
        "",
        f"- GoEmotions: macro F1 {rows['Transformer @0.5']:.3f} @0.5 -> {tuned:.3f} tuned (thresholds tuned on validation, evaluated on test) vs majority {f_major:.3f}; clearly beats baseline, in line with published ~0.46-0.52 for this model.",
        f"- Ekman-6 F1 {ek_t:.3f} vs lexicon 0.245; EmoBank r (V/A/D) = {rs['valence']:.3f}/{rs['arousal']:.3f}/{rs['dominance']:.3f} vs lexicon valence 0.523. VAD is an indirect ridge on emotion probabilities, not a trained regression head, so arousal/dominance stay weak.",
        "- Caveat: the model was trained on GoEmotions (Reddit); EmoBank (mixed genres) is out-of-domain. Bootstrap CIs ignore threshold-tuning variance.",
    ]
    (ROOT / "eval/reports/transformer.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-40:]))


if __name__ == "__main__":
    main()
