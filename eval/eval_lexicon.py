"""Evaluate Engine A (lexicon): EmoBank VAD vs VADER, GoEmotions Ekman-6 macro F1 vs majority baseline.

Run: `uv run python eval/eval_lexicon.py` -> eval/reports/lexicon.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.engines.lexicon import LexiconEngine
from engine.mappings.labels import CANONICAL, EKMAN, to_ekman
from engine.schemas import EngineInput
from eval.datasets import load_emobank, load_goemotions
from eval.metrics import bootstrap_ci, ccc, mae, pearson

N_GO = 2000
THRESH = 0.3


def fmt(f, *arrs) -> str:
    p, lo, hi = bootstrap_ci(f, *arrs, n=500)
    return f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def macro_f1(yt, yp) -> float:
    return float(f1_score(np.asarray(yt, int), np.asarray(yp, int), average="macro", zero_division=0))


def main() -> None:
    eng = LexiconEngine()
    vader = SentimentIntensityAnalyzer()

    texts, V, A, D = load_emobank("test")
    outs = [eng(EngineInput(text=t)) for t in texts]
    pred = {k: np.array([getattr(o.vad, k) if o.vad else 0.0 for o in outs]) for k in ("valence", "arousal", "dominance")}
    vad_cmp = np.array([vader.polarity_scores(t)["compound"] for t in texts])
    lines = [
        "# Engine A (lexicon) evaluation",
        "",
        f"## EmoBank test (n={len(texts)}), 95% bootstrap CI (500 resamples)",
        "",
        "| Dim | Model | Pearson r | CCC | MAE |",
        "|---|---|---|---|---|",
    ]
    rows = [
        ("valence", "Lexicon", pred["valence"], V),
        ("valence", "VADER compound (baseline)", vad_cmp, V),
        ("arousal", "Lexicon", pred["arousal"], A),
        ("dominance", "Lexicon", pred["dominance"], D),
        ("valence", "Constant 0 (baseline)", np.zeros_like(V), V),
    ]
    res = {}
    for dim, name, p, y in rows:
        r = pearson(p, y) if p.std() > 0 else float("nan")
        res[(dim, name)] = r
        lines.append(f"| {dim} | {name} | {fmt(pearson, p, y)} | {fmt(ccc, p, y)} | {fmt(mae, p, y)} |")

    gtexts, Y = load_goemotions("test")
    gtexts, Y = gtexts[:N_GO], Y[:N_GO]
    truth = np.array([[to_ekman({c: float(v) for c, v in zip(CANONICAL, row, strict=True)})[e] for e in EKMAN] for row in Y])
    truth_b = (truth >= THRESH).astype(int)
    preds = []
    for t in gtexts:
        em = eng(EngineInput(text=t)).emotions
        ek = to_ekman(em)
        preds.append([ek[e] for e in EKMAN])
    pred_b = (np.array(preds) >= THRESH).astype(int)
    major = int(np.argmax(truth_b.sum(0)))
    base_b = np.zeros_like(truth_b)
    base_b[:, major] = 1
    lines += [
        "",
        f"## GoEmotions test, first {len(gtexts)} rows, Ekman-6 macro F1 (threshold {THRESH}, multi-label)",
        "",
        "| Model | Macro F1 [95% CI] |",
        "|---|---|",
        f"| Lexicon (EmoLex) | {fmt(macro_f1, truth_b, pred_b)} |",
        f"| Majority class ('{EKMAN[major]}' always on) | {fmt(macro_f1, truth_b, base_b)} |",
    ]
    f_lex, f_base = macro_f1(truth_b, pred_b), macro_f1(truth_b, base_b)
    v_lex, v_vad = res[("valence", "Lexicon")], res[("valence", "VADER compound (baseline)")]
    lines += [
        "",
        "## Verdict",
        "",
        f"- EmoBank valence: lexicon r={v_lex:.3f} {'beats' if v_lex > v_vad else 'does not beat'} VADER r={v_vad:.3f} (see CI overlap above).",
        "- Caveat: the constant-0 row shows raw MAE is dominated by EmoBank's narrow label range; rank by r/CCC.",
        f"- GoEmotions Ekman-6: lexicon F1={f_lex:.3f} {'beats' if f_lex > f_base else 'does not beat'} majority baseline F1={f_base:.3f}.",
    ]
    out = ROOT / "eval/reports/lexicon.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
