"""Evaluate engine E event typing on the EmpatheticDialogues test split (one row per conversation)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.engines import appraisal as A
from engine.mappings.labels import ED_TO_EVENT, EventType
from eval.metrics import bootstrap_ci

N_ZS = 600


def macro(y, p):
    return f1_score(y, p, average="macro", labels=sorted(set(y)), zero_division=0)


def row(name, y, p, n):
    a = bootstrap_ci(accuracy_score, y, p)
    f = bootstrap_ci(macro, y, p)
    return f"| {name} | {n} | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] | {f[0]:.3f} [{f[1]:.3f}, {f[2]:.3f}] |"


def main() -> None:
    df = pd.read_csv(A.ROOT / "data/raw/empathetic_dialogues/empatheticdialogues/test.csv", on_bad_lines="skip")
    df = df.drop_duplicates("conv_id")
    df = df[df["context"].isin(ED_TO_EVENT)].copy()
    df["text"] = [str(p).replace("_comma_", ",") for p in df["prompt"]]
    df["y"] = [ED_TO_EVENT[c].value for c in df["context"]]
    eng = A.AppraisalEngine()
    fb = eng._fallback()
    assert fb is not None, "run scripts/train_appraisal_fallback.py first"
    clf, emb = fb
    X = emb.encode(df["text"].tolist(), batch_size=64, normalize_embeddings=True)
    fbp = clf.predict_proba(X)
    classes = list(clf.classes_)
    y_all = df["y"].to_numpy()
    p_fb_all = np.array(classes)[fbp.argmax(1)]

    # stratified sample for zero-shot
    samp = df.groupby("y", group_keys=False).apply(lambda g: g.sample(max(1, round(N_ZS * len(g) / len(df))), random_state=0))
    idx = [df.index.get_loc(i) for i in samp.index]
    zs_probs, comb = [], []
    t0 = time.perf_counter()
    for i in idx:
        zs = eng._zero_shot(df["text"].iloc[i], A.EVENT_DESC, A.EVENT_TEMPLATE, multi=False)
        zs_v = np.array([zs[EventType(c)] for c in classes])
        zs_probs.append(zs_v)
        comb.append(0.5 * zs_v + 0.5 * fbp[i])
    zs_ms = (time.perf_counter() - t0) / len(idx) * 1000
    zs_all = np.array(zs_probs)
    # zero-shot may also predict event types with no ED source; restrict argmax to all 16 for honesty
    full = [EventType(c) for c in classes]
    y = y_all[idx]
    p_zs = np.array(classes)[zs_all.argmax(1)]
    p_fb = p_fb_all[idx]
    p_comb = np.array(classes)[np.array(comb).argmax(1)]
    maj = pd.Series(y_all).value_counts().index[0]
    p_maj = np.array([maj] * len(y))

    lines = [
        "# Engine E (appraisal) report",
        "",
        (f"ED test split, one row per conversation ({len(df)} rows; zero-shot/combined on a stratified sample of "
         f"{len(y)}). Labels: ED_TO_EVENT (lossy, {len(classes)} of 16 event types have ED support). "
         "95% bootstrap CIs."),
        "",
        "| System | n | Accuracy | Macro-F1 |",
        "|---|---|---|---|",
        row("Majority baseline", y, p_maj, len(y)),
        row("Zero-shot NLI", y, p_zs, len(y)),
        row("Fallback LR (same sample)", y, p_fb, len(y)),
        row("Combined 0.5/0.5", y, p_comb, len(y)),
        row("Fallback LR (all test)", y_all, p_fb_all, len(y_all)),
        "",
        ("Fallback = frozen MiniLM embeddings + balanced LogisticRegression (replaces the DistilRoBERTa fine-tune "
         "for now: CPU cost). Zero-shot restricted to the event types present in ED labels for scoring "
         f"({len(full)}); the engine itself ranks all 16."),
        "",
    ]
    # latency of full engine
    from engine.schemas import EngineInput

    msgs = df["text"].tolist()[:20]
    eng(EngineInput(text=msgs[0]))
    t = time.perf_counter()
    for m in msgs:
        eng(EngineInput(text=m))
    lat = (time.perf_counter() - t) / len(msgs) * 1000
    lines += [f"Latency (CPU): full engine {lat:.0f} ms/message (20 ED prompts); zero-shot event only {zs_ms:.0f} ms.", ""]
    # challenge categories
    try:
        import json

        ch = [json.loads(line) for line in (ROOT / "eval/challenge_set.jsonl").read_text().splitlines() if line.strip()]
        sar = [c for c in ch if "sarcasm" in c.get("expected", {})]
        if sar:
            hit = sum(eng(EngineInput(text=c["text"])).sarcasm_suspected == bool(c["expected"]["sarcasm"]) for c in sar)
            lines += [f"Challenge sarcasm flag agreement: {hit}/{len(sar)} (clauses.sarcasm_cues >= 0.5).", ""]
    except Exception as exc:  # noqa: BLE001
        lines += [f"Challenge scoring skipped: {type(exc).__name__}", ""]
    lines += ["VERDICT_PLACEHOLDER"]
    (ROOT / "eval/reports/appraisal.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
