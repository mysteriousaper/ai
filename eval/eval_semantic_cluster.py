"""Evaluate Engine F (semantic cluster map) on GoEmotions test -> eval/reports/semantic_cluster.md.

Run: `uv run python -m eval.eval_semantic_cluster` (not by path: eval/datasets.py would shadow HF datasets) (needs scripts/build_cluster_map.py first).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.engines.semantic_cluster import SemanticClusterEngine
from eval.datasets import load_goemotions
from eval.eval_transformer import N_BOOT, N_EKMAN, boot, ekman_f1, macro_f1

MAP_V1 = ROOT / "data/processed/cluster_map"
MAP_V2 = ROOT / "data/processed/cluster_map_v2"
PROC = ROOT / "data/processed"


def evaluate(version: str, vt, Yv, tt, Yt, base):
    """Returns (metrics dict, thr, macro-F1 string with CI, ekman). `base` = cached B probs (v2 only)."""
    eng = SemanticClusterEngine(map_dir=MAP_V2 if version == "v2" else MAP_V1)
    bv, bt = (base["val"], base["test"]) if version == "v2" else (None, None)
    Pv = eng.predict_probs(vt, bv) if version == "v2" else eng.predict_probs(vt)
    Pt = eng.predict_probs(tt, bt) if version == "v2" else eng.predict_probs(tt)
    grid = np.linspace(0.02, 0.6, 59)
    thr = float(grid[int(np.argmax([macro_f1(Yv, (Pv >= t).astype(int)) for t in grid]))])
    pred = (Pt >= thr).astype(int)
    return {
        "thr": thr, "f1": boot(macro_f1, Yt, pred), "f1_point": macro_f1(Yt, pred),
        "ekman": ekman_f1(Yt[:N_EKMAN], Pt[:N_EKMAN]),
    }


def main() -> None:
    vt, Yv = load_goemotions("validation")
    tt, Yt = load_goemotions("test")
    Yv, Yt = Yv.astype(int), Yt.astype(int)
    base = {"val": np.load(PROC / "goemotions_val_probs.npy"), "test": np.load(PROC / "goemotions_test_probs.npy")}
    res = {v: evaluate(v, vt, Yv, tt, Yt, base) for v in ("v1", "v2")}
    neu = np.zeros_like(Yt)
    neu[:, -1] = 1
    f_major = macro_f1(Yt, neu)
    meta = {v: json.loads((d / "map.json").read_text()) for v, d in (("v1", MAP_V1), ("v2", MAP_V2))}
    m1, m2 = meta["v1"]["metrics"], meta["v2"]["metrics"]
    r1, r2 = res["v1"], res["v2"]

    def row(name, f):
        return f"| {name} | {f(m1, r1)} | {f(m2, r2)} |"

    L = [
        "# Engine F (semantic / emotion-space cluster map) evaluation: v1 vs v2",
        "",
        "v1 = MiniLM embeddings (topical). v2 = emotion-space map: standardized logits of Engine B's 28 probabilities"
        + (f" + PCA-{16} of MiniLM x0.3" if m2["with_emb"] else " (no embedding part: it lost on validation)")
        + f"; UMAP {m2['umap_metric']}, HDBSCAN min_cluster_size {m2['min_cluster_size']} ({m2['selection']}), "
        f"soft-membership T={m2['temperature']}. Variant chosen on validation NMI among configs with noise <= 0.40.",
        "",
        f"## v1 vs v2 (GoEmotions test n={len(Yt)}; one global threshold tuned on validation; 95% CI, {N_BOOT} resamples)",
        "",
        "| Metric | v1 | v2 |",
        "|---|---|---|",
        row("Macro F1 (28 labels)", lambda m, r: r["f1"]),
        row("Ekman-6 F1 (first 2000, thr 0.3)", lambda m, r: f"{r['ekman']:.3f}"),
        row("NMI vs primary label (HDBSCAN, non-noise, train)", lambda m, r: f"{m['nmi_hdbscan_nonnoise']:.3f}"),
        row("NMI, K-means same k (all pts)", lambda m, r: f"{m['nmi_kmeans_all']:.3f}"),
        row(f"Purity (HDBSCAN; majority share {m1['majority_primary_share']:.3f})", lambda m, r: f"{m['purity_hdbscan']:.3f}"),
        row("Purity, K-means", lambda m, r: f"{m['purity_kmeans']:.3f}"),
        row("Noise share", lambda m, r: f"{m['noise_share']:.3f}"),
        row("Leaf clusters / boxes", lambda m, r: f"{m['n_clusters']} / {m['n_boxes']}"),
        row("Stability ARI (2 other seeds)", lambda m, r: f"{m['stability_ari'][0]:.3f}, {m['stability_ari'][1]:.3f}"),
        "",
        "| Reference | Macro F1 | Ekman-6 F1 |",
        "|---|---|---|",
        "| Transformer (B), per-label tuned thresholds | 0.522 [0.499, 0.540] | 0.628 |",
        "| Lexicon (A) | n/a (see lexicon.md) | 0.245 [0.229, 0.259] |",
        f"| Majority ('neutral') | {f_major:.3f} | 0.093 |",
        "",
        (
            f"v2 held-out validation (nearest-prototype assignment): NMI {m2['val_nmi']:.3f}, purity {m2['val_purity']:.3f}. "
            f"Variant search: {len(m2['search'])} configs (logit-only vs +emb, cosine vs euclidean UMAP, 6 HDBSCAN settings), "
            "see cluster_map_v2/metrics.json."
        ),
        "",
        f"## v2 top-level boxes ({m2['n_boxes']}) and child leaves",
        "",
    ]
    leaves = meta["v2"]["leaves"]
    for b in meta["v2"]["boxes"]:
        L.append(f"### Box {b['id']}: {b['name']} (n={b['size']})")
        for c in sorted(b["children"], key=lambda i: -leaves[i]["size"]):
            lf = leaves[c]
            dl = ", ".join(f"{d['label']} {d['share']:.2f}" for d in lf["dominant_labels"])
            L.append(f"- leaf {c} (n={lf['size']}): {dl}; words: {', '.join(lf['top_words'][:5])}")
        L.append("")
    L += [
        "## v1 top-level boxes (for reference; leaves omitted)",
        "",
        *[f"- Box {b['id']}: {b['name']} (n={b['size']})" for b in meta["v1"]["boxes"]],
        "",
        "## Verdict",
        "",
        f"- v2 is {'clearly more emotional' if m2['nmi_hdbscan_nonnoise'] > 1.5 * m1['nmi_hdbscan_nonnoise'] else 'only somewhat more emotional'} than v1 (NMI {m1['nmi_hdbscan_nonnoise']:.2f} -> {m2['nmi_hdbscan_nonnoise']:.2f}, macro F1 {r1['f1_point']:.3f} -> {r2['f1_point']:.3f}, noise {m1['noise_share']:.2f} -> {m2['noise_share']:.2f}); it still does not beat B (0.522), because its output is a smoothed mix of cluster label distributions. Train NMI/purity are in-sample and non-noise only; held-out validation NMI is {m2['val_nmi']:.2f}, and the 24 variants differ by <0.04 val NMI (emb+logit vs logit-only is within noise).",
        "- Independence: v2 emotions are DERIVED from Engine B's probabilities, so they are NOT independent evidence for fusion. Fusion should treat F as explanation/neighbourhood (box path, leaf words, similar texts) with low weight; do not count it as a second vote next to B.",
        "- Caveats: English only; noise points get no leaf (assignment is nearest prototype, so it can differ from offline labels); boxes are data-driven neighbourhoods, not natural psychological categories; GoEmotions labels are noisy and neutral-heavy.",
    ]
    (ROOT / "eval/reports/semantic_cluster.md").write_text("\n".join(L) + "\n")
    print(f"v1 macroF1={r1['f1_point']:.3f} ekman={r1['ekman']:.3f} | v2 macroF1={r2['f1_point']:.3f} ekman={r2['ekman']:.3f}")


if __name__ == "__main__":
    main()
