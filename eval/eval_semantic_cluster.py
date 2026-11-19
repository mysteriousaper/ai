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

from engine.engines.semantic_cluster import MAP_DIR, SemanticClusterEngine
from eval.datasets import load_goemotions
from eval.eval_transformer import N_BOOT, N_EKMAN, boot, ekman_f1, macro_f1


def main() -> None:
    eng = SemanticClusterEngine()
    vt, Yv = load_goemotions("validation")
    tt, Yt = load_goemotions("test")
    Yv, Yt = Yv.astype(int), Yt.astype(int)
    Pv, Pt = eng.predict_probs(vt), eng.predict_probs(tt)
    grid = np.linspace(0.02, 0.6, 59)
    f_val = [macro_f1(Yv, (Pv >= t).astype(int)) for t in grid]
    thr = float(grid[int(np.argmax(f_val))])
    pred = (Pt >= thr).astype(int)
    f1 = boot(macro_f1, Yt, pred)
    ek = ekman_f1(Yt[:N_EKMAN], Pt[:N_EKMAN])
    neu = np.zeros_like(Yt)
    neu[:, -1] = 1
    f_major = macro_f1(Yt, neu)

    meta = json.loads((MAP_DIR / "map.json").read_text())
    met = meta["metrics"]
    L = [
        "# Engine F (semantic cluster map v0.1: MiniLM + UMAP + HDBSCAN) evaluation",
        "",
        f"## GoEmotions test (n={len(Yt)}), single global threshold {thr:.2f} tuned on validation, 95% CI ({N_BOOT} resamples)",
        "",
        "| Model | Macro F1 | Ekman-6 F1 (first 2000, thr 0.3) |",
        "|---|---|---|",
        f"| Semantic cluster (F) | {f1} | {ek:.3f} |",
        "| Transformer (B), per-label tuned thresholds | 0.522 [0.499, 0.540] | 0.628 |",
        "| Lexicon (A) | n/a (see lexicon.md) | 0.245 [0.229, 0.259] |",
        f"| Majority ('neutral') | {f_major:.3f} | 0.093 |",
        "",
        "## Cluster quality (train corpus, from build log / metrics.json)",
        "",
        "| Metric | HDBSCAN | K-means (same k) |",
        "|---|---|---|",
        f"| n leaf clusters | {met['n_clusters']} | {met['n_clusters']} |",
        f"| noise share | {met['noise_share']:.3f} | 0 |",
        f"| NMI vs primary label | {met['nmi_hdbscan_nonnoise']:.3f} (non-noise), {met['nmi_hdbscan_noise_as_cluster']:.3f} (noise as cluster) | {met['nmi_kmeans_all']:.3f} (all), {met['nmi_kmeans_same_points']:.3f} (same pts) |",
        f"| purity (majority primary share {met['majority_primary_share']:.3f}) | {met['purity_hdbscan']:.3f} | {met['purity_kmeans']:.3f} |",
        f"| stability ARI, 2 other seeds/samples | {met['stability_ari'][0]:.3f}, {met['stability_ari'][1]:.3f} | n/a |",
        f"| min_cluster_size (+ selection; grid 50/100/200 eom, 50/100 leaf) | {met['min_cluster_size']} ({met['selection']}) | |",
        "",
        f"## Top-level boxes ({met['n_boxes']}) and child leaves",
        "",
    ]
    leaves = meta["leaves"]
    for b in meta["boxes"]:
        L.append(f"### Box {b['id']}: {b['name']} (n={b['size']})")
        for c in sorted(b["children"], key=lambda i: -leaves[i]["size"]):
            lf = leaves[c]
            dl = ", ".join(f"{d['label']} {d['share']:.2f}" for d in lf["dominant_labels"])
            L.append(f"- leaf {c} (n={lf['size']}): {dl}; words: {', '.join(lf['top_words'][:5])}")
        L.append("")
    L += [
        "## Verdict",
        "",
        f"- As an emotion classifier F is weak (macro F1 {macro_f1(Yt, pred):.3f} vs B 0.522): it mixes neighbour label distributions, so rare labels never cross one global threshold (Ekman-6 at the fixed 0.3 threshold is even below the majority baseline for the same reason). Known failure: 'this is terrifying' lands in an admiration leaf (xfail test). It was not meant to beat B.",
        f"- Its value is neighbourhoods and explanation: nearest leaf, box path, words and example texts. Clusters are semantic/topical more than purely emotional (NMI vs label {met['nmi_hdbscan_nonnoise']:.2f}); stability ARI {met['stability_ari_mean']:.2f}.",
        "- English-only (MiniLM-L6). Leaf assignment uses nearest 384-D prototype, not HDBSCAN prediction, so it can differ from the offline labels. Do not read boxes as natural psychological categories.",
    ]
    (ROOT / "eval/reports/semantic_cluster.md").write_text("\n".join(L) + "\n")
    print(f"thr={thr:.2f} macroF1={macro_f1(Yt, pred):.3f} ekman={ek:.3f}")


if __name__ == "__main__":
    main()
