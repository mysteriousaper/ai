"""Offline build of Engine F's semantic map (fit ONCE, seed 42).

GoEmotions train -> MiniLM embeddings -> UMAP(8D, cosine) -> HDBSCAN leaves (+ K-means baseline)
-> Ward hierarchy over 384-D leaf centroids -> names (labels, c-TF-IDF, representative texts).
Artifacts in data/processed/cluster_map/: map.json, prototypes.npy, label_dist.npy,
reducer.joblib, embeddings_train.npy, metrics.json, build.log.

Run: `uv run python scripts/build_cluster_map.py`
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.mappings.labels import CANONICAL
from eval.datasets import load_goemotions

OUT = ROOT / "data/processed/cluster_map"
MODEL_DIR = ROOT / "data/models/sentence-transformers__all-MiniLM-L6-v2"
SEED = 42
N_FIT = 20000
GRID = ((50, "eom"), (100, "eom"), (200, "eom"), (50, "leaf"), (100, "leaf"))  # (min_cluster_size, selection)
N_TOP = 8
MAX_NOISE = 0.76  # GoEmotions is short, diffuse text: every config has >49% noise; leaf mode is finer
log = logging.getLogger("cluster_map")


def embed(texts: list[str]) -> np.ndarray:
    path = OUT / "embeddings_train.npy"
    if path.exists():
        e = np.load(path)
        if len(e) == len(texts):
            return e
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(str(MODEL_DIR), device="cpu")
    e = m.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
    np.save(path, e)
    return e


def fit_umap(emb: np.ndarray, seed: int, n_fit: int = N_FIT):
    import umap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(emb), min(n_fit, len(emb)), replace=False)
    red = umap.UMAP(n_components=8, metric="cosine", n_neighbors=30, min_dist=0.0, random_state=seed)
    red.fit(emb[idx])
    return red, red.transform(emb)


def run_hdbscan(z: np.ndarray, mcs: int, sel: str = "eom"):
    import hdbscan

    return hdbscan.HDBSCAN(min_cluster_size=mcs, cluster_selection_method=sel, prediction_data=True, core_dist_n_jobs=4).fit(z)


def purity(labels: np.ndarray, prim: np.ndarray) -> float:
    m = labels >= 0
    tot = 0
    for c in np.unique(labels[m]):
        tot += np.bincount(prim[labels == c]).max()
    return float(tot / max(m.sum(), 1))


def nmi(labels: np.ndarray, prim: np.ndarray, mask: np.ndarray | None = None) -> float:
    from sklearn.metrics import normalized_mutual_info_score as n

    if mask is None:
        mask = np.ones(len(labels), bool)
    return float(n(prim[mask], labels[mask]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(OUT / "build.log", mode="w"), logging.StreamHandler()],
    )
    t0 = time.time()
    texts, Y = load_goemotions("train")
    prim = Y.argmax(1)  # primary label = first/lowest-id positive label
    emb = embed(texts)
    log.info("corpus n=%d emb=%s (%.0fs)", len(texts), emb.shape, time.time() - t0)

    red, z = fit_umap(emb, SEED)
    joblib.dump(red, OUT / "reducer.joblib")
    log.info("UMAP fitted on %d rows (%.0fs)", N_FIT, time.time() - t0)

    # --- tune min_cluster_size
    best = None
    grid = {}
    for mcs, sel in GRID:
        cl = run_hdbscan(z, mcs, sel)
        lab = cl.labels_
        k = int(lab.max() + 1)
        noise = float((lab < 0).mean())
        sc = nmi(lab, prim, lab >= 0) if k > 1 else 0.0
        grid[f"{mcs}-{sel}"] = {"n_clusters": k, "noise": noise, "nmi_nonnoise": sc}
        log.info("HDBSCAN mcs=%d/%s k=%d noise=%.3f NMI(non-noise)=%.3f", mcs, sel, k, noise, sc)
        ok = 6 <= k and noise <= MAX_NOISE
        key = (ok, sc if ok else -noise)
        if best is None or key > best[0]:
            best = (key, mcs, sel, cl)
    _, mcs, sel, cl = best
    labels = cl.labels_
    K = int(labels.max() + 1)
    noise = float((labels < 0).mean())
    log.info("chosen min_cluster_size=%d/%s  n_clusters=%d noise=%.3f", mcs, sel, K, noise)

    # --- K-means baseline (same k, same UMAP space)
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(z)
    m = labels >= 0
    metrics = {
        "min_cluster_size": mcs, "selection": sel, "grid": grid, "n_clusters": K, "noise_share": noise,
        "nmi_hdbscan_nonnoise": nmi(labels, prim, m),
        "nmi_hdbscan_noise_as_cluster": nmi(np.where(labels < 0, K, labels), prim),
        "nmi_kmeans_all": nmi(km.labels_, prim),
        "nmi_kmeans_same_points": nmi(km.labels_, prim, m),
        "purity_hdbscan": purity(labels, prim),
        "purity_kmeans": purity(km.labels_, prim),
        "majority_primary_share": float(np.bincount(prim).max() / len(prim)),
    }

    # --- stability: different seeds AND different UMAP samples; ARI on points non-noise in both
    from sklearn.metrics import adjusted_rand_score

    aris = []
    for s in (1, 2):
        _, z2 = fit_umap(emb, SEED + s)
        l2 = run_hdbscan(z2, mcs, sel).labels_
        both = m & (l2 >= 0)
        aris.append(float(adjusted_rand_score(labels[both], l2[both])))
        log.info("stability seed %d: k=%d noise=%.3f ARI=%.3f (n=%d)", SEED + s, l2.max() + 1, (l2 < 0).mean(), aris[-1], both.sum())
    metrics["stability_ari"] = aris
    metrics["stability_ari_mean"] = float(np.mean(aris))

    # --- prototypes (384-D, L2-normalised) + label distributions
    proto = np.zeros((K, emb.shape[1]), np.float32)
    ldist = np.zeros((K, len(CANONICAL)), np.float32)
    sizes = np.zeros(K, int)
    for c in range(K):
        mem = np.where(labels == c)[0]
        sizes[c] = len(mem)
        v = emb[mem].mean(0)
        proto[c] = v / np.linalg.norm(v)
        ldist[c] = Y[mem].mean(0)
    np.save(OUT / "prototypes.npy", proto)
    np.save(OUT / "label_dist.npy", ldist)

    # --- hierarchy: Ward over centroids in 384-D
    from scipy.cluster.hierarchy import fcluster, linkage

    n_top = min(N_TOP, max(2, K // 2))
    Zl = linkage(proto, method="ward")
    parent = fcluster(Zl, t=n_top, criterion="maxclust") - 1
    n_top = int(parent.max() + 1)

    # --- names: c-TF-IDF per leaf
    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(stop_words="english", min_df=5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    cls_counts = np.vstack([np.asarray(X[labels == c].sum(0)).ravel() for c in range(K)]).astype(float)
    A = cls_counts.sum(1).mean()
    idf = np.log(1 + A / np.maximum(cls_counts.sum(0), 1))

    def top_words(counts: np.ndarray, n: int = 5) -> list[str]:
        tf = counts / max(counts.sum(), 1)
        return vocab[np.argsort(-(tf * idf))[:n]].tolist()

    def dom(d: np.ndarray, n: int = 3) -> list[dict]:
        return [{"label": CANONICAL[j], "share": round(float(d[j]), 3)} for j in np.argsort(-d)[:n]]

    leaves = []
    for c in range(K):
        mem = np.where(labels == c)[0]
        sim = emb[mem] @ proto[c]
        rep = [texts[mem[i]][:160] for i in np.argsort(-sim)[:3]]
        d = dom(ldist[c])
        leaves.append({
            "id": c, "size": int(sizes[c]), "parent": int(parent[c]),
            "dominant_labels": d, "top_words": top_words(cls_counts[c]),
            "representative_texts": rep,
            "name": "/".join(x["label"] for x in d[:2]) + " | " + ", ".join(top_words(cls_counts[c], 3)),
        })
    boxes = []
    for b in range(n_top):
        ch = np.where(parent == b)[0]
        w = sizes[ch] / sizes[ch].sum()
        d = dom((ldist[ch] * w[:, None]).sum(0))
        words = top_words(cls_counts[ch].sum(0))
        boxes.append({
            "id": b, "children": ch.tolist(), "size": int(sizes[ch].sum()),
            "dominant_labels": d, "top_words": words,
            "name": "/".join(x["label"] for x in d[:2]) + " | " + ", ".join(words[:3]),
        })
    metrics["n_boxes"] = n_top
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT / "map.json").write_text(json.dumps({
        "version": "0.1.0", "seed": SEED, "embedding_model": "all-MiniLM-L6-v2",
        "umap": {"n_components": 8, "metric": "cosine", "n_neighbors": 30, "min_dist": 0.0, "n_fit": N_FIT},
        "hdbscan_min_cluster_size": mcs, "labels": list(CANONICAL),
        "metrics": metrics, "boxes": boxes, "leaves": leaves,
    }, indent=1))
    log.info("METRICS %s", json.dumps({k: v for k, v in metrics.items() if k != "grid"}))
    log.info("done (%.0fs)", time.time() - t0)


if __name__ == "__main__":
    main()
