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


def main_v1() -> None:
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


# ======================================================================================
# v2: emotion-space map. Representation = standardized logits of Engine B's 28 probabilities
# (optionally + 16-D PCA of MiniLM embeddings, weight 0.3). Artifacts in cluster_map_v2/.
# Run: `uv run python scripts/build_cluster_map.py probs` (cache B probabilities, slow, background)
# then `uv run python scripts/build_cluster_map.py v2`. `v1` rebuilds the old topical map.
# ======================================================================================
OUT2 = ROOT / "data/processed/cluster_map_v2"
PROC = ROOT / "data/processed"
GRID2 = ((30, "leaf"), (50, "leaf"), (100, "leaf"), (50, "eom"), (100, "eom"), (200, "eom"))
MAX_NOISE2 = 0.40
MAX_K = 100
PCA_DIMS, PCA_W = 16, 0.3


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def cached_probs(texts: list[str], path: Path) -> np.ndarray:
    if path.exists():
        a = np.load(path)
        if len(a) == len(texts):
            return a
    from engine.engines.transformer import TransformerEngine

    eng = TransformerEngine()
    order = np.argsort([len(t) for t in texts])  # length-sorted batches: far less padding
    out = np.zeros((len(texts), len(CANONICAL)))
    t0 = time.time()
    for i in range(0, len(texts), 2048):
        idx = order[i : i + 2048]
        out[idx] = eng.predict_probs([texts[j] for j in idx], batch_size=64)
        log.info("probs %d/%d (%.0fs)", min(i + 2048, len(texts)), len(texts), time.time() - t0)
    np.save(path, out)
    return out


class Rep:
    """Representation transform (fitted on train): standardized logits [+ weighted PCA of embeddings]."""

    def __init__(self, with_emb: bool):
        self.with_emb = with_emb

    def fit(self, probs: np.ndarray, emb: np.ndarray | None):
        L = logit(probs)
        self.mu, self.sd = L.mean(0), L.std(0) + 1e-6
        if self.with_emb:
            from sklearn.decomposition import PCA

            self.pca = PCA(PCA_DIMS, random_state=SEED).fit(emb)
            e = self.pca.transform(emb)
            self.emu, self.esd = e.mean(0), e.std(0) + 1e-6
        return self

    def transform(self, probs: np.ndarray, emb: np.ndarray | None) -> np.ndarray:
        z = (logit(probs) - self.mu) / self.sd
        if self.with_emb:
            z = np.hstack([z, PCA_W * (self.pca.transform(emb) - self.emu) / self.esd])
        return z.astype(np.float32)

    def save(self, path: Path) -> None:
        d = {"mu": self.mu, "sd": self.sd, "with_emb": np.array(self.with_emb)}
        if self.with_emb:
            d |= {"pca_components": self.pca.components_, "pca_mean": self.pca.mean_, "emu": self.emu, "esd": self.esd}
        np.savez(path, **d)


def fit_umap2(z: np.ndarray, metric: str, seed: int):
    import umap

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(z), min(N_FIT, len(z)), replace=False)
    red = umap.UMAP(n_components=8, metric=metric, n_neighbors=30, min_dist=0.0, random_state=seed)
    red.fit(z[idx])
    return red.transform(z)


def hdb2(u: np.ndarray, mcs: int, sel: str):
    import hdbscan

    return hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=10, cluster_selection_method=sel, core_dist_n_jobs=4).fit(u).labels_


def protos_of(z: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    P = np.vstack([z[labels == c].mean(0) for c in range(K)])
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def assign(z: np.ndarray, P: np.ndarray) -> np.ndarray:
    zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)
    return (zn @ P.T).argmax(1)


def main_probs() -> None:
    OUT2.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler()])
    texts, _ = load_goemotions("train")
    cached_probs(texts, OUT2 / "train_probs.npy")


def main_v2() -> None:
    OUT2.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(OUT2 / "build.log", mode="w"), logging.StreamHandler()],
    )
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    t0 = time.time()
    texts, Y = load_goemotions("train")
    vtexts, Yv = load_goemotions("validation")
    prim, vprim = Y.argmax(1), Yv.argmax(1)
    probs = cached_probs(texts, OUT2 / "train_probs.npy")
    vprobs = cached_probs(vtexts, PROC / "goemotions_val_probs.npy")
    emb = embed(texts)
    from sentence_transformers import SentenceTransformer

    vemb = SentenceTransformer(str(MODEL_DIR), device="cpu").encode(
        vtexts, batch_size=128, normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)
    log.info("data ready n=%d val=%d (%.0fs)", len(texts), len(vtexts), time.time() - t0)

    # --- variant search: representation x UMAP metric x HDBSCAN config; score on held-out validation
    results, best = [], None
    for with_emb in (False, True):
        rep = Rep(with_emb).fit(probs, emb)
        z, vz = rep.transform(probs, emb), rep.transform(vprobs, vemb)
        for metric in ("cosine", "euclidean"):
            u = fit_umap2(z, metric, SEED)
            for mcs, sel in GRID2:
                lab = hdb2(u, mcs, sel)
                K = int(lab.max() + 1)
                noise = float((lab < 0).mean())
                if K < 6:
                    continue
                P = protos_of(z, lab, K)
                va = assign(vz, P)
                v_nmi, v_pur = nmi(va, vprim), purity(va, vprim)
                tag = f"{'emb+' if with_emb else ''}logit/{metric}/{mcs}-{sel}"
                results.append({"cfg": tag, "k": K, "noise": noise, "val_nmi": v_nmi, "val_purity": v_pur})
                log.info("%s k=%d noise=%.3f val NMI=%.3f purity=%.3f", tag, K, noise, v_nmi, v_pur)
                ok = noise <= MAX_NOISE2 and K <= MAX_K
                key = (ok, v_nmi if ok else -noise)
                if best is None or key > best[0]:
                    best = (key, with_emb, metric, mcs, sel, lab, K, noise, u, z, rep, tag)
    _, with_emb, metric, mcs, sel, labels, K, noise, u, z, rep, tag = best
    log.info("CHOSEN %s k=%d noise=%.3f", tag, K, noise)
    m = labels >= 0
    prim_m = prim

    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(u)
    metrics = {
        "variant": tag, "with_emb": with_emb, "umap_metric": metric, "min_cluster_size": mcs, "selection": sel,
        "n_clusters": K, "noise_share": noise, "search": results,
        "nmi_hdbscan_nonnoise": nmi(labels, prim_m, m),
        "nmi_hdbscan_noise_as_cluster": nmi(np.where(labels < 0, K, labels), prim_m),
        "nmi_kmeans_all": nmi(km.labels_, prim_m), "nmi_kmeans_same_points": nmi(km.labels_, prim_m, m),
        "purity_hdbscan": purity(labels, prim_m), "purity_kmeans": purity(km.labels_, prim_m),
        "majority_primary_share": float(np.bincount(prim_m).max() / len(prim_m)),
    }
    P = protos_of(z, labels, K)
    va = assign(rep.transform(vprobs, vemb), P)
    metrics["val_nmi"], metrics["val_purity"] = nmi(va, vprim), purity(va, vprim)
    # nearest-prototype assignment of ALL train points (what the engine does) vs primary label
    ta = assign(z, P)
    metrics["nmi_proto_assign_all"], metrics["purity_proto_assign_all"] = nmi(ta, prim_m), purity(ta, prim_m)

    aris = []
    for s in (1, 2):
        l2 = hdb2(fit_umap2(z, metric, SEED + s), mcs, sel)
        both = m & (l2 >= 0)
        aris.append(float(adjusted_rand_score(labels[both], l2[both])))
        log.info("stability seed %d: k=%d noise=%.3f ARI=%.3f", SEED + s, l2.max() + 1, (l2 < 0).mean(), aris[-1])
    metrics["stability_ari"], metrics["stability_ari_mean"] = aris, float(np.mean(aris))

    ldist = np.vstack([Y[labels == c].mean(0) for c in range(K)]).astype(np.float32)
    sizes = np.bincount(labels[m], minlength=K)
    np.save(OUT2 / "prototypes.npy", P.astype(np.float32))
    np.save(OUT2 / "label_dist.npy", ldist)
    rep.save(OUT2 / "rep.npz")

    from scipy.cluster.hierarchy import fcluster, linkage

    n_top = min(N_TOP, max(2, K // 2))
    parent = fcluster(linkage(P, method="ward"), t=n_top, criterion="maxclust") - 1
    n_top = int(parent.max() + 1)

    from sklearn.feature_extraction.text import CountVectorizer

    cv = CountVectorizer(stop_words="english", min_df=5, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b")
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    cc = np.vstack([np.asarray(X[labels == c].sum(0)).ravel() for c in range(K)]).astype(float)
    idf = np.log(1 + cc.sum(1).mean() / np.maximum(cc.sum(0), 1))

    def top_words(counts: np.ndarray, n: int = 5) -> list[str]:
        return vocab[np.argsort(-(counts / max(counts.sum(), 1) * idf))[:n]].tolist()

    def dom(d: np.ndarray, n: int = 3) -> list[dict]:
        return [{"label": CANONICAL[j], "share": round(float(d[j]), 3)} for j in np.argsort(-d)[:n]]

    leaves = []
    for c in range(K):
        mem = np.where(labels == c)[0]
        zn = z[mem] / np.maximum(np.linalg.norm(z[mem], axis=1, keepdims=True), 1e-9)
        rep_t = [texts[mem[i]][:160] for i in np.argsort(-(zn @ P[c]))[:3]]
        d = dom(ldist[c])
        leaves.append({
            "id": c, "size": int(sizes[c]), "parent": int(parent[c]), "dominant_labels": d,
            "top_words": top_words(cc[c]), "representative_texts": rep_t,
            "name": "/".join(x["label"] for x in d[:2]) + " | " + ", ".join(top_words(cc[c], 3)),
        })
    boxes = []
    for b in range(n_top):
        ch = np.where(parent == b)[0]
        w = sizes[ch] / sizes[ch].sum()
        d = dom((ldist[ch] * w[:, None]).sum(0))
        words = top_words(cc[ch].sum(0))
        boxes.append({
            "id": b, "children": ch.tolist(), "size": int(sizes[ch].sum()), "dominant_labels": d, "top_words": words,
            "name": "/".join(x["label"] for x in d[:2]) + " | " + ", ".join(words[:3]),
        })
    metrics["n_boxes"] = n_top
    # soft-membership temperature: best validation macro-F1 (global threshold) over a small grid
    from eval.eval_transformer import macro_f1

    vzn = rep.transform(vprobs, vemb)
    vzn = vzn / np.maximum(np.linalg.norm(vzn, axis=1, keepdims=True), 1e-9)
    vsim, tune = vzn @ P.T, {}
    for T in (0.03, 0.05, 0.1, 0.2, 0.4):
        e = np.exp((vsim - vsim.max(1, keepdims=True)) / T)
        pv = (e / e.sum(1, keepdims=True)) @ ldist
        tune[T] = max(macro_f1(Yv.astype(int), (pv >= t).astype(int)) for t in np.linspace(0.02, 0.6, 30))
    temperature = max(tune, key=tune.get)
    metrics["temperature"], metrics["temperature_val_macro_f1"] = temperature, {str(k): round(v, 4) for k, v in tune.items()}
    (OUT2 / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (OUT2 / "map.json").write_text(json.dumps({
        "version": "0.2.0", "seed": SEED, "representation": "standardized logit of Engine B probs"
        + (f" + PCA{PCA_DIMS} of MiniLM x{PCA_W}" if with_emb else ""),
        "with_emb": with_emb, "umap_metric": metric, "temperature": temperature, "labels": list(CANONICAL),
        "metrics": metrics, "boxes": boxes, "leaves": leaves,
    }, indent=1))
    log.info("METRICS %s", json.dumps({k: v for k, v in metrics.items() if k != "search"}))
    log.info("done (%.0fs)", time.time() - t0)


if __name__ == "__main__":
    {"v1": main_v1, "v2": main_v2, "probs": main_probs}[sys.argv[1] if len(sys.argv) > 1 else "v2"]()
