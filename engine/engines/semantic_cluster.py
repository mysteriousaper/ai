"""Engine F. Position in a hierarchical emotion-concept map.

Two map versions (constructor arg `map_version`, default "v2" if its artifacts exist, else "v1"):
- v1 (0.1.0): MiniLM sentence embedding vs leaf prototypes. Clusters are TOPICAL (see the report).
- v2 (0.2.0): emotion-space map. Representation = standardized logits of Engine B's 28 probabilities
  (optionally + PCA-16 of MiniLM x0.3, as chosen at build time). Nearest-prototype (cosine) soft
  membership softmax(sim / T). Emotions = membership-weighted mix of the leaves' label distributions.
  NOTE: v2 emotions are derived from Engine B, so they are NOT independent evidence for fusion;
  use F as explanation/neighbourhood (box path, leaf words), low weight.

The map is fitted ONCE offline by scripts/build_cluster_map.py and loaded lazily; nothing is refitted
at request time. English only (languages={"en"}). VAD is not this engine's job and stays None.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from engine.engines.base import Engine
from engine.mappings.labels import CANONICAL
from engine.schemas import EngineInput, EngineOutput

ROOT = Path(__file__).resolve().parents[2]
MAP_DIR = ROOT / "data/processed/cluster_map"
MAP_DIR_V2 = ROOT / "data/processed/cluster_map_v2"
MODEL_DIR = ROOT / "data/models/sentence-transformers__all-MiniLM-L6-v2"
TEMPERATURE = 0.05  # v1
TEMPERATURE_V2 = 0.1
TOP_LEAVES = 3
PCA_WEIGHT = 0.3


def _is_v2(d: Path) -> bool:
    return (d / "rep.npz").exists()


def artifacts_available(map_dir: Path | None = None, map_version: str | None = None) -> bool:
    """True if the map in `map_dir` (default: per map_version, v2 preferred) and its models can be loaded."""
    if map_dir is None:
        v = map_version or ("v2" if artifacts_available(MAP_DIR_V2, "v2") else "v1")
        map_dir = MAP_DIR_V2 if v == "v2" else MAP_DIR
    d = Path(map_dir)
    if not all((d / f).exists() for f in ("map.json", "prototypes.npy", "label_dist.npy")):
        return False
    if _is_v2(d):
        from engine.engines.transformer import models_available

        if not models_available():
            return False
        if bool(np.load(d / "rep.npz")["with_emb"]):
            return (MODEL_DIR / "config.json").exists()
        return True
    return (MODEL_DIR / "config.json").exists()


class SemanticClusterEngine(Engine):
    name: ClassVar[str] = "semantic_cluster"
    version: ClassVar[str] = "0.2.0"
    requires = frozenset({"text"})
    languages = frozenset({"en"})

    def __init__(self, map_dir: Path | None = None, map_version: str | None = None):
        if map_version not in (None, "v1", "v2"):
            raise ValueError(f"unknown map_version {map_version!r}")
        if map_dir is None:
            map_version = map_version or ("v2" if artifacts_available(MAP_DIR_V2, "v2") else "v1")
            map_dir = MAP_DIR_V2 if map_version == "v2" else MAP_DIR
        self.map_dir = Path(map_dir)
        self.map_version = "v2" if _is_v2(self.map_dir) else "v1"
        if self.map_version == "v1":
            self.version = "0.1.0"  # instance override: v1 maps keep reporting 0.1.0
        self._loaded = False
        self._model = None
        self._b = None

    def _load(self) -> bool:
        if self._loaded:
            return True
        if not artifacts_available(self.map_dir):
            return False
        meta = json.loads((self.map_dir / "map.json").read_text())
        self.leaves = meta["leaves"]
        self.boxes = {b["id"]: b for b in meta["boxes"]}
        self.proto = np.load(self.map_dir / "prototypes.npy").astype(np.float64)
        self.ldist = np.load(self.map_dir / "label_dist.npy").astype(np.float64)
        order = [meta["labels"].index(c) for c in CANONICAL]
        self.ldist = self.ldist[:, order]
        self.temperature = TEMPERATURE
        self.with_emb = self.map_version == "v1"
        if self.map_version == "v2":
            from engine.engines.transformer import TransformerEngine

            r = np.load(self.map_dir / "rep.npz")
            self.rep = {k: r[k] for k in r.files}
            self.with_emb = bool(self.rep["with_emb"])
            self.temperature = float(meta.get("temperature", TEMPERATURE_V2))
            self._b = TransformerEngine()
        if self.with_emb:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(str(MODEL_DIR), device="cpu")
        self._loaded = True
        return True

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    def _represent(self, texts: list[str], base_probs: np.ndarray | None = None) -> np.ndarray:
        """Vectors compared against the prototypes: v1 = MiniLM embedding; v2 = emotion-space representation."""
        if self.map_version == "v1":
            return self._embed(texts)
        r = self.rep
        p = self._b.predict_probs(texts) if base_probs is None else np.asarray(base_probs)
        z = (self._logit(p) - r["mu"]) / r["sd"]
        if self.with_emb:
            e = ((self._embed(texts) - r["pca_mean"]) @ r["pca_components"].T - r["emu"]) / r["esd"]
            z = np.hstack([z, PCA_WEIGHT * e])
        return z

    def _soft(self, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = vec.astype(np.float64)
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
        sim = v @ self.proto.T
        z = sim / self.temperature
        z -= z.max(1, keepdims=True)
        m = np.exp(z)
        return sim, m / m.sum(1, keepdims=True)

    def predict_probs(self, texts: list[str], base_probs: np.ndarray | None = None) -> np.ndarray:
        """Batch emotions (n, 28) in CANONICAL order; used by the evaluation. Needs the map.

        v2 only: `base_probs` = precomputed Engine B probabilities (n, 28) to skip re-running B."""
        if not self._load():
            raise RuntimeError("cluster_map_missing")
        _, m = self._soft(self._represent(texts, base_probs))
        return np.clip(m @ self.ldist, 0.0, 1.0)

    def analyze(self, inp: EngineInput) -> EngineOutput:
        if not self._load():
            return EngineOutput.empty(self.name, self.version, "cluster_map_missing")
        sim, m = self._soft(self._represent([inp.text]))
        sim, m = sim[0], m[0]
        probs = np.clip(m @ self.ldist, 0.0, 1.0)
        top = np.argsort(-m)[:TOP_LEAVES]
        top_feats = []
        for k in top:
            leaf = self.leaves[int(k)]
            box = self.boxes[leaf["parent"]]
            top_feats.append({
                "leaf": int(k), "name": leaf["name"], "membership": round(float(m[k]), 4),
                "similarity": round(float(sim[k]), 4),
                "dominant_labels": [d["label"] for d in leaf["dominant_labels"]],
                "path": [box["name"], leaf["name"]], "box": int(box["id"]),
            })
        conf = float(np.clip(m.max() * max(sim.max(), 0.0), 0.0, 1.0))
        return EngineOutput(
            engine=self.name, version=self.version, vad=None,
            emotions={c: float(p) for c, p in zip(CANONICAL, probs, strict=True)},
            features={
                "top_leaves": top_feats, "nearest_leaf": int(top[0]), "max_similarity": round(float(sim.max()), 4),
                "box": top_feats[0]["box"], "box_path": top_feats[0]["path"], "map_version": self.map_version,
                "derived_from_engine_b": self.map_version == "v2",
            },
            confidence=conf, coverage=1.0,
        )
