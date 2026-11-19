"""Engine F. Sentence embedding -> position in the hierarchical emotion-concept map.

The map (leaf prototypes, 28-D label distributions, Ward hierarchy, names) is fitted ONCE offline
by scripts/build_cluster_map.py and loaded lazily; nothing is refitted at request time.
A message is embedded, compared (cosine) to the leaf prototypes, and gets a soft membership
softmax(sim / 0.05). Emotions = membership-weighted mix of the leaves' label distributions.

Language: the English all-MiniLM-L6-v2 model is used, so languages={"en"}. The multilingual route
(paraphrase-multilingual-MiniLM-L12-v2, then languages=None) needs the map rebuilt with that model.
VAD is not this engine's job and stays None.
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
MODEL_DIR = ROOT / "data/models/sentence-transformers__all-MiniLM-L6-v2"
TEMPERATURE = 0.05
TOP_LEAVES = 3


def artifacts_available(map_dir: Path | None = None) -> bool:
    d = map_dir or MAP_DIR
    return all((d / f).exists() for f in ("map.json", "prototypes.npy", "label_dist.npy")) and (
        MODEL_DIR / "config.json"
    ).exists()


class SemanticClusterEngine(Engine):
    name: ClassVar[str] = "semantic_cluster"
    version: ClassVar[str] = "0.1.0"
    requires = frozenset({"text"})
    languages = frozenset({"en"})

    def __init__(self, map_dir: Path | None = None):
        self.map_dir = Path(map_dir) if map_dir else MAP_DIR
        self._loaded = False
        self._model = None

    def _load(self) -> bool:
        if self._loaded:
            return True
        if not artifacts_available(self.map_dir):
            return False
        from sentence_transformers import SentenceTransformer

        meta = json.loads((self.map_dir / "map.json").read_text())
        self.leaves = meta["leaves"]
        self.boxes = {b["id"]: b for b in meta["boxes"]}
        self.proto = np.load(self.map_dir / "prototypes.npy").astype(np.float64)
        self.ldist = np.load(self.map_dir / "label_dist.npy").astype(np.float64)
        order = [meta["labels"].index(c) for c in CANONICAL]
        self.ldist = self.ldist[:, order]
        self._model = SentenceTransformer(str(MODEL_DIR), device="cpu")
        self._loaded = True
        return True

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

    def _soft(self, emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sim = emb.astype(np.float64) @ self.proto.T
        z = sim / TEMPERATURE
        z -= z.max(1, keepdims=True)
        m = np.exp(z)
        return sim, m / m.sum(1, keepdims=True)

    def predict_probs(self, texts: list[str]) -> np.ndarray:
        """Batch emotions (n, 28) in CANONICAL order; used by the evaluation. Needs the map."""
        if not self._load():
            raise RuntimeError("cluster_map_missing")
        _, m = self._soft(self._embed(texts))
        return np.clip(m @ self.ldist, 0.0, 1.0)

    def analyze(self, inp: EngineInput) -> EngineOutput:
        if not self._load():
            return EngineOutput.empty(self.name, self.version, "cluster_map_missing")
        emb = self._embed([inp.text])
        sim, m = self._soft(emb)
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
            features={"top_leaves": top_feats, "nearest_leaf": int(top[0]), "max_similarity": round(float(sim.max()), 4)},
            confidence=conf, coverage=1.0,
        )
