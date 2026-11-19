"""Engine B (v0): pretrained GoEmotions RoBERTa (28 labels) + a ridge VAD head on the probabilities.

Backends: "onnx" (int8 quantized, onnxruntime CPU, default if present) or "torch". Lazy-loaded.
Per-label thresholds come from data/processed/transformer_thresholds.json (default 0.5).
VAD = Ridge(28 emotion probabilities -> EmoBank V/A/D), fitted by scripts/tune_transformer.py.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import ClassVar

import numpy as np

from engine.engines.base import Engine
from engine.mappings.labels import CANONICAL
from engine.schemas import VAD, EngineInput, EngineOutput

ROOT = Path(__file__).resolve().parents[2]
TORCH_DIR = ROOT / "data/models/SamLowe__roberta-base-go_emotions"
ONNX_DIR = ROOT / "data/models/SamLowe__roberta-base-go_emotions-onnx"
ONNX_FILE = ONNX_DIR / "onnx" / "model_quantized.onnx"
THRESHOLDS_PATH = ROOT / "data/processed/transformer_thresholds.json"
RIDGE_PATH = ROOT / "data/processed/transformer_vad_ridge.joblib"
MAX_TOKENS = 256
DEFAULT_THRESHOLD = 0.5


def models_available() -> bool:
    return (TORCH_DIR / "config.json").exists() or ONNX_FILE.exists()


class _Backend:
    """Tokenizer + model producing sigmoid probabilities in CANONICAL order."""

    def __init__(self, kind: str):
        from transformers import AutoTokenizer

        self.kind = kind
        tok_dir = ONNX_DIR / "onnx" if kind == "onnx" else TORCH_DIR
        self.tok = AutoTokenizer.from_pretrained(str(tok_dir))
        if kind == "onnx":
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            self.sess = ort.InferenceSession(str(ONNX_FILE), so, providers=["CPUExecutionProvider"])
            self.input_names = {i.name for i in self.sess.get_inputs()}
            cfg = json.loads((ONNX_DIR / "config.json").read_text())
            id2label = {int(k): v for k, v in cfg["id2label"].items()}
        else:
            import torch
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(str(TORCH_DIR)).eval()
            self.torch = torch
            id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        # map by name, never by position
        self.order = np.array([next(i for i, n in id2label.items() if n == c) for c in CANONICAL])

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=True)["input_ids"])

    def probs(self, texts: list[str]) -> np.ndarray:
        enc = self.tok(texts, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="np")
        if self.kind == "onnx":
            feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.sess.run(None, feed)[0]
        else:
            with self.torch.no_grad():
                t = {k: self.torch.from_numpy(v) for k, v in enc.items()}
                logits = self.model(**t).logits.numpy()
        p = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        return p[:, self.order]


class TransformerEngine(Engine):
    name: ClassVar[str] = "transformer"
    version: ClassVar[str] = "0.1.0"

    def __init__(self, backend: str | None = None):
        if backend is None:
            backend = "onnx" if ONNX_FILE.exists() else "torch"
        if backend not in ("onnx", "torch"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend_name = backend
        self._backend: _Backend | None = None
        self._ridge = None
        self._ridge_loaded = False
        self._thr: np.ndarray | None = None

    # ---- lazy resources
    @property
    def backend(self) -> _Backend:
        if self._backend is None:
            self._backend = _Backend(self.backend_name)
        return self._backend

    @property
    def thresholds(self) -> np.ndarray:
        if self._thr is None:
            thr = np.full(len(CANONICAL), DEFAULT_THRESHOLD)
            if THRESHOLDS_PATH.exists():
                data = json.loads(THRESHOLDS_PATH.read_text())
                for i, c in enumerate(CANONICAL):
                    if c in data:
                        thr[i] = float(data[c])
            self._thr = thr
        return self._thr

    @property
    def ridge(self):
        """(model, std) or None if the VAD head has not been fitted."""
        if not self._ridge_loaded:
            self._ridge_loaded = True
            if RIDGE_PATH.exists():
                import joblib

                self._ridge = joblib.load(RIDGE_PATH)
        return self._ridge

    # ---- batch API (used by tuning/eval)
    def predict_probs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        out = [self.backend.probs(texts[i : i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.concatenate(out) if out else np.zeros((0, len(CANONICAL)))

    def vad_from_probs(self, probs: np.ndarray) -> np.ndarray | None:
        r = self.ridge
        if r is None:
            return None
        return np.clip(r["model"].predict(np.atleast_2d(probs)), -1.0, 1.0)

    @staticmethod
    def confidence_from_probs(p: np.ndarray) -> float:
        """max prob scaled by (1 - normalised entropy of the normalised distribution)."""
        q = p / max(p.sum(), 1e-12)
        ent = -float(np.sum(q * np.log(np.clip(q, 1e-12, 1.0)))) / math.log(len(q))
        return float(np.clip(p.max() * (1.0 - ent), 0.0, 1.0))

    def analyze(self, inp: EngineInput) -> EngineOutput:
        t0 = time.perf_counter()
        be = self.backend
        n_tok = be.n_tokens(inp.text)
        p = be.probs([inp.text])[0]
        coverage = 1.0 if n_tok <= MAX_TOKENS else MAX_TOKENS / n_tok
        emotions = {c: float(np.clip(v, 0.0, 1.0)) for c, v in zip(CANONICAL, p, strict=True)}
        thr = self.thresholds
        predicted = [c for c, v, t in zip(CANONICAL, p, thr, strict=True) if v >= t]
        top5 = [{"label": CANONICAL[i], "prob": round(float(p[i]), 4)} for i in np.argsort(-p)[:5]]

        vad = None
        r = self.ridge
        if r is not None:
            v, a, d = (float(x) for x in self.vad_from_probs(p)[0])
            vad = VAD(valence=v, arousal=a, dominance=d, std=tuple(float(s) for s in r["std"]))
        latency = (time.perf_counter() - t0) * 1000.0
        return EngineOutput(
            engine=self.name,
            version=self.version,
            vad=vad,
            emotions=emotions,
            features={
                "predicted_labels": predicted,
                "top5": top5,
                "backend": self.backend_name,
                "latency_ms": round(latency, 2),
                "n_tokens": n_tok,
            },
            confidence=self.confidence_from_probs(p),
            coverage=float(coverage),
        )
