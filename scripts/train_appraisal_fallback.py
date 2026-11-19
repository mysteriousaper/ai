"""Fallback event-type classifier for engine E (local, zero cost).

EmpatheticDialogues prompt -> EventType via ED_TO_EVENT. Frozen MiniLM sentence embeddings +
balanced LogisticRegression (C tuned on valid). Replaces the DistilRoBERTa fine-tune for now (CPU cost).
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine.mappings.labels import ED_TO_EVENT

ED_DIR = ROOT / "data/raw/empathetic_dialogues/empatheticdialogues"
EMB_MODEL = ROOT / "data/models/sentence-transformers__all-MiniLM-L6-v2"
OUT = ROOT / "data/processed/appraisal"


def load_ed(split: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(ED_DIR / f"{split}.csv", on_bad_lines="skip")
    df = df.drop_duplicates("conv_id")
    df = df[df["context"].isin(ED_TO_EVENT)]
    texts = [str(p).replace("_comma_", ",") for p in df["prompt"]]
    return texts, [ED_TO_EVENT[c].value for c in df["context"]]


def embed(texts: list[str], cache: Path) -> np.ndarray:
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer

    x = SentenceTransformer(str(EMB_MODEL), device="cpu").encode(texts, batch_size=64, normalize_embeddings=True)
    np.save(cache, x)
    return x


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {s: load_ed(s) for s in ("train", "valid", "test")}
    X = {s: embed(t, OUT / f"emb_{s}.npy") for s, (t, _) in data.items()}
    best = (-1.0, 1.0, None)
    for c in (0.1, 0.3, 1, 3, 10, 30):
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000).fit(X["train"], data["train"][1])
        f1 = f1_score(data["valid"][1], clf.predict(X["valid"]), average="macro")
        print(f"C={c} valid macro-F1={f1:.4f}")
        if f1 > best[0]:
            best = (f1, c, clf)
    f1, c, clf = best
    joblib.dump(clf, OUT / "fallback.joblib")
    print(f"best C={c} valid macro-F1={f1:.4f}; test macro-F1="
          f"{f1_score(data['test'][1], clf.predict(X['test']), average='macro'):.4f}; classes={len(clf.classes_)}")


if __name__ == "__main__":
    main()
