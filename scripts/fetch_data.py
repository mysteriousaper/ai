"""Download every dataset, lexicon and pretrained model the Engine uses.

Reproducible and idempotent: files already present are skipped. A manifest with
sha256 checksums is written to data/raw/MANIFEST.json so later runs can verify them.

Usage:
    uv run python scripts/fetch_data.py            # datasets + lexicons + models
    uv run python scripts/fetch_data.py --no-models
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet CDN host not on our allowlist; plain LFS works

import argparse
import hashlib
import json
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "data" / "models"

HF = "https://huggingface.co"
PARQUET_BRANCH = "resolve/refs%2Fconvert%2Fparquet"


@dataclass
class Source:
    name: str
    licence: str
    status: str  # commercial-OK | research-only | needs-licence | share-alike
    files: dict[str, str]  # relative dest path -> url
    extract: bool = False
    notes: str = ""
    extra: dict = field(default_factory=dict)


SOURCES: list[Source] = [
    Source(
        "go_emotions",
        "Apache-2.0",
        "commercial-OK",
        {
            f"go_emotions/{split}.parquet": f"{HF}/api/datasets/google-research-datasets/go_emotions/parquet/simplified/{split}/0.parquet"
            for split in ("train", "validation", "test")
        }
        | {"go_emotions/raw.parquet": f"{HF}/api/datasets/google-research-datasets/go_emotions/parquet/raw/train/0.parquet"},
    ),
    Source(
        "emobank",
        "CC-BY-SA-4.0",
        "share-alike",
        {"emobank/emobank.csv": "https://raw.githubusercontent.com/JULIELab/EmoBank/master/corpus/emobank.csv"},
    ),
    Source(
        "warriner_xanew",
        "CC-BY-NC-ND-3.0",
        "research-only",
        {"warriner/Ratings_Warriner_et_al.csv": "https://raw.githubusercontent.com/JULIELab/XANEW/master/Ratings_Warriner_et_al.csv"},
        notes="Secondary distribution by JULIE Lab of Warriner et al. 2013.",
    ),
    Source(
        "nrc_vad_v2_1",
        "NRC non-commercial terms",
        "needs-licence",
        {"nrc/NRC-VAD-Lexicon-v2.1.zip": "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"},
        extract=True,
        notes="Commercial licence available from NRC for a fee. Includes Armenian/Russian auto-translations.",
    ),
    Source(
        "nrc_emolex",
        "NRC non-commercial terms",
        "needs-licence",
        {"nrc/NRC-Emotion-Lexicon.zip": "https://saifmohammad.com/WebDocs/Lexicons/NRC-Emotion-Lexicon.zip"},
        extract=True,
    ),
    Source(
        "empathetic_dialogues",
        "CC-BY-NC-4.0",
        "research-only",
        {"empathetic_dialogues/empatheticdialogues.tar.gz": "https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz"},
        extract=True,
    ),
    Source(
        "daily_dialog",
        "CC-BY-NC-SA-4.0",
        "research-only",
        {
            f"daily_dialog/{split}.parquet": f"{HF}/datasets/li2017dailydialog/daily_dialog/{PARQUET_BRANCH}/default/{split}/0000.parquet"
            for split in ("train", "validation", "test")
        },
    ),
    Source(
        "semeval2018_task1",
        "SemEval-2018 terms (research only, no redistribution)",
        "research-only",
        {"semeval2018/SemEval2018-Task1-all-data.zip": "https://saifmohammad.com/WebDocs/AIT-2018/AIT2018-DATA/SemEval2018-Task1-all-data.zip"},
        extract=True,
    ),
]

MODEL_IDS = {
    # id: (licence, status, role)
    "SamLowe/roberta-base-go_emotions": ("MIT", "commercial-OK", "Engine B off-the-shelf baseline (28 GoEmotions labels)"),
    "j-hartmann/emotion-english-distilroberta-base": ("unspecified", "needs-licence", "Engine B comparison (Ekman 6 + neutral)"),
    "sentence-transformers/all-MiniLM-L6-v2": ("Apache-2.0", "commercial-OK", "Engine F English embeddings"),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ("Apache-2.0", "commercial-OK", "Engine F multilingual embeddings (ru/hy)"),
    "distilroberta-base": ("Apache-2.0", "commercial-OK", "Backbone for our own fine-tune + VAD head"),
}

MANUAL = [
    ("EmoSurv (keystrokes)", "https://ieee-dataport.org/open-access/emosurv-typing-biometric-keystroke-dynamics-dataset-emotion-labels-created-using",
     "Free IEEE account login required. Download all 4 CSVs into data/raw/emosurv/. Non-commercial research only."),
    ("ISEAR", "https://www.unige.ch/cisa/research/materials-and-online-research/research-material/",
     "Optional. Licence unclear; request from University of Geneva (CISA) if wanted. Put files in data/raw/isear/."),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "emotion-engine-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.rename(dest)


def extract(archive: Path) -> None:
    out = archive.parent
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    elif archive.name.endswith(".tar.gz"):
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")


def fetch_sources() -> list[dict]:
    manifest = []
    for src in SOURCES:
        for rel, url in src.files.items():
            dest = RAW / rel
            if dest.exists():
                print(f"  skip   {rel}")
            else:
                print(f"  fetch  {rel}")
                try:
                    download(url, dest)
                except OSError as e:  # URLError is an OSError; report and continue, never loop
                    print(f"  FAILED {rel}: {e}")
                    continue
                if src.extract:
                    extract(dest)
            manifest.append(
                {"name": src.name, "file": rel, "url": url, "licence": src.licence,
                 "status": src.status, "sha256": sha256(dest), "bytes": dest.stat().st_size, "notes": src.notes}
            )
    return manifest


def fetch_models() -> list[dict]:
    from huggingface_hub import snapshot_download

    out = []
    for mid, (lic, status, role) in MODEL_IDS.items():
        dest = MODELS / mid.replace("/", "__")
        print(f"  model  {mid}")
        snapshot_download(mid, local_dir=dest, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "merges.txt", "vocab*", "1_Pooling/*", "modules.json"])
        out.append({"model": mid, "path": str(dest.relative_to(ROOT)), "licence": lic, "status": status, "role": role})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-models", action="store_true")
    args = ap.parse_args()

    print("Datasets and lexicons:")
    manifest = {"sources": fetch_sources()}
    if not args.no_models:
        print("Models:")
        manifest["models"] = fetch_models()
    manifest["manual"] = [{"name": n, "url": u, "instructions": i} for n, u, i in MANUAL]
    (RAW / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print("\nManual downloads still needed:")
    for n, u, i in MANUAL:
        present = (RAW / n.split()[0].lower()).exists()
        print(f"  [{'x' if present else ' '}] {n}: {u}\n      {i}")


if __name__ == "__main__":
    main()
