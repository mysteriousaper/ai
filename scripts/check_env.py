"""Print a Markdown environment report: hardware, key package versions, smoke-test timings.

Usage:
    uv run python scripts/check_env.py            # includes the GoEmotions model download
    uv run python scripts/check_env.py --offline  # skip anything that needs the network
"""

import argparse
import importlib.metadata as md
import os
import platform
import sys
import time

import engine  # noqa: F401  (applies privacy defaults before other imports)

PACKAGES = [
    "torch", "transformers", "datasets", "accelerate", "sentence-transformers",
    "scikit-learn", "scipy", "numpy", "pandas", "umap-learn", "spacy", "en-core-web-sm",
    "emoji", "lingua-language-detector", "vadersentiment", "pydantic", "fastapi",
    "uvicorn", "mlflow", "pytest", "hypothesis",
]
GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - start) * 1000


def hardware() -> list[tuple[str, str]]:
    import torch

    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    mps = getattr(torch.backends, "mps", None)
    return [
        ("Python", sys.version.split()[0]),
        ("Platform", platform.platform()),
        ("CPU cores", str(os.cpu_count())),
        ("RAM", f"{ram_gb:.1f} GB"),
        ("CUDA GPU", gpu),
        ("Apple MPS", str(bool(mps and mps.is_available()))),
        ("torch threads", str(torch.get_num_threads())),
    ]


def smoke_tests(offline: bool) -> list[tuple[str, str, float]]:
    rows = []

    import spacy

    nlp, ms = timed(lambda: spacy.load("en_core_web_sm"))
    rows.append(("spaCy load en_core_web_sm", "ok", ms))
    doc, ms = timed(lambda: nlp("I wasn't happy at all yesterday."))
    rows.append(("spaCy parse 1 sentence", " ".join(t.lemma_ for t in doc), ms))

    from lingua import Language, LanguageDetectorBuilder

    det = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.RUSSIAN, Language.ARMENIAN
    ).build()
    lang, ms = timed(lambda: det.detect_language_of("Ես շատ հոգնած եմ"))
    rows.append(("lingua LID (Armenian sample)", str(lang), ms))

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    score, ms = timed(lambda: sia.polarity_scores("I am not happy")["compound"])
    rows.append(("VADER 'I am not happy'", f"compound={score:+.3f}", ms))

    if not offline:
        from transformers import pipeline

        clf, ms = timed(
            lambda: pipeline("text-classification", model=GOEMOTIONS_MODEL, top_k=3)
        )
        rows.append((f"load {GOEMOTIONS_MODEL}", "ok", ms))
        text = "I miss my dad so much, it hurts"
        clf(text)  # warm-up
        out, ms = timed(lambda: clf(text)[0])
        top = ", ".join(f"{o['label']} {o['score']:.2f}" for o in out)
        rows.append((f"GoEmotions on '{text}'", top, ms))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    print("## Hardware\n\n| Item | Value |\n|---|---|")
    for k, v in hardware():
        print(f"| {k} | {v} |")

    print("\n## Packages\n\n| Package | Version |\n|---|---|")
    for p in PACKAGES:
        try:
            print(f"| {p} | {md.version(p)} |")
        except md.PackageNotFoundError:
            print(f"| {p} | **MISSING** |")

    print("\n## Smoke tests (CPU)\n\n| Check | Result | ms |\n|---|---|---|")
    for name, result, ms in smoke_tests(args.offline):
        print(f"| {name} | {result} | {ms:.0f} |")


if __name__ == "__main__":
    main()
