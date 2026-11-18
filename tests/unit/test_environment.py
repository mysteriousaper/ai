"""Phase 0 environment checks: the toolchain works and privacy defaults hold."""

import importlib
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "sentence_transformers",
    "sklearn",
    "scipy",
    "numpy",
    "pandas",
    "umap",
    "spacy",
    "en_core_web_sm",
    "emoji",
    "lingua",
    "vaderSentiment.vaderSentiment",
    "pydantic",
    "fastapi",
    "uvicorn",
    "mlflow",
    "hypothesis",
]


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_required_module_imports(module):
    importlib.import_module(module)


def test_sklearn_has_builtin_hdbscan():
    from sklearn.cluster import HDBSCAN  # noqa: F401  (ADR-0004)


def test_spacy_english_model_lemmatises():
    import spacy

    nlp = spacy.load("en_core_web_sm")
    lemmas = [t.lemma_ for t in nlp("The cats were running")]
    assert "cat" in lemmas
    assert "run" in lemmas


def test_language_id_covers_english_russian_armenian():
    from lingua import Language, LanguageDetectorBuilder

    detector = LanguageDetectorBuilder.from_languages(
        Language.ENGLISH, Language.RUSSIAN, Language.ARMENIAN
    ).build()
    assert detector.detect_language_of("I am so tired of this week") == Language.ENGLISH
    assert detector.detect_language_of("Я очень устал на этой неделе") == Language.RUSSIAN
    assert detector.detect_language_of("Ես շատ հոգնած եմ այս շաբաթ") == Language.ARMENIAN


def test_vader_baseline_polarity_sign():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    assert sia.polarity_scores("I love this so much")["compound"] > 0
    assert sia.polarity_scores("I hate this so much")["compound"] < 0


@pytest.mark.parametrize(
    "path",
    [
        "data/raw/goemotions.csv",
        "data/processed/x.parquet",
        "mlruns/0/meta.yaml",
        "mlflow.db",
        ".env",
    ],
)
def test_sensitive_paths_are_gitignored(path):
    result = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT, check=False)
    assert result.returncode == 0, f"{path} must be git-ignored"


@pytest.mark.parametrize(
    "var,value",
    [
        ("HF_HUB_DISABLE_TELEMETRY", "1"),
        ("MLFLOW_DISABLE_TELEMETRY", "true"),
        ("DO_NOT_TRACK", "1"),
        ("GRADIO_ANALYTICS_ENABLED", "False"),
    ],
)
def test_engine_import_disables_third_party_telemetry(monkeypatch, var, value):
    monkeypatch.delenv(var, raising=False)
    import engine.privacy

    engine.privacy.disable_third_party_telemetry()
    assert os.environ[var] == value


def test_telemetry_defaults_respect_explicit_user_setting(monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    import engine.privacy

    engine.privacy.disable_third_party_telemetry()
    assert os.environ["DO_NOT_TRACK"] == "0"


@pytest.mark.network
@pytest.mark.slow
def test_pretrained_goemotions_model_downloads_and_runs():
    from transformers import pipeline

    clf = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    scores = clf("I miss my dad so much, it hurts")[0]
    top = max(scores, key=lambda s: s["score"])["label"]
    assert top in {"sadness", "grief", "remorse", "disappointment"}
