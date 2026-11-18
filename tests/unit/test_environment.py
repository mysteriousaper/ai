"""Phase 0 smoke test: every core dependency imports and the runtime is what we expect."""
import importlib
import sys

import pytest

CORE = [
    "torch", "transformers", "datasets", "accelerate", "sentence_transformers",
    "sklearn", "scipy", "numpy", "pandas", "spacy", "hdbscan", "umap",
    "pydantic", "fastapi", "uvicorn", "vaderSentiment", "emoji", "langdetect",
    "trackio", "hypothesis",
]


def test_python_version():
    assert sys.version_info[:2] == (3, 11)


@pytest.mark.parametrize("module", CORE)
def test_import(module):
    importlib.import_module(module)


def test_spacy_model_loads():
    import spacy

    doc = spacy.load("en_core_web_sm")("I am not happy today.")
    assert any(t.dep_ == "neg" for t in doc)


def test_torch_cpu_works():
    import torch

    x = torch.randn(4, 4)
    assert torch.allclose(x @ torch.eye(4), x)


def test_vader_baseline_negation():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    assert sia.polarity_scores("I am happy")["compound"] > 0
    assert sia.polarity_scores("I am not happy")["compound"] < 0
