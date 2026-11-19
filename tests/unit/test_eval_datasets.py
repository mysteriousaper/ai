import numpy as np
import pytest

from engine.mappings.labels import CANONICAL
from eval import datasets as D


def _skip_if_missing(path):
    if not path.exists():
        pytest.skip(f"data file missing: {path}")


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_goemotions_shapes_and_labels(split):
    _skip_if_missing(D.GOEMOTIONS_DIR / "test.parquet")
    texts, y = D.load_goemotions(split)
    assert len(texts) == y.shape[0] > 0
    assert y.shape[1] == len(CANONICAL) == 28
    assert set(np.unique(y)) <= {0, 1}
    assert (y.sum(axis=1) >= 1).all()


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_emobank_shapes_and_ranges(split):
    _skip_if_missing(D.EMOBANK_CSV)
    texts, v, a, d = D.load_emobank(split)
    assert len(texts) == len(v) == len(a) == len(d) > 0
    assert all(isinstance(t, str) and t for t in texts)
    for arr in (v, a, d):
        assert arr.min() >= -1.0 and arr.max() <= 1.0


@pytest.mark.parametrize("emotion", D.SEMEVAL_EMOTIONS)
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_semeval_eireg_shapes_and_ranges(emotion, split):
    _skip_if_missing(D.SEMEVAL_EIREG_DIR)
    texts, y = D.load_semeval_eireg(emotion, split)
    assert len(texts) == len(y) > 0
    assert y.dtype.kind == "f"
    assert y.min() >= 0.0 and y.max() <= 1.0
