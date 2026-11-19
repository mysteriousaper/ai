import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from eval import metrics as M

finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-100, max_value=100)
vectors = st.lists(finite, min_size=3, max_size=40)


def _pair(draw_x, draw_y):
    x, y = np.array(draw_x), np.array(draw_y)
    n = min(len(x), len(y))
    return x[:n], y[:n]


@given(vectors)
def test_ccc_self_is_one(xs):
    x = np.array(xs)
    assume(x.var() > 1e-9)
    assert M.ccc(x, x) == pytest.approx(1.0)


@given(vectors, vectors)
def test_pearson_in_range(xs, ys):
    x, y = _pair(xs, ys)
    assume(x.var() > 1e-9 and y.var() > 1e-9)
    r = M.pearson(x, y)
    assert -1.0 <= r <= 1.0


@given(vectors, vectors)
def test_ccc_bounded_by_abs_pearson(xs, ys):
    x, y = _pair(xs, ys)
    assume(x.var() > 1e-9 and y.var() > 1e-9)
    assert M.ccc(x, y) <= abs(M.pearson(x, y)) + 1e-9


@given(vectors)
def test_mae_zero_on_identity(xs):
    x = np.array(xs)
    assert M.mae(x, x) == 0.0


@settings(max_examples=30, deadline=None)
@given(vectors)
def test_bootstrap_point_inside_interval(xs):
    x = np.array(xs)
    point, lo, hi = M.bootstrap_ci(lambda a: float(np.mean(a)), x, n=200, seed=1)
    assert lo <= point <= hi


def test_bootstrap_is_deterministic_for_seed():
    x = np.arange(50, dtype=float)
    assert M.bootstrap_ci(lambda a: float(a.mean()), x, n=100, seed=3) == M.bootstrap_ci(
        lambda a: float(a.mean()), x, n=100, seed=3
    )


@given(st.lists(st.integers(0, 2), min_size=5, max_size=60))
def test_single_label_f1_in_range(ys):
    labels = [0, 1, 2]
    preds = list(reversed(ys))
    score = M.f1_single(ys, preds, labels)
    assert 0.0 <= score <= 1.0


@given(st.lists(st.integers(0, 2), min_size=1, max_size=60))
def test_single_label_perfect_is_one(ys):
    assume(set(ys) == {0, 1, 2})
    assert M.f1_single(ys, ys, [0, 1, 2]) == pytest.approx(1.0)


@settings(deadline=None, max_examples=50)
@given(st.integers(4, 40), st.integers(2, 5), st.integers(0, 10_000))
def test_multilabel_f1_in_range_and_perfect(n, k, seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=(n, k))
    y_prob = rng.random((n, k))
    labels = [f"l{j}" for j in range(k)]
    rep = M.multilabel_report(y_true, y_prob, labels)
    assert 0.0 <= rep["macro_f1"] <= 1.0
    assert 0.0 <= rep["micro_f1"] <= 1.0
    assert set(rep["per_label_f1"]) == set(labels)

    perfect = M.multilabel_report(y_true, y_true.astype(float), labels)
    if y_true.sum() > 0:
        assert perfect["micro_f1"] == pytest.approx(1.0)
    for lab, j in zip(labels, range(k)):
        if y_true[:, j].sum() > 0:
            assert perfect["per_label_f1"][lab] == pytest.approx(1.0)


def test_multilabel_perfect_macro_f1_is_one():
    y = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]])
    rep = M.multilabel_report(y, y.astype(float), ["a", "b", "c"])
    assert rep["macro_f1"] == pytest.approx(1.0)
    assert rep["roc_auc_macro"] == pytest.approx(1.0)
