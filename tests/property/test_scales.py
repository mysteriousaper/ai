import pytest
from hypothesis import given
from hypothesis import strategies as st

from engine.mappings import labels as L
from engine.mappings import scales as S

finite = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6)


@pytest.mark.parametrize(("fn", "lo", "mid", "hi"), [
    (S.warriner, 1, 5, 9), (S.sam, 1, 5, 9), (S.emobank, 1, 3, 5), (S.nrc_v1, 0, 0.5, 1), (S.nrc_v2, -1, 0, 1),
])
def test_anchor_points(fn, lo, mid, hi):
    assert fn(lo) == pytest.approx(-1)
    assert fn(mid) == pytest.approx(0)
    assert fn(hi) == pytest.approx(1)


@given(finite)
def test_all_converters_stay_in_range(x):
    for fn in S.CONVERTERS.values():
        assert -1.0 <= fn(x) <= 1.0


@given(st.floats(min_value=1, max_value=9), st.floats(min_value=1, max_value=9))
def test_monotonic(a, b):
    if a <= b:
        assert S.warriner(a) <= S.warriner(b)


@given(st.floats(min_value=1, max_value=9))
def test_sam_roundtrip(x):
    assert S.common_to_sam(S.sam(x)) == pytest.approx(x, abs=1e-9)


@given(st.floats(min_value=0, max_value=1))
def test_unit_roundtrip(x):
    assert S.common_to_unit(S.unit_to_common(x)) == pytest.approx(x, abs=1e-9)


def test_bad_range_rejected():
    with pytest.raises(ValueError):
        S.linear_to_common(1, 5, 5)


@given(st.dictionaries(st.sampled_from(L.CANONICAL), st.floats(min_value=0, max_value=1)))
def test_projection_keeps_probabilities_valid(probs):
    for out, targets in ((L.to_ekman(probs), L.EKMAN), (L.to_plutchik(probs), L.PLUTCHIK)):
        assert set(out) == set(targets)
        assert all(0.0 <= v <= 1.0 for v in out.values())
        assert max(out.values(), default=0) <= max(probs.values(), default=0) + 1e-12
