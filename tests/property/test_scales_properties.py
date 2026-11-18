"""Property tests for VAD scale conversions."""

from hypothesis import given
from hypothesis import strategies as st

from engine.mappings import scales as S


@st.composite
def native_value(draw: st.DrawFn) -> tuple[str, float]:
    name = draw(st.sampled_from(sorted(S.SCALES)))
    sc = S.SCALES[name]
    return name, draw(st.floats(min_value=sc.low, max_value=sc.high, allow_nan=False))


@st.composite
def native_pair(draw: st.DrawFn) -> tuple[str, float, float]:
    name = draw(st.sampled_from(sorted(S.SCALES)))
    sc = S.SCALES[name]
    vals = st.floats(min_value=sc.low, max_value=sc.high, allow_nan=False)
    a, b = draw(vals), draw(vals)
    return name, min(a, b), max(a, b)


unit = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False)


@given(native_value())
def test_output_in_unit_range(nv: tuple[str, float]) -> None:
    name, x = nv
    assert -1.0 <= S.to_unit(x, name) <= 1.0


@given(st.sampled_from(sorted(S.SCALES)), unit)
def test_inverse_output_in_native_range(name: str, y: float) -> None:
    sc = S.SCALES[name]
    assert sc.low <= S.from_unit(y, name) <= sc.high


@given(native_pair())
def test_monotonic(pair: tuple[str, float, float]) -> None:
    name, a, b = pair
    assert S.to_unit(a, name) <= S.to_unit(b, name)
    assert S.from_unit(S.to_unit(a, name), name) <= S.from_unit(S.to_unit(b, name), name)


@given(native_value())
def test_round_trip_native(nv: tuple[str, float]) -> None:
    name, x = nv
    assert abs(S.from_unit(S.to_unit(x, name), name) - x) <= 1e-9


@given(st.sampled_from(sorted(S.SCALES)), unit)
def test_round_trip_unit(name: str, y: float) -> None:
    assert abs(S.to_unit(S.from_unit(y, name), name) - y) <= 1e-9
