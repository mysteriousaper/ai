"""Property tests for coarse label aggregation."""

from hypothesis import given
from hypothesis import strategies as st

from engine.mappings import labels as L

prob = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
fine_probs = st.dictionaries(st.sampled_from(L.GOEMOTIONS_LABELS), prob)
FUNCS = (L.to_ekman, L.to_plutchik)


@given(fine_probs)
def test_outputs_in_unit_interval(probs: dict[str, float]) -> None:
    for fn in FUNCS:
        for v in fn(probs).values():
            assert 0.0 <= v <= 1.0


@given(fine_probs, st.sampled_from(L.GOEMOTIONS_LABELS), prob)
def test_monotone_in_every_input(probs: dict[str, float], label: str, bump: float) -> None:
    """Raising one fine-grained probability never lowers any coarse probability."""
    higher = dict(probs)
    higher[label] = max(probs.get(label, 0.0), bump)
    for fn in FUNCS:
        lo, hi = fn(probs), fn(higher)
        assert all(hi[k] >= lo[k] for k in lo)


@given(fine_probs)
def test_coarse_at_least_max_member(probs: dict[str, float]) -> None:
    """Noisy-OR is bounded below by its strongest member."""
    for table, fn in ((L.EKMAN_MAP, L.to_ekman), (L.PLUTCHIK_MAP, L.to_plutchik)):
        out = fn(probs)
        for coarse, members in table.items():
            strongest = max((probs.get(m, 0.0) for m in members), default=0.0)
            assert out[coarse] >= strongest - 1e-12
