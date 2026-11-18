"""Native VAD scales to the common [-1, 1] scale (brief §4.5)."""

import math

import pytest

from engine.mappings import scales as S

# (scale, native low, native mid, native high)
ENDPOINTS = [
    ("warriner", 1.0, 5.0, 9.0),
    ("sam", 1.0, 5.0, 9.0),
    ("emobank", 1.0, 3.0, 5.0),
    ("nrc_vad_v1", 0.0, 0.5, 1.0),
    ("nrc_vad_v2", -1.0, 0.0, 1.0),
]


def test_registry_has_every_scale() -> None:
    assert set(S.SCALES) == {name for name, *_ in ENDPOINTS}


@pytest.mark.parametrize(("name", "lo", "mid", "hi"), ENDPOINTS)
def test_endpoints_and_midpoint(name: str, lo: float, mid: float, hi: float) -> None:
    assert S.to_unit(lo, name) == -1.0
    assert S.to_unit(mid, name) == 0.0
    assert S.to_unit(hi, name) == 1.0
    assert S.from_unit(-1.0, name) == lo
    assert S.from_unit(0.0, name) == mid
    assert S.from_unit(1.0, name) == hi


def test_named_wrappers() -> None:
    assert S.warriner_to_unit(9) == 1.0
    assert S.unit_to_warriner(-1) == 1.0
    assert S.sam_to_unit(5) == 0.0
    assert S.unit_to_sam(1) == 9.0
    assert S.emobank_to_unit(4) == 0.5
    assert S.unit_to_emobank(0.5) == 4.0
    assert S.nrc_vad_v1_to_unit(0.75) == 0.5
    assert S.unit_to_nrc_vad_v1(0.5) == 0.75
    assert S.nrc_vad_v2_to_unit(-0.3) == -0.3
    assert S.unit_to_nrc_vad_v2(-0.3) == -0.3


def test_known_value_warriner_aardvark() -> None:
    # Warriner et al. 2013 rating for "aardvark": V.Mean.Sum = 6.26.
    assert S.warriner_to_unit(6.26) == pytest.approx(0.315)


@pytest.mark.parametrize(("name", "lo", "mid", "hi"), ENDPOINTS)
def test_out_of_range_raises(name: str, lo: float, mid: float, hi: float) -> None:
    for bad in (lo - 1e-6, hi + 1e-6, math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError):
            S.to_unit(bad, name)
    for bad in (-1.000001, 1.000001, math.nan, math.inf):
        with pytest.raises(ValueError):
            S.from_unit(bad, name)


def test_unknown_scale_raises() -> None:
    with pytest.raises(ValueError):
        S.to_unit(0.5, "anew")
    with pytest.raises(ValueError):
        S.from_unit(0.5, "anew")


def test_non_numbers_rejected() -> None:
    with pytest.raises(TypeError):
        S.to_unit(True, "nrc_vad_v1")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        S.to_unit("5", "warriner")  # type: ignore[arg-type]


def test_huge_int_raises_value_error_not_overflow() -> None:
    with pytest.raises(ValueError):
        S.to_unit(10**400, "warriner")
    with pytest.raises(ValueError):
        S.from_unit(10**400, "warriner")
