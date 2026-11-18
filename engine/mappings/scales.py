"""Conversions from every source's native affect scale to the Engine's common [-1, 1] scale.

All forward functions are linear and clip to [-1, 1] (inputs slightly outside the documented
range happen with averaged ratings). Inverses exist for reporting in the source's own units.
"""

from __future__ import annotations


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def linear_to_common(x: float, lo: float, hi: float) -> float:
    """Map [lo, hi] linearly to [-1, 1]."""
    if hi <= lo:
        raise ValueError("hi must be greater than lo")
    return _clip(2.0 * (x - lo) / (hi - lo) - 1.0)


def common_to_linear(y: float, lo: float, hi: float) -> float:
    """Inverse of linear_to_common."""
    return lo + (_clip(y) + 1.0) * (hi - lo) / 2.0


# Named converters, one per source. Keep these explicit: each is unit-tested.
def warriner(x: float) -> float:  # Warriner et al. 2013, 1–9
    return linear_to_common(x, 1.0, 9.0)


def sam(x: float) -> float:  # Self-Assessment Manikin, 1–9 (Phase 5 self-reports)
    return linear_to_common(x, 1.0, 9.0)


def emobank(x: float) -> float:  # EmoBank, 1–5
    return linear_to_common(x, 1.0, 5.0)


def nrc_v1(x: float) -> float:  # NRC VAD v1 (2018), 0–1
    return linear_to_common(x, 0.0, 1.0)


def nrc_v2(x: float) -> float:  # NRC VAD v2.x (2025), already -1–1
    return _clip(x)


def unit_to_common(x: float) -> float:  # any [0, 1] score (e.g. SemEval V-reg)
    return linear_to_common(x, 0.0, 1.0)


def common_to_unit(y: float) -> float:
    return common_to_linear(y, 0.0, 1.0)


def common_to_sam(y: float) -> float:
    return common_to_linear(y, 1.0, 9.0)


CONVERTERS = {
    "warriner": warriner,
    "sam": sam,
    "emobank": emobank,
    "nrc_v1": nrc_v1,
    "nrc_v2": nrc_v2,
    "unit": unit_to_common,
}
