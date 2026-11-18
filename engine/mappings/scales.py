"""Conversions between each resource's native VAD scale and the common [-1, 1] scale (§4.5).

All conversions are linear: the native minimum maps to -1, the native midpoint to 0 and the
native maximum to 1. Out-of-range values, NaN and infinities raise ``ValueError``; we never
clip silently, because an out-of-range score means the wrong scale was picked upstream.

Native ranges and their primary sources:

* ``warriner``: 1..9. Warriner, Kuperman & Brysbaert 2013, "Norms of valence, arousal, and
  dominance for 13,915 English lemmas", Behavior Research Methods 45:1191-1207,
  doi:10.3758/s13428-012-0314-x (9-point rating scale).
* ``sam``: 1..9. The 9-point Self-Assessment Manikin as used for ANEW: Bradley & Lang 1994,
  "Measuring emotion: the Self-Assessment Manikin and the semantic differential", J. Behav.
  Ther. & Exp. Psychiatry 25(1):49-59; Bradley & Lang 1999 (ANEW technical report).
* ``emobank``: 1..5. Buechel & Hahn 2017, "EmoBank: Studying the Impact of Annotation
  Perspective and Representation Format on Dimensional Emotion Analysis", EACL 2017 (5-point
  SAM). Confirmed 2026-11-18 on https://github.com/JULIELab/EmoBank ``corpus/emobank.csv``:
  10,062 rows, observed V 1.2-4.6, A 1.8-4.4, D 1.78-4.2 (means of raters, so within 1..5).
* ``nrc_vad_v1``: 0..1. Mohammad 2018, "Obtaining Reliable Human Ratings of Valence, Arousal,
  and Dominance for 20,000 English Words", ACL 2018. https://saifmohammad.com/WebPages/nrc-vad.html
  ("the scores range from 0 (lowest V/A/D) to 1 (highest V/A/D)").
* ``nrc_vad_v2``: -1..1. Mohammad 2025, "NRC VAD Lexicon v2: Norms for Valence, Arousal, and
  Dominance for over 55k English Terms". Confirmed 2026-11-18 in the v2.1 zip's README.txt
  ("the scores range from -1 (lowest V/A/D) to 1"), and the data file's min/max are exactly
  -1.000/1.000 on all three dimensions. Identity conversion.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real


@dataclass(frozen=True, slots=True)
class Scale:
    """A closed native rating interval ``[low, high]``."""

    name: str
    low: float
    high: float
    source: str

    def __post_init__(self) -> None:
        if not (isfinite(self.low) and isfinite(self.high) and self.low < self.high):
            raise ValueError(f"invalid scale bounds for {self.name!r}")


SCALES: dict[str, Scale] = {
    s.name: s
    for s in (
        Scale("warriner", 1.0, 9.0, "Warriner et al. 2013, doi:10.3758/s13428-012-0314-x"),
        Scale("sam", 1.0, 9.0, "Bradley & Lang 1994, 9-point Self-Assessment Manikin"),
        Scale("emobank", 1.0, 5.0, "Buechel & Hahn 2017 (EACL), 5-point SAM"),
        Scale("nrc_vad_v1", 0.0, 1.0, "Mohammad 2018 (ACL), NRC VAD v1"),
        Scale("nrc_vad_v2", -1.0, 1.0, "Mohammad 2025, NRC VAD v2.1 README"),
    )
}


def _scale(name: str) -> Scale:
    try:
        return SCALES[name]
    except KeyError:
        raise ValueError(f"unknown scale {name!r}; known: {sorted(SCALES)}") from None


def _real(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"expected a real number, got {type(value).__name__}")
    v = float(value)
    if not isfinite(v):
        raise ValueError(f"value must be finite, got {v!r}")
    return v


def to_unit(value: float, scale: str) -> float:
    """Map a native score on ``scale`` to [-1, 1]."""
    sc = _scale(scale)
    v = _real(value)
    if not sc.low <= v <= sc.high:
        raise ValueError(f"{v!r} is outside the {scale} range [{sc.low}, {sc.high}]")
    # (2v - (low + high)) / (high - low): exact at the ends and the midpoint, identity for -1..1.
    y = (2.0 * v - (sc.low + sc.high)) / (sc.high - sc.low)
    return min(1.0, max(-1.0, y))


def from_unit(value: float, scale: str) -> float:
    """Map a [-1, 1] score back to the native range of ``scale`` (inverse of :func:`to_unit`)."""
    sc = _scale(scale)
    y = _real(value)
    if not -1.0 <= y <= 1.0:
        raise ValueError(f"{y!r} is outside [-1, 1]")
    v = (y * (sc.high - sc.low) + (sc.low + sc.high)) / 2.0
    return min(sc.high, max(sc.low, v))


def warriner_to_unit(value: float) -> float:
    """Warriner et al. 2013 (1..9) to [-1, 1]."""
    return to_unit(value, "warriner")


def unit_to_warriner(value: float) -> float:
    """[-1, 1] to Warriner et al. 2013 (1..9)."""
    return from_unit(value, "warriner")


def sam_to_unit(value: float) -> float:
    """9-point SAM (1..9) to [-1, 1]."""
    return to_unit(value, "sam")


def unit_to_sam(value: float) -> float:
    """[-1, 1] to 9-point SAM (1..9)."""
    return from_unit(value, "sam")


def emobank_to_unit(value: float) -> float:
    """EmoBank (1..5) to [-1, 1]."""
    return to_unit(value, "emobank")


def unit_to_emobank(value: float) -> float:
    """[-1, 1] to EmoBank (1..5)."""
    return from_unit(value, "emobank")


def nrc_vad_v1_to_unit(value: float) -> float:
    """NRC VAD v1 (0..1) to [-1, 1]."""
    return to_unit(value, "nrc_vad_v1")


def unit_to_nrc_vad_v1(value: float) -> float:
    """[-1, 1] to NRC VAD v1 (0..1)."""
    return from_unit(value, "nrc_vad_v1")


def nrc_vad_v2_to_unit(value: float) -> float:
    """NRC VAD v2 / v2.1 (-1..1) to [-1, 1] (identity, with validation)."""
    return to_unit(value, "nrc_vad_v2")


def unit_to_nrc_vad_v2(value: float) -> float:
    """[-1, 1] to NRC VAD v2 / v2.1 (-1..1) (identity, with validation)."""
    return from_unit(value, "nrc_vad_v2")
