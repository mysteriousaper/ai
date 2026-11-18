"""Canonical discrete label set and its coarse views (brief §4.5).

Canonical set: the 27 GoEmotions emotions plus ``neutral`` (Demszky et al. 2020,
"GoEmotions: A Dataset of Fine-Grained Emotions", ACL 2020, arXiv:2005.00547).

Coarse views:

* **Ekman (6)**: the official GoEmotions grouping, copied verbatim.
* **Plutchik (8 primaries)**: our own proposal (no official table exists), built on
  Plutchik's wheel and its dyads (Plutchik 1980, *Emotion: A Psychoevolutionary Synthesis*;
  Plutchik 2001, "The Nature of Emotions", *American Scientist* 89(4):344-350).

Aggregation is **noisy-OR**: ``P(coarse) = 1 - prod(1 - p_i)`` over the fine labels mapped to
that coarse label. GoEmotions is multi-label, so the fine probabilities are independent
"is this emotion present" scores, not a distribution that sums to 1; noisy-OR keeps the result in
[0, 1], is monotone in every input and is never below its strongest member. It does assume the
member detections are independent, which over-counts correlated labels (e.g. anger and annoyance
usually fire together); fusion calibration (brief §4.6) has to correct for that.

``neutral`` maps to no coarse category: it stays visible in the fine-grained list only.
"""

from collections.abc import Mapping
from math import expm1, fsum, isfinite, log1p
from numbers import Real

# --- Pinned upstream data ---------------------------------------------------------------------
# Source: https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/emotions.txt
# Fetched 2026-11-18, sha256 45c3ef86782d2a4d7fedcd6d8c111aa0d0e94720689bd164fac94fefb4495a89.
# Order = class id used in the dataset's TSV files. Licence: Apache-2.0 (google-research repo).
GOEMOTIONS_LABELS: tuple[str, ...] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)

NEUTRAL = "neutral"

# Source: https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/ekman_mapping.json
# Fetched 2026-11-18, sha256 d6b1fea382917c1685c42e1324a0344106be9390f52a3850ce032656d103a328.
# Copied verbatim (keys and member order as in the file). ``neutral`` is not in the file.
EKMAN_MAP: dict[str, tuple[str, ...]] = {
    "anger": ("anger", "annoyance", "disapproval"),
    "disgust": ("disgust",),
    "fear": ("fear", "nervousness"),
    "joy": (
        "joy",
        "amusement",
        "approval",
        "excitement",
        "gratitude",
        "love",
        "optimism",
        "relief",
        "pride",
        "admiration",
        "desire",
        "caring",
    ),
    "sadness": ("sadness", "disappointment", "embarrassment", "grief", "remorse"),
    "surprise": ("surprise", "realization", "confusion", "curiosity"),
}
EKMAN_LABELS: tuple[str, ...] = tuple(EKMAN_MAP)

# --- Plutchik: our proposal -------------------------------------------------------------------
# Order follows the wheel (Plutchik 2001, Fig. "wheel of emotions").
PLUTCHIK_LABELS: tuple[str, ...] = (
    "joy",
    "trust",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
)

# GoEmotions label -> Plutchik primaries. A label maps to two primaries where Plutchik names it
# as a dyad (a blend of two adjacent or near primaries); under noisy-OR it then raises both.
_PLUTCHIK_BY_LABEL: dict[str, tuple[str, ...]] = {
    "admiration": ("trust",),  # Wheel: admiration is the intense form of trust.
    "amusement": ("joy",),  # approximate: not on the wheel; a mild, playful joy.
    "anger": ("anger",),  # Primary.
    "annoyance": ("anger",),  # Wheel: annoyance is the mild form of anger.
    "approval": ("trust",),  # approximate: wheel's "acceptance" (mild trust) is the nearest.
    "caring": ("trust", "joy"),  # approximate: warmth toward others; nearest is "love" dyad.
    "confusion": ("surprise",),  # approximate: same choice as the official Ekman map.
    "curiosity": ("trust", "surprise"),  # Plutchik secondary dyad: curiosity = trust + surprise.
    "desire": ("anticipation",),  # approximate: wanting points to the future (interest).
    "disappointment": ("surprise", "sadness"),  # Plutchik primary dyad surprise + sadness.
    "disapproval": ("disgust", "anger"),  # approximate: see rationale (name clash with Plutchik).
    "disgust": ("disgust",),  # Primary.
    "embarrassment": ("fear", "disgust"),  # approximate: Plutchik's "shame" = fear + disgust.
    "excitement": ("joy", "anticipation"),  # approximate: aroused, forward-looking joy.
    "fear": ("fear",),  # Primary.
    "gratitude": ("joy", "trust"),  # approximate: pleasure plus trust in a benefactor.
    "grief": ("sadness",),  # Wheel: grief is the intense form of sadness.
    "joy": ("joy",),  # Primary.
    "love": ("joy", "trust"),  # Plutchik primary dyad: love = joy + trust.
    "nervousness": ("fear", "anticipation"),  # Plutchik dyad: anxiety = anticipation + fear.
    "optimism": ("anticipation", "joy"),  # Plutchik primary dyad: optimism = anticipation + joy.
    "pride": ("joy",),  # approximate: Plutchik's "pride" dyad is anger + joy; see rationale.
    "realization": ("surprise",),  # approximate: same choice as the official Ekman map.
    "relief": ("joy",),  # approximate: pleasant end of a threat; not on the wheel.
    "remorse": ("sadness", "disgust"),  # Plutchik primary dyad: remorse = sadness + disgust.
    "sadness": ("sadness",),  # Primary.
    "surprise": ("surprise",),  # Primary.
}

PLUTCHIK_RATIONALE: dict[str, str] = {
    "admiration": "Plutchik's wheel places admiration as the high-intensity form of trust.",
    "amusement": "Not on the wheel; amusement is a light, playful variety of joy.",
    "anger": "Plutchik primary emotion.",
    "annoyance": "Plutchik's wheel places annoyance as the low-intensity form of anger.",
    "approval": "Nearest wheel term is acceptance, the low-intensity form of trust.",
    "caring": "Warm concern for others; closest to the love dyad (joy + trust).",
    "confusion": "Unexpected, not-yet-understood input; follows the official Ekman map.",
    "curiosity": "Plutchik lists curiosity as the secondary dyad trust + surprise.",
    "desire": "Wanting something not yet had is forward-looking, i.e. anticipation.",
    "disappointment": "Plutchik's surprise + sadness dyad (named disapproval or "
    "disappointment in different versions of the wheel).",
    "disapproval": "GoEmotions defines it as an unfavourable opinion, which is closer to "
    "contempt (disgust + anger) than to Plutchik's own 'disapproval' (surprise + sadness).",
    "disgust": "Plutchik primary emotion.",
    "embarrassment": "Self-conscious exposure; Plutchik's shame dyad is fear + disgust.",
    "excitement": "High-arousal pleasure about what is coming: joy + anticipation.",
    "fear": "Plutchik primary emotion.",
    "gratitude": "Pleasure at a benefit plus trust in the giver: joy + trust.",
    "grief": "Plutchik's wheel places grief as the high-intensity form of sadness.",
    "joy": "Plutchik primary emotion.",
    "love": "Plutchik's primary dyad joy + trust.",
    "nervousness": "Plutchik's anxiety dyad is anticipation + fear.",
    "optimism": "Plutchik's primary dyad anticipation + joy.",
    "pride": "Plutchik's pride dyad is anger + joy, but GoEmotions pride is satisfaction in an "
    "achievement, with no anger; we keep joy only.",
    "realization": "Sudden understanding; follows the official Ekman map (surprise).",
    "relief": "Pleasant end of a threat or worry; a form of joy.",
    "remorse": "Plutchik's primary dyad sadness + disgust.",
    "sadness": "Plutchik primary emotion.",
    "surprise": "Plutchik primary emotion.",
}

PLUTCHIK_APPROXIMATE: frozenset[str] = frozenset(
    {
        "amusement",
        "approval",
        "caring",
        "confusion",
        "desire",
        "disapproval",
        "embarrassment",
        "excitement",
        "gratitude",
        "pride",
        "realization",
        "relief",
    }
)

PLUTCHIK_MAP: dict[str, tuple[str, ...]] = {
    primary: tuple(g for g in GOEMOTIONS_LABELS if primary in _PLUTCHIK_BY_LABEL.get(g, ()))
    for primary in PLUTCHIK_LABELS
}

_KNOWN = frozenset(GOEMOTIONS_LABELS)


class UnknownLabelError(ValueError):
    """A label outside the canonical GoEmotions set (labels are case-sensitive)."""


def _check(probs: Mapping[str, float]) -> None:
    for label, p in probs.items():
        if label not in _KNOWN:
            raise UnknownLabelError(f"unknown emotion label {label!r}")
        if isinstance(p, bool) or not isinstance(p, Real):
            raise TypeError(f"probability for {label!r} must be a real number")
        if not (isfinite(p) and 0.0 <= p <= 1.0):
            raise ValueError(f"probability for {label!r} must be in [0, 1], got {p!r}")


def _noisy_or(probs: Mapping[str, float], table: Mapping[str, tuple[str, ...]]) -> dict[str, float]:
    _check(probs)
    out: dict[str, float] = {}
    for coarse, members in table.items():
        ps = [float(probs.get(m, 0.0)) for m in members]
        strongest = max(ps, default=0.0)
        if strongest >= 1.0:
            out[coarse] = 1.0
            continue
        # 1 - prod(1 - p) computed as -expm1(sum(log1p(-p))): plain 1 - (1 - p) loses the low
        # bits (0.1 -> 0.09999999999999998, 1e-17 -> 0). The floor keeps the documented
        # "never below its strongest member" exact under rounding.
        out[coarse] = min(1.0, max(strongest, -expm1(fsum(log1p(-p) for p in ps))))
    return out


def to_ekman(probs: Mapping[str, float]) -> dict[str, float]:
    """Aggregate canonical-label probabilities to Ekman's 6 by noisy-OR.

    Missing labels count as 0. Raises :class:`UnknownLabelError` for labels outside
    :data:`GOEMOTIONS_LABELS` and ``ValueError`` for probabilities outside [0, 1] or NaN.
    """
    return _noisy_or(probs, EKMAN_MAP)


def to_plutchik(probs: Mapping[str, float]) -> dict[str, float]:
    """Aggregate canonical-label probabilities to Plutchik's 8 primaries by noisy-OR."""
    return _noisy_or(probs, PLUTCHIK_MAP)
