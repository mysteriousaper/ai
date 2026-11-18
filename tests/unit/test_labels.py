"""Label tables and coarse aggregation (brief §4.5)."""

import math
from collections.abc import Callable, Mapping

import pytest

from engine.mappings import labels as L

Aggregate = Callable[[Mapping[str, float]], dict[str, float]]

OFFICIAL_EMOTIONS_TXT = [
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
]


def test_goemotions_labels_match_official_order() -> None:
    assert list(L.GOEMOTIONS_LABELS) == OFFICIAL_EMOTIONS_TXT
    assert len(L.GOEMOTIONS_LABELS) == 28
    assert len(set(L.GOEMOTIONS_LABELS)) == 28


def test_ekman_map_matches_official_file() -> None:
    assert L.EKMAN_MAP == {
        "anger": ("anger", "annoyance", "disapproval"),
        "disgust": ("disgust",),
        "fear": ("fear", "nervousness"),
        "joy": (
            "joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism",
            "relief", "pride", "admiration", "desire", "caring",
        ),
        "sadness": ("sadness", "disappointment", "embarrassment", "grief", "remorse"),
        "surprise": ("surprise", "realization", "confusion", "curiosity"),
    }
    assert tuple(L.EKMAN_MAP) == L.EKMAN_LABELS


def test_ekman_covers_every_non_neutral_label_exactly_once() -> None:
    mapped = [g for fine in L.EKMAN_MAP.values() for g in fine]
    assert sorted(mapped) == sorted(x for x in L.GOEMOTIONS_LABELS if x != "neutral")


def test_plutchik_has_eight_primaries_and_covers_every_non_neutral_label() -> None:
    assert L.PLUTCHIK_LABELS == (
        "joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation",
    )
    assert tuple(L.PLUTCHIK_MAP) == L.PLUTCHIK_LABELS
    mapped = {g for fine in L.PLUTCHIK_MAP.values() for g in fine}
    assert mapped == set(L.GOEMOTIONS_LABELS) - {"neutral"}
    for fine in L.PLUTCHIK_MAP.values():
        assert fine, "every primary receives at least one label"
        assert set(fine) <= set(L.GOEMOTIONS_LABELS)


def test_plutchik_rationale_for_every_label() -> None:
    assert set(L.PLUTCHIK_RATIONALE) == set(L.GOEMOTIONS_LABELS) - {"neutral"}
    assert all(len(r) > 10 for r in L.PLUTCHIK_RATIONALE.values())
    assert L.PLUTCHIK_APPROXIMATE <= set(L.PLUTCHIK_RATIONALE)


def test_neutral_maps_to_nothing() -> None:
    assert L.to_ekman({"neutral": 1.0}) == dict.fromkeys(L.EKMAN_LABELS, 0.0)
    assert L.to_plutchik({"neutral": 1.0}) == dict.fromkeys(L.PLUTCHIK_LABELS, 0.0)


def test_noisy_or_values() -> None:
    out = L.to_ekman({"anger": 0.5, "annoyance": 0.5, "joy": 0.2})
    assert out["anger"] == pytest.approx(0.75)
    assert out["joy"] == pytest.approx(0.2)
    assert out["fear"] == 0.0
    # love is a Plutchik dyad: joy + trust.
    p = L.to_plutchik({"love": 0.6})
    assert p["joy"] == pytest.approx(0.6) and p["trust"] == pytest.approx(0.6)


def test_empty_input_gives_zeros_with_all_keys() -> None:
    assert L.to_ekman({}) == dict.fromkeys(L.EKMAN_LABELS, 0.0)
    assert list(L.to_plutchik({})) == list(L.PLUTCHIK_LABELS)


@pytest.mark.parametrize("fn", [L.to_ekman, L.to_plutchik])
def test_unknown_label_raises(fn: Aggregate) -> None:
    with pytest.raises(L.UnknownLabelError):
        fn({"happiness": 0.3})
    with pytest.raises(ValueError):  # UnknownLabelError is a ValueError
        fn({"Joy": 0.3})


@pytest.mark.parametrize("fn", [L.to_ekman, L.to_plutchik])
@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf])
def test_bad_probability_raises(fn: Aggregate, bad: float) -> None:
    with pytest.raises(ValueError):
        fn({"joy": bad})


@pytest.mark.parametrize("fn", [L.to_ekman, L.to_plutchik])
def test_bool_probability_rejected(fn: Aggregate) -> None:
    with pytest.raises(TypeError):
        fn({"joy": True})
