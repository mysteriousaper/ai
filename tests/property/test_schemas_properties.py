"""Property tests for the Emotional State Object's ordering rules."""

from datetime import UTC, datetime

from hypothesis import given
from hypothesis import strategies as st

from engine import schemas as S

prob = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def _minimal(emotions: list[dict[str, object]], need: list[dict[str, object]]) -> dict:
    return {
        "timestamp": datetime(2026, 11, 18, tzinfo=UTC),
        "language": None,
        "vad": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0, "std": [1.0, 1.0, 1.0]},
        "emotions": emotions,
        "coarse": {"plutchik": {}, "ekman": {}},
        "intensity": 0.0,
        "situation": {
            "event_type": "other_or_unclear",
            "agent": "unknown",
            "goal_congruence": None,
            "controllability": None,
            "certainty": None,
            "evidence_spans": [],
        },
        "need": need,
        "pattern": {"baseline_deviation": None, "trend": "unknown", "volatility": 0.0},
        "overall_confidence": 0.0,
        "mixed_or_ambiguous": True,
        "explanations": [],
        "engine_versions": {},
    }


@given(
    st.dictionaries(st.sampled_from(list(S.EmotionLabel)), prob),
    st.dictionaries(st.sampled_from(list(S.NeedType)), prob),
)
def test_lists_always_sorted_and_round_trip(
    emo: dict[S.EmotionLabel, float], need: dict[S.NeedType, float]
) -> None:
    eso = S.EmotionalStateObject.model_validate(
        _minimal(
            [{"label": k, "p": v} for k, v in emo.items()],
            [{"type": k, "p": v} for k, v in need.items()],
        )
    )
    ps = [e.p for e in eso.emotions]
    assert ps == sorted(ps, reverse=True)
    assert {e.label for e in eso.emotions} == set(emo)
    nps = [n.p for n in eso.need]
    assert nps == sorted(nps, reverse=True)
    assert S.EmotionalStateObject.model_validate_json(eso.model_dump_json()) == eso
