"""Brief §10 hard rule: with safety_flag set, no adapter suggestion is ever produced."""

from datetime import UTC, datetime
from typing import Any

import pytest

from engine.adapters.base import (
    Adapter,
    DomainSuggestion,
    SafetyResponse,
    SuggestionItem,
    UserProfile,
)
from engine.schemas import EmotionalStateObject


def make_state(flag: bool, reason: str | None = None) -> EmotionalStateObject:
    return EmotionalStateObject.model_validate(
        {
            "timestamp": datetime(2026, 11, 18, tzinfo=UTC),
            "language": "en",
            "vad": {"valence": -0.8, "arousal": -0.5, "dominance": -0.7, "std": [0.1, 0.1, 0.1]},
            "emotions": [{"label": "sadness", "p": 0.9}],
            "coarse": {"plutchik": {"sadness": 0.9}, "ekman": {"sadness": 0.9}},
            "intensity": 0.8,
            "situation": {
                "event_type": "loss_or_failure",
                "agent": "circumstance",
                "goal_congruence": -0.9,
                "controllability": -0.8,
                "certainty": 0.5,
                "evidence_spans": [],
            },
            "need": [{"type": "rest", "p": 0.7}],
            "pattern": {"baseline_deviation": -2.0, "trend": "worsening", "volatility": 0.3},
            "overall_confidence": 0.6,
            "mixed_or_ambiguous": False,
            "explanations": [],
            "engine_versions": {},
            "safety_flag": flag,
            "safety_reason": reason,
        }
    )


class SpyAdapter(Adapter):
    name = "spy"
    version = "0.0.1"

    def __init__(self) -> None:
        self.calls = 0

    def adapt(self, state: EmotionalStateObject, profile: UserProfile) -> DomainSuggestion:
        self.calls += 1
        return DomainSuggestion(
            adapter=self.name,
            adapter_version=self.version,
            items=[SuggestionItem(id="beach", title="A quiet beach", score=0.8, rationale="calm")],
        )


PROFILE = UserProfile(user_id="u1", region="AM")


def test_unflagged_state_goes_to_adapt() -> None:
    a = SpyAdapter()
    out = a.suggest(make_state(False), PROFILE)
    assert isinstance(out, DomainSuggestion)
    assert a.calls == 1


def test_flagged_state_returns_safety_response_and_never_calls_adapt() -> None:
    a = SpyAdapter()
    out = a.suggest(make_state(True, "self_harm_explicit"), PROFILE)
    assert isinstance(out, SafetyResponse)
    assert not isinstance(out, DomainSuggestion)
    assert a.calls == 0
    assert out.show_human_help is True
    assert out.reason == "self_harm_explicit"
    assert out.message


def test_safety_response_has_no_suggestion_fields() -> None:
    out = SpyAdapter().suggest(make_state(True), PROFILE)
    assert "items" not in out.model_dump()


def test_adapt_raising_when_called_proves_it_is_skipped() -> None:
    class Exploding(Adapter):
        name = "boom"
        version = "1"

        def adapt(self, state: EmotionalStateObject, profile: UserProfile) -> DomainSuggestion:
            raise AssertionError("adapt must not run for a flagged state")

    assert isinstance(Exploding().suggest(make_state(True), PROFILE), SafetyResponse)


def test_subclass_cannot_override_suggest() -> None:
    with pytest.raises(TypeError, match="suggest"):

        class Sneaky(SpyAdapter):
            def suggest(self, state: Any, profile: Any) -> Any:  # type: ignore[misc]
                return self.adapt(state, profile)


def test_mixin_cannot_override_suggest() -> None:
    class Mixin:
        def suggest(self, state: Any, profile: Any) -> Any:
            return "bypass"

    with pytest.raises(TypeError, match="suggest"):

        class Sneaky(Mixin, SpyAdapter):  # type: ignore[misc]
            pass


def test_subclass_cannot_override_getattribute() -> None:
    with pytest.raises(TypeError):

        class Sneaky(SpyAdapter):
            def __getattribute__(self, name: str) -> Any:
                return super().__getattribute__(name)


def test_suggest_cannot_be_patched_on_class_or_instance() -> None:
    with pytest.raises(TypeError):
        SpyAdapter.suggest = lambda self, s, p: "bypass"  # type: ignore[method-assign]
    with pytest.raises(TypeError):
        Adapter.suggest = lambda self, s, p: "bypass"  # type: ignore[method-assign]
    a = SpyAdapter()
    with pytest.raises(AttributeError):
        a.suggest = lambda s, p: "bypass"  # type: ignore[method-assign]
    vars(a)["suggest"] = lambda s, p: "bypass"  # even a raw __dict__ write must not win
    assert isinstance(a.suggest(make_state(True), PROFILE), SafetyResponse)


def test_suggest_cannot_be_deleted() -> None:
    with pytest.raises(TypeError):
        del SpyAdapter.suggest
    with pytest.raises(AttributeError):
        del SpyAdapter().suggest


def test_adapt_must_return_domain_suggestion() -> None:
    class Wrong(Adapter):
        name = "wrong"
        version = "1"

        def adapt(self, state: EmotionalStateObject, profile: UserProfile) -> DomainSuggestion:
            return {"items": []}  # type: ignore[return-value]

    with pytest.raises(TypeError):
        Wrong().suggest(make_state(False), PROFILE)


def test_suggest_requires_a_real_state_object() -> None:
    with pytest.raises(TypeError):
        SpyAdapter().suggest({"safety_flag": True}, PROFILE)  # type: ignore[arg-type]


def test_flag_flipped_after_validation_is_still_honoured() -> None:
    state = make_state(False)
    state.safety_flag = True  # validate_assignment keeps this a real bool
    a = SpyAdapter()
    assert isinstance(a.suggest(state, PROFILE), SafetyResponse)
    assert a.calls == 0


def test_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        Adapter()  # type: ignore[abstract]


def test_unvalidated_truthy_flag_fails_safe() -> None:
    """model_construct skips validation; any value other than False counts as flagged."""
    state = make_state(False)
    for odd in (1, "yes", None):
        raw = EmotionalStateObject.model_construct(**(state.__dict__ | {"safety_flag": odd}))
        a = SpyAdapter()
        assert isinstance(a.suggest(raw, PROFILE), SafetyResponse)
        assert a.calls == 0
