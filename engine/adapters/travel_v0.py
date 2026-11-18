"""Toy travel adapter: proves the ESO is enough to drive a domain. Rules, not ML."""

from __future__ import annotations

from typing import Any

from engine.adapters.base import Adapter, DomainSuggestion
from engine.mappings.labels import NeedType
from engine.schemas import EmotionalStateObject

ARCHETYPES: dict[str, dict[str, Any]] = {
    "calm_nature": {"examples": ["lake cabin", "mountain village", "quiet coastline"], "pace": "slow"},
    "gentle_culture": {"examples": ["small museum town", "botanical garden city"], "pace": "slow"},
    "social_city": {"examples": ["food-market city", "festival weekend"], "pace": "medium"},
    "adventure": {"examples": ["hiking trek", "surf camp", "road trip"], "pace": "fast"},
    "celebration": {"examples": ["beach resort", "nightlife city", "fine-dining trip"], "pace": "medium"},
}


def _need_p(state: EmotionalStateObject, need: NeedType) -> float:
    return max((n.p for n in state.need if n.type == need), default=0.0)


class TravelV0(Adapter):
    domain = "travel"

    def adapt(self, state: EmotionalStateObject, user_profile: dict[str, Any] | None = None) -> DomainSuggestion:
        v, a = state.vad.valence, state.vad.arousal
        scores = dict.fromkeys(ARCHETYPES, 0.0)
        why: list[str] = []

        if v < -0.2 and a < 0.0:
            scores["calm_nature"] += 1.0
            scores["gentle_culture"] += 0.5
            why.append("low valence + low arousal → restorative, low-stimulation places")
        if v < -0.2 and a >= 0.0:
            scores["calm_nature"] += 0.7
            scores["adventure"] += 0.3
            why.append("negative, activated state → calming or physically absorbing trips")
        if v >= 0.2 and a >= 0.2:
            scores["adventure"] += 0.8
            scores["social_city"] += 0.6
            why.append("positive, energised → active or social trips")
        if v >= 0.2 and a < 0.2:
            scores["gentle_culture"] += 0.6
            scores["calm_nature"] += 0.4

        scores["calm_nature"] += _need_p(state, NeedType.REST)
        scores["social_city"] += _need_p(state, NeedType.CONNECTION)
        scores["celebration"] += _need_p(state, NeedType.CELEBRATION) * 1.5
        scores["adventure"] += _need_p(state, NeedType.CHANGE_OF_SCENE) * 0.5
        scores["calm_nature"] += _need_p(state, NeedType.CHANGE_OF_SCENE) * 0.5

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        items = [{"archetype": k, "score": round(s, 3), **ARCHETYPES[k]} for k, s in ranked if s > 0][:3]
        if state.uncertain or state.mixed_or_ambiguous:
            why.append("emotional signal uncertain → suggestions are generic")
        return DomainSuggestion(domain=self.domain, items=items, rationale=why, confidence=state.overall_confidence)
