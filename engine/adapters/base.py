"""Domain adapter interface. The core never imports a domain; domains import the core.

`run_adapter` is the only sanctioned way to call an adapter: it enforces the safety rule, so a
product cannot accidentally show "solutions" to someone in crisis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from engine.schemas import EmotionalStateObject, SafetyLevel

HELP_RESOURCES = [
    "If you are in immediate danger, call your local emergency number (112 in Armenia/EU, 911 in the US).",
    "Find a free, confidential helpline in your country: https://findahelpline.com",
]


class DomainSuggestion(BaseModel):
    domain: str
    items: list[dict[str, Any]] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1, default=0.0)
    safety_override: bool = False


class Adapter(ABC):
    domain: ClassVar[str]

    @abstractmethod
    def adapt(self, state: EmotionalStateObject, user_profile: dict[str, Any] | None = None) -> DomainSuggestion: ...


def run_adapter(adapter: Adapter, state: EmotionalStateObject, user_profile: dict[str, Any] | None = None) -> DomainSuggestion:
    if state.safety_flag.level == SafetyLevel.URGENT:
        return DomainSuggestion(
            domain=adapter.domain,
            items=[{"type": "help_resource", "text": t} for t in HELP_RESOURCES],
            rationale=["safety_flag=urgent: domain suggestions suppressed, human help shown instead"],
            confidence=1.0,
            safety_override=True,
        )
    return adapter.adapt(state, user_profile)
