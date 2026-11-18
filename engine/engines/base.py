"""The single interface every signal engine implements (brief §4.4).

Adding a modality later (voice, camera) = add a new Engine subclass declaring which inputs it
`requires`. Fusion only ever sees EngineOutput, so nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

from engine.schemas import EngineInput, EngineOutput

Modality = Literal["text", "keystrokes"]


class Engine(ABC):
    name: ClassVar[str]
    version: ClassVar[str] = "0.0.0"
    requires: ClassVar[frozenset[Modality]] = frozenset({"text"})
    languages: ClassVar[frozenset[str] | None] = frozenset({"en"})  # None = language-agnostic

    def can_run(self, inp: EngineInput) -> str | None:
        """Return a reason string if the engine cannot read this input, else None."""
        if "keystrokes" in self.requires and not inp.keystroke_events:
            return "no_keystrokes"
        if "text" in self.requires and not inp.text.strip():
            return "empty_text"
        lang = inp.context.language
        if self.languages is not None and lang is not None and lang not in self.languages:
            return f"unsupported_language:{lang}"
        return None

    def __call__(self, inp: EngineInput) -> EngineOutput:
        reason = self.can_run(inp)
        if reason:
            return EngineOutput.empty(self.name, self.version, reason)
        return self.analyze(inp)

    @abstractmethod
    def analyze(self, inp: EngineInput) -> EngineOutput:
        """Analyse one message. Must return a valid EngineOutput; never raise on odd text."""
