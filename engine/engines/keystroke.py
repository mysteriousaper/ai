"""Engine D. Typing dynamics as z-scores against the user's personal baseline.

Interface stub (Phase 2). Implemented in Phase 3.
"""

from __future__ import annotations

from engine.engines.base import Engine
from engine.schemas import EngineInput, EngineOutput


class KeystrokeEngine(Engine):
    name = "keystroke"
    version = "0.0.0"
    requires = frozenset({"keystrokes"})
    languages = None

    def analyze(self, inp: EngineInput) -> EngineOutput:
        raise NotImplementedError("Phase 3")
