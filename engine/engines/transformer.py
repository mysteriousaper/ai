"""Engine B. Fine-tuned encoder: multi-label GoEmotions head + VAD regression head.

Interface stub (Phase 2). Implemented in Phase 3.
"""

from __future__ import annotations

from engine.engines.base import Engine
from engine.schemas import EngineInput, EngineOutput


class TransformerEngine(Engine):
    name = "transformer"
    version = "0.0.0"
    requires = frozenset({"text"})
    languages = frozenset({"en"})

    def analyze(self, inp: EngineInput) -> EngineOutput:
        raise NotImplementedError("Phase 3")
