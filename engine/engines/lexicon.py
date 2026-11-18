"""Engine A. NRC VAD v2.1 + EmoLex lookups with negation/intensifier/contrast rules.

Interface stub (Phase 2). Implemented in Phase 3.
"""

from __future__ import annotations

from engine.engines.base import Engine
from engine.schemas import EngineInput, EngineOutput


class LexiconEngine(Engine):
    name = "lexicon"
    version = "0.0.0"
    requires = frozenset({"text"})
    languages = frozenset({"en"})

    def analyze(self, inp: EngineInput) -> EngineOutput:
        raise NotImplementedError("Phase 3")
