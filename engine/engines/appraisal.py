"""Engine E. LLM (Claude) structured appraisal: event type, agent, appraisals, need, evidence spans; local fallback classifier.

Interface stub (Phase 2). Implemented in Phase 3.
"""

from __future__ import annotations

from engine.engines.base import Engine
from engine.schemas import EngineInput, EngineOutput


class AppraisalEngine(Engine):
    name = "appraisal"
    version = "0.0.0"
    requires = frozenset({"text"})
    languages = None

    def analyze(self, inp: EngineInput) -> EngineOutput:
        raise NotImplementedError("Phase 3")
