"""Confidence-weighted late fusion (baseline). Implemented in Phase 4."""

from __future__ import annotations

from collections.abc import Sequence

from engine.schemas import EmotionalStateObject, EngineOutput


def fuse(outputs: Sequence[EngineOutput], weights: dict[str, float] | None = None) -> EmotionalStateObject:
    """Weight each engine by prior weight x confidence x coverage; skip empty outputs.

    Disagreement between engines lowers overall_confidence and may set mixed_or_ambiguous.
    """
    raise NotImplementedError("Phase 4")
