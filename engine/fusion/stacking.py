"""Stacking meta-model (logistic regression / small MLP) over engine outputs. Phase 4."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from engine.schemas import EngineOutput


def featurize(outputs: Sequence[EngineOutput], engines: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Fixed-width vector: per engine [p(label)..., vad..., confidence, coverage, present_flag]."""
    raise NotImplementedError("Phase 4")
