"""Probability calibration (temperature scaling / isotonic) and ECE. Phase 4."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    raise NotImplementedError("Phase 4")
