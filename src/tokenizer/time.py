from __future__ import annotations

import numpy as np


def soft_log_seconds(t: np.ndarray | float) -> np.ndarray:
    values = np.asarray(t, dtype=np.float64)
    return 8.0 * np.log1p(values / 8.0)


def periodic_encode(values: np.ndarray, period: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    radians = 2.0 * np.pi * values / float(period)
    return np.stack([np.sin(radians), np.cos(radians)], axis=-1)
