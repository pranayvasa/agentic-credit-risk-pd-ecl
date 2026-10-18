"""Master rating scale: maps PD to grades 1 (best) .. N (worst) using upper PD bounds."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_grade(pd_: np.ndarray, upper_bounds: list[float]) -> np.ndarray:
    """Grade k holds PDs in (bound_{k-1}, bound_k]."""
    return np.searchsorted(np.asarray(upper_bounds), np.asarray(pd_), side="left") + 1


def scale_table(upper_bounds: list[float]) -> pd.DataFrame:
    lower = [0.0, *upper_bounds[:-1]]
    return pd.DataFrame({
        "grade": np.arange(1, len(upper_bounds) + 1),
        "pd_lower": lower,
        "pd_upper": upper_bounds,
        "pd_mid": [np.sqrt(max(lo, 1e-4) * hi) if hi < 1 else lo * 1.5
                   for lo, hi in zip(lower, upper_bounds, strict=True)],
    })  # fmt: skip
