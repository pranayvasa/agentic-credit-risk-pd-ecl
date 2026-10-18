"""Leakage guards. Fitting functions call these so non-train rows can never reach a fitter."""

from __future__ import annotations

import pandas as pd


class LeakageError(RuntimeError):
    """Rows from a partition other than the one allowed were passed to a fitting step."""


def require_partition(frame: pd.DataFrame, assignment: pd.DataFrame, partition: str = "train"):
    """Raise unless every row of ``frame`` (indexed by SK_ID_CURR) belongs to ``partition``."""
    part = assignment.set_index("SK_ID_CURR")["partition"]
    found = part.reindex(frame.index)
    if found.isna().any():
        raise LeakageError(f"{int(found.isna().sum())} rows have no split assignment")
    bad = found[found != partition]
    if len(bad):
        raise LeakageError(
            f"{len(bad)} rows from {sorted(bad.astype(str).unique())} passed to a "
            f"{partition}-only fitting step"
        )
    return frame
