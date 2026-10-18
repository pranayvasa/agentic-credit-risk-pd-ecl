"""Optimal WoE binning (optbinning), fitted on train only.

WoE convention (optbinning): WoE = ln(%non-events / %events), so a higher WoE means lower
risk. In a logistic regression of default on WoE, every coefficient should be negative.
Missing values get their own bin with its empirical WoE (optbinning's default of 0 is
overridden).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from optbinning import BinningProcess

from acr.core.metrics import psi

TRANSFORM_KW = {"metric_missing": "empirical", "metric_special": "empirical"}


def prepare(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    """optbinning needs plain object strings for categoricals and float numerics."""
    X = X.copy()
    for c in X.columns:
        if c in categorical:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        else:
            X[c] = X[c].astype("float64")
    return X


def fit_binning(X: pd.DataFrame, y: pd.Series, categorical: list[str], min_bin_size: float,
                max_n_bins: int) -> BinningProcess:  # fmt: skip
    numeric = [c for c in X.columns if c not in categorical]
    bp = BinningProcess(
        list(X.columns),
        categorical_variables=categorical,
        min_bin_size=min_bin_size,
        max_n_bins=max_n_bins,
        binning_fit_params={c: {"monotonic_trend": "auto_asc_desc"} for c in numeric},
    )
    bp.fit(prepare(X, categorical), y.to_numpy())
    return bp


def woe(bp: BinningProcess, X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = bp.transform(prepare(X, categorical), metric="woe", **TRANSFORM_KW)
    out.index = X.index
    return out


def bins(bp: BinningProcess, X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    out = bp.transform(prepare(X, categorical), metric="indices", **TRANSFORM_KW)
    out.index = X.index
    return out


def iv_table(bp: BinningProcess, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        ob = bp.get_binned_variable(f)
        bt = ob.binning_table.build()
        rows.append({"feature": f, "iv": float(bt.loc["Totals", "IV"]),
                     "n_bins": int(len(bt) - 3), "status": ob.status})  # fmt: skip
    return pd.DataFrame(rows)


def binning_tables(bp: BinningProcess, features: list[str]) -> pd.DataFrame:
    """Long table of every bin for every feature: for documentation and the points table."""
    frames = []
    for f in features:
        bt = bp.get_binned_variable(f).binning_table.build().drop(index="Totals")
        bt = bt.reset_index(names="bin_index")
        bt["Bin"] = bt["Bin"].map(
            lambda b: ", ".join(map(str, b)) if isinstance(b, (list, np.ndarray)) else str(b)
        )
        frames.append(
            bt.assign(feature=f)[
                ["feature", "bin_index", "Bin", "Count", "Event rate", "WoE", "IV"]
            ]
        )
    return pd.concat(frames, ignore_index=True).rename(
        columns={
            "Bin": "bin",
            "Count": "count",
            "Event rate": "event_rate",
            "WoE": "woe",
            "IV": "iv",
        }
    )


def csi(bins_a: pd.DataFrame, bins_b: pd.DataFrame) -> dict[str, float]:
    """Characteristic stability index per feature over the fitted bins."""
    return {c: psi(bins_a[c], bins_b[c]) for c in bins_a.columns}
