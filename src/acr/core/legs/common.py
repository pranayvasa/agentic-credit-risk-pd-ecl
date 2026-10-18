"""Shared leg utilities: feature dictionary, safe ratios and leg-level quality checks.

Leg transforms are per-customer aggregations of that customer's own records. They learn
nothing from other customers or from TARGET (leg builders are never given TARGET), so they
can run on all partitions without leakage. Anything that learns from data (binning, WoE,
selection, calibration) happens later, on train only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from acr.core.schema import DAYS_SENTINEL, ID

DAYS_PER_MONTH = 30.4375
DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class FeatureDef:
    name: str
    definition: str
    source_table: str
    leg: str
    kind: str = "numeric"  # numeric | categorical
    role: str = "candidate"  # candidate | segment_only


class FeatureDict:
    def __init__(self, leg: str) -> None:
        self.leg = leg
        self._defs: dict[str, FeatureDef] = {}

    def add(self, name: str, definition: str, source: str, kind: str = "numeric",
            role: str = "candidate") -> str:  # fmt: skip
        if name in self._defs:
            raise ValueError(f"duplicate feature {name}")
        self._defs[name] = FeatureDef(name, definition, source, self.leg, kind, role)
        return name

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(d) for d in self._defs.values()])

    def names(self) -> list[str]:
        return list(self._defs)


@dataclass
class LegResult:
    features: pd.DataFrame  # indexed by SK_ID_CURR, one row per requested ID
    dictionary: pd.DataFrame
    has_record: pd.Series  # bool per ID: customer has any source record


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """num / den with zero or missing denominators giving NaN, never inf."""
    num = num.astype("float64")
    den = den.astype("float64")
    return (num / den.where(den != 0)).replace([np.inf, -np.inf], np.nan)


def clean_days(s: pd.Series) -> pd.Series:
    """Replace the 365243 sentinel with NaN."""
    s = s.astype("float64")
    return s.where(s != DAYS_SENTINEL)


def finalise(features: pd.DataFrame, fd: FeatureDict, ids: np.ndarray,
             count_cols: list[str] = ()) -> pd.DataFrame:  # fmt: skip
    """Reindex to all requested IDs, zero-fill count features, check against the dictionary."""
    out = features.reindex(pd.Index(ids, name=ID))
    for c in count_cols:
        out[c] = out[c].fillna(0)
    expected, got = fd.names(), list(out.columns)
    if sorted(expected) != sorted(got):
        raise ValueError(f"{fd.leg}: dictionary/feature mismatch "
                         f"{sorted(set(expected) ^ set(got))}")  # fmt: skip
    num = out.select_dtypes("number").columns
    out[num] = out[num].astype("float32")
    return out[expected]


def leg_quality(res: LegResult) -> dict:
    f = res.features
    num = f.select_dtypes("number")
    all_null = [c for c in f.columns if f[c].isna().all()]
    constant = [c for c in f.columns if f[c].nunique(dropna=True) <= 1 and c not in all_null]
    issues = []
    if not f.index.is_unique:
        issues.append({"severity": "critical", "check": "id_unique", "detail": "duplicate IDs"})
    if np.isinf(num.to_numpy(dtype="float64", na_value=np.nan)).any():
        issues.append({"severity": "high", "check": "no_inf", "detail": "infinite values"})
    if all_null:
        issues.append({"severity": "medium", "check": "all_null", "detail": all_null})
    if constant:
        issues.append({"severity": "low", "check": "constant", "detail": constant})
    return {
        "rows": len(f),
        "n_features": f.shape[1],
        "coverage": round(float(res.has_record.mean()), 6),
        "missing_share_mean": round(float(f.isna().mean().mean()), 6),
        "issues": issues,
        "status": "fail" if any(i["severity"] == "critical" for i in issues) else "pass",
    }
