"""Performance, stability and calibration metrics shared by modelling and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-6


def gini(y, p) -> float:
    return float(2 * roc_auc_score(y, p) - 1)


def ks(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi_from_counts(expected: pd.Series, actual: pd.Series) -> float:
    """Population stability index between two distributions over the same bins."""
    idx = expected.index.union(actual.index)
    e = expected.reindex(idx, fill_value=0).astype(float)
    a = actual.reindex(idx, fill_value=0).astype(float)
    e = (e / e.sum()).clip(lower=EPS)
    a = (a / a.sum()).clip(lower=EPS)
    return float(((a - e) * np.log(a / e)).sum())


def psi(expected_bins: pd.Series, actual_bins: pd.Series) -> float:
    return psi_from_counts(expected_bins.value_counts(dropna=False),
                           actual_bins.value_counts(dropna=False))  # fmt: skip


def decile_table(y, p, q: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bucket"] = pd.qcut(df["p"].rank(method="first"), q, labels=False)
    return df.groupby("bucket").agg(n=("y", "size"), defaults=("y", "sum"), pd_mean=("p", "mean"),
                                    dr=("y", "mean")).reset_index()  # fmt: skip


def ece(y, p, q: int = 10) -> float:
    """Expected calibration error over PD deciles (count-weighted |mean PD - default rate|)."""
    t = decile_table(y, p, q)
    return float((t["n"] * (t["pd_mean"] - t["dr"]).abs()).sum() / t["n"].sum())


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration: logit(P(y=1)) = a + b * logit(p). Perfect calibration: a=0, b=1."""
    import statsmodels.api as sm

    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    lp = np.log(p / (1 - p))
    if np.std(lp) < 1e-9:  # constant predictions: slope undefined
        return float("nan"), float("nan")
    res = sm.Logit(np.asarray(y), sm.add_constant(lp, has_constant="add")).fit(disp=0)
    return float(res.params[1]), float(res.params[0])


def hosmer_lemeshow(y, p, q: int = 10) -> tuple[float, float]:
    t = decile_table(y, p, q)
    exp = t["n"] * t["pd_mean"]
    obs = t["defaults"]
    stat = float((((obs - exp) ** 2) / (exp * (1 - t["pd_mean"])).clip(lower=EPS)).sum())
    return stat, float(1 - stats.chi2.cdf(stat, q - 2))
