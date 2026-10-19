"""Explainability for the LightGBM challenger: exact TreeSHAP and checks on it.

SHAP values come from LightGBM itself (``pred_contrib=True``): an exact, deterministic
TreeSHAP decomposition in log-odds. For every applicant, bias + sum of feature
contributions = the model's raw score, so each prediction is fully attributed.

Caveats stated in the outputs: SHAP describes how the model attributes a prediction, not
cause and effect; correlated features (e.g. EXT_SOURCE_MEAN vs EXT_SOURCE_1/2/3) can share
credit arbitrarily, which is why importance is also reported by leg.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def shap_frame(booster: lgb.Booster, X: pd.DataFrame) -> pd.DataFrame:
    contrib = booster.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib, index=X.index, columns=[*booster.feature_name(), "bias"])


def additivity_error(booster: lgb.Booster, X: pd.DataFrame, sh: pd.DataFrame) -> float:
    return float(np.abs(sh.sum(axis=1).to_numpy() - booster.predict(X, raw_score=True)).max())


def global_importance(sh: pd.DataFrame) -> pd.DataFrame:
    f = sh.drop(columns="bias")
    imp = pd.DataFrame({"mean_abs_shap": f.abs().mean(), "mean_shap": f.mean()})
    total = imp["mean_abs_shap"].sum()
    imp["share"] = (
        imp["mean_abs_shap"] / total if total > 0 else 0.0
    )  # all-zero: model made no splits
    return imp.sort_values("mean_abs_shap", ascending=False)


def leg_importance(imp: pd.DataFrame, leg_of: pd.Series) -> pd.Series:
    return imp.groupby(leg_of.reindex(imp.index))["share"].sum().sort_values(ascending=False)


def reason_codes(sh: pd.DataFrame, X: pd.DataFrame, k: int) -> pd.DataFrame:
    """Top-k features that pushed each applicant's PD *up* (largest positive SHAP), with the
    applicant's value for that feature. Long format: one row per applicant and rank."""
    f = sh.drop(columns="bias")
    vals = f.to_numpy()
    order = np.argsort(-vals, axis=1)[:, :k]
    cols = np.asarray(f.columns)
    rows = []
    for i, idx in enumerate(f.index):
        for rank, j in enumerate(order[i], 1):
            if vals[i, j] <= 0:
                break
            v = X.iat[i, X.columns.get_loc(cols[j])]
            rows.append((idx, rank, cols[j], v if pd.notna(v) else None, float(vals[i, j])))
    out = pd.DataFrame(rows, columns=["SK_ID_CURR", "rank", "feature", "value", "shap_logodds"])
    out["value"] = out["value"].astype("string")
    return out


def stability(imp_a: pd.DataFrame, imp_b: pd.DataFrame, top_n: int) -> dict:
    """Do two samples rank the same drivers? Top-N overlap and Spearman over all features."""
    a, b = imp_a["mean_abs_shap"], imp_b["mean_abs_shap"].reindex(imp_a.index)
    top_a, top_b = set(a.nlargest(top_n).index), set(b.nlargest(top_n).index)
    if a.sum() == 0 or b.sum() == 0:  # a model with no splits explains nothing
        return {"top_n": top_n, "top_overlap": 0.0, "spearman": float("nan")}
    return {"top_n": top_n, "top_overlap": len(top_a & top_b) / top_n,
            "spearman": float(spearmanr(a, b).statistic)}  # fmt: skip


def monotonicity_violations(booster: lgb.Booster, X: pd.DataFrame, signs: dict[str, int],
                            grid: int = 11) -> pd.DataFrame:  # fmt: skip
    """Sweep each constrained feature across its quantiles for every row of X and count rows
    whose prediction moves against the allowed direction (tolerance 1e-9 in log-odds)."""
    rows = []
    for f, s in signs.items():
        if s == 0 or f not in X.columns:
            continue
        qs = np.unique(np.nanquantile(X[f].astype(float), np.linspace(0, 1, grid)))
        if len(qs) < 2:
            continue
        preds = []
        for q in qs:
            Xq = X.copy()
            Xq[f] = np.float32(q)
            preds.append(booster.predict(Xq, raw_score=True))
        steps = np.diff(np.vstack(preds), axis=0) * s
        rows.append({"feature": f, "sign": s, "rows": len(X), "grid_points": len(qs),
                     "violating_rows": int((steps < -1e-9).any(axis=0).sum())})  # fmt: skip
    return pd.DataFrame(rows)
