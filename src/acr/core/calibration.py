"""Calibration decision and Platt recalibration, driven by out-of-fold predictions on train.

Hosmer-Lemeshow is reported in validation, but with ~170k observations it rejects almost any
model, so it does not drive the decision here. The decision uses the decile ECE and the
logistic calibration slope of out-of-fold train predictions, with thresholds in config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acr.config import CalibrationCfg
from acr.core.metrics import EPS, calibration_slope_intercept, ece


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


@dataclass
class Calibrator:
    applied: bool
    slope: float = 1.0
    intercept: float = 0.0
    reason: str = ""
    oof_ece: float = 0.0
    oof_slope: float = 1.0
    oof_intercept: float = 0.0

    def apply(self, p: np.ndarray) -> np.ndarray:
        if not self.applied:
            return np.asarray(p, dtype="float64")
        return 1 / (1 + np.exp(-(self.intercept + self.slope * _logit(p))))


def decide(y_oof, p_oof, cfg: CalibrationCfg) -> Calibrator:
    e = ece(y_oof, p_oof)
    slope, intercept = calibration_slope_intercept(y_oof, p_oof)
    needs = e > cfg.ece_max or abs(slope - 1) > cfg.slope_tol
    reason = (f"OOF ECE {e:.4f} (max {cfg.ece_max}), slope {slope:.3f} (tolerance "
              f"{cfg.slope_tol})")  # fmt: skip
    if not needs:
        return Calibrator(False, reason="not needed: " + reason, oof_ece=e, oof_slope=slope,
                          oof_intercept=intercept)  # fmt: skip
    return Calibrator(True, slope, intercept, "Platt recalibration applied: " + reason, e, slope,
                      intercept)  # fmt: skip


@dataclass
class GradeRecalibrator:
    """Monotone grade-level recalibration in log-odds.

    Anchors: for each master-scale grade (assigned on out-of-fold train PDs), the mean predicted
    log-odds (x) and the observed default log-odds (y, with a small Jeffreys-style prior so
    sparse grades are stable). Isotonic regression makes y non-decreasing, a tiny step makes it
    strictly increasing, and PDs are mapped by linear interpolation between anchors (slope 1
    beyond the end anchors). The map is strictly increasing, so it never reorders applicants:
    Gini, KS and SHAP reason codes are unchanged; only PD levels move.
    """

    xs: list[float]
    ys: list[float]
    table: list[dict]

    @property
    def identity(self) -> bool:
        return not self.table

    def apply(self, p) -> np.ndarray:
        lp = _logit(np.asarray(p, dtype="float64"))
        xs, ys = np.asarray(self.xs), np.asarray(self.ys)
        out = np.interp(lp, xs, ys)
        out = np.where(lp < xs[0], ys[0] + (lp - xs[0]), out)
        out = np.where(lp > xs[-1], ys[-1] + (lp - xs[-1]), out)
        return 1 / (1 + np.exp(-out))


def fit_grade_recalibrator(y_oof, p_oof, bounds: list[float], min_n: int = 500,
                           prior: float = 0.5) -> GradeRecalibrator:  # fmt: skip
    import pandas as pd
    from sklearn.isotonic import IsotonicRegression

    from acr.core.master_scale import assign_grade

    df = pd.DataFrame({"y": np.asarray(y_oof), "p": np.asarray(p_oof, dtype="float64")})
    df["grade"] = assign_grade(df["p"].to_numpy(), bounds)
    g = df.groupby("grade").agg(n=("y", "size"), defaults=("y", "sum"), mean_pd=("p", "mean"),
                                mean_logit=("p", lambda s: float(np.mean(_logit(s.to_numpy())))))  # fmt: skip
    g = g[g["n"] >= min_n].copy()
    if len(g) < 2:  # too little data for a grade-level curve: leave PDs unchanged
        return GradeRecalibrator([-1.0, 1.0], [-1.0, 1.0], [])
    g["observed_dr"] = g["defaults"] / g["n"]
    g["target_dr"] = (g["defaults"] + prior) / (g["n"] + 2 * prior)
    target = _logit(g["target_dr"].to_numpy())
    iso = IsotonicRegression(increasing=True).fit(g["mean_logit"], target, sample_weight=g["n"])
    ys = iso.predict(g["mean_logit"])
    ys = ys + np.arange(len(ys)) * 1e-6  # strictly increasing: no ties, no reordering
    g["target_logit"] = ys
    g["recalibrated_dr"] = 1 / (1 + np.exp(-ys))
    table = g.reset_index().round(6).to_dict(orient="records")
    return GradeRecalibrator(list(g["mean_logit"]), list(ys), table)
