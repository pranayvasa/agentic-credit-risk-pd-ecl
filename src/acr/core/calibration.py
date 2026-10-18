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
