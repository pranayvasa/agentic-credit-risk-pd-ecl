"""Champion: WoE logistic regression scorecard.

Points scaling (standard PDO form):
    factor = PDO / ln(2),   offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(good:bad odds) = offset - factor * (b0 + sum_i b_i * WoE_i)
Points for bin j of feature i, with the intercept and offset spread evenly over n features:
    points_ij = -factor * b_i * WoE_ij + (offset - factor * b0) / n
A higher score means lower risk. PD comes from the logistic model, not from rounded points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from acr.config import ScorecardCfg
from acr.core.selection import logit_fit


@dataclass
class Scorecard:
    features: list[str]
    intercept: float
    coefs: dict[str, float]
    coef_table: pd.DataFrame
    pdo: float
    base_score: float
    base_odds: float

    @property
    def factor(self) -> float:
        return self.pdo / np.log(2)

    @property
    def offset(self) -> float:
        return self.base_score - self.factor * np.log(self.base_odds)

    def linear(self, W: pd.DataFrame) -> np.ndarray:
        b = np.array([self.coefs[f] for f in self.features])
        return self.intercept + W[self.features].to_numpy(dtype="float64") @ b

    def predict_pd(self, W: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.linear(W)))

    def score(self, W: pd.DataFrame) -> np.ndarray:
        return self.offset - self.factor * self.linear(W)

    def points_table(self, bin_tables: pd.DataFrame) -> pd.DataFrame:
        n = len(self.features)
        t = bin_tables[bin_tables["feature"].isin(self.features)].copy()
        t = t[t["count"] > 0]
        t["coef"] = t["feature"].map(self.coefs)
        t["points"] = (
            -self.factor * t["coef"] * t["woe"] + (self.offset - self.factor * self.intercept) / n
        )
        t["points_rounded"] = t["points"].round().astype(int)
        order = {f: i for i, f in enumerate(self.features)}
        return t.sort_values(
            ["feature", "bin_index"], key=lambda s: s.map(order) if s.name == "feature" else s
        )


def fit_scorecard(W_train: pd.DataFrame, y_train: pd.Series, features: list[str],
                  cfg: ScorecardCfg) -> Scorecard:  # fmt: skip
    res = logit_fit(W_train[features], y_train)
    names = ["intercept", *features]
    table = pd.DataFrame({"term": names, "coef": res.params, "std_err": res.bse, "z": res.tvalues,
                          "p_value": res.pvalues})  # fmt: skip
    return Scorecard(features, float(res.params[0]), dict(zip(features, map(float, res.params[1:]),
                     strict=True)), table, cfg.pdo, cfg.base_score, cfg.base_odds)  # fmt: skip
