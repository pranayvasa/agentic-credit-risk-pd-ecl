"""Challenger: LightGBM on all candidate features (raw values, not WoE).

Hyperparameters are fixed in config; nothing is tuned automatically, by an LLM or otherwise.
Early stopping uses a stratified slice carved out of train, never test or OOT.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from acr.config import LGBMFitCfg


def prepare(X: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
    X = X.copy()
    for c in X.columns:
        X[c] = X[c].astype("category") if c in categorical else X[c].astype("float32")
    return X


def _params(params: dict, seed: int) -> dict:
    return {**params, "random_state": seed, "deterministic": True, "force_row_wise": True,
            "n_jobs": 4, "verbose": -1}  # fmt: skip


def fit_challenger(X: pd.DataFrame, y: pd.Series, categorical: list[str], params: dict,
                   fit_cfg: LGBMFitCfg, seed: int) -> lgb.LGBMClassifier:  # fmt: skip
    X = prepare(X, categorical)
    Xa, Xv, ya, yv = train_test_split(X, y, test_size=fit_cfg.validation_frac, stratify=y,
                                      random_state=seed)  # fmt: skip
    probe = lgb.LGBMClassifier(**_params(params, seed))
    probe.fit(Xa, ya, eval_X=Xv, eval_y=yv, eval_metric="auc",
              callbacks=[lgb.early_stopping(fit_cfg.early_stopping_rounds, verbose=False)])  # fmt: skip
    best = int(probe.best_iteration_ or params["n_estimators"])
    # Refit on all of train with the early-stopped number of trees.
    final = lgb.LGBMClassifier(**_params({**params, "n_estimators": best}, seed))
    final.fit(X, y)
    return final


def predict_pd(model: lgb.LGBMClassifier, X: pd.DataFrame, categorical: list[str]) -> np.ndarray:
    return model.predict_proba(prepare(X, categorical))[:, 1]


def importance(model: lgb.LGBMClassifier) -> pd.DataFrame:
    gain = model.booster_.feature_importance(importance_type="gain")
    return (pd.DataFrame({"feature": model.booster_.feature_name(), "gain": gain})
            .assign(share=lambda d: d["gain"] / d["gain"].sum())
            .sort_values("gain", ascending=False).reset_index(drop=True))  # fmt: skip
