"""Node 5: modelling. Champion WoE scorecard, LightGBM challenger, calibration, master scale.

Also fits the application-leg-only "origination proxy" scorecard used by the ECL layer's
relative SICR test (decision D6). All fitting uses train only; predictions are written for
every partition so the validator can evaluate them without touching the models.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from acr.audit import node
from acr.core import binning as B
from acr.core import challenger as C
from acr.core.calibration import decide
from acr.core.guards import require_partition
from acr.core.master_scale import assign_grade, scale_table
from acr.core.metrics import gini, ks
from acr.core.scorecard import fit_scorecard
from acr.core.selection import prune
from acr.nodes._util import artefact_dir, cfg, sha256


def _oof(fit_predict, X: pd.DataFrame, y: pd.Series, folds: int, seed: int) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        oof[va] = fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[va])
    return oof


@node(
    "model",
    role="Fit the champion WoE logistic scorecard and LightGBM challenger on train; decide "
    "calibration from out-of-fold train predictions; map PD to the master scale; fit the "
    "application-only origination-proxy scorecard for ECL staging.",
    reads=["config", "split", "selection"],
    writes=["model"],
)
def model(view):
    c = cfg(view)
    m, seed = c.base.model, c.base.seed
    sp = view["selection"]["paths"]
    meta = json.loads(Path(sp["meta"]).read_text())
    with open(sp["binning"], "rb") as f:
        bp = pickle.load(f)
    features = pd.read_parquet(sp["features_all"])
    dictionary = pd.read_csv(sp["dictionary"]).set_index("name")
    # Labelled rows come from the split assignment. Any other rows in `features` are the
    # unlabelled application_test applicants: scored, never fitted on.
    a = pd.read_parquet(view["split"]["assignment_path"]).set_index("SK_ID_CURR")
    train_ids = a.index[a["partition"] == "train"]
    y = a["TARGET"]
    ytr = y.loc[train_ids]

    live = list(bp.variable_names)
    cat_live = [f for f in live if f in meta["categorical"]]
    W = B.woe(bp, features[live], cat_live)
    Wtr = require_partition(W.loc[train_ids], a.reset_index(), "train")

    # --- champion scorecard ---
    sel = meta["selected"]
    sc = fit_scorecard(Wtr, ytr, sel, m.scorecard)

    def lr_fp(Xa, ya, Xb):
        return fit_scorecard(Xa, ya, sel, m.scorecard).predict_pd(Xb)

    oof_ch = _oof(lr_fp, Wtr, ytr, m.calibration.cv_folds, seed)
    cal_ch = decide(ytr, oof_ch, m.calibration)
    pd_ch_raw = sc.predict_pd(W)
    pd_ch = cal_ch.apply(pd_ch_raw)

    # --- origination proxy: application leg only ---
    app_feats = [f for f in meta["screened"] if dictionary.loc[f, "leg"] == "application"]
    iv = B.iv_table(bp, live).set_index("feature")["iv"]
    proxy_log: list[dict] = []
    proxy_feats = prune(Wtr, ytr, app_feats, iv, c.base.features, proxy_log)
    proxy = fit_scorecard(Wtr, ytr, proxy_feats, m.scorecard)
    pd_proxy = proxy.predict_pd(W)

    # --- challenger ---
    cands = [f for f in meta["candidates"] if f in live]
    Xtr = require_partition(features.loc[train_ids, cands], a.reset_index(), "train")
    lgbm = C.fit_challenger(Xtr, ytr, cat_live, dict(m.lightgbm), m.lightgbm_fit, seed)
    n_trees = int(lgbm.n_estimators)
    fixed = {**dict(m.lightgbm), "n_estimators": n_trees}

    def gb_fp(Xa, ya, Xb):
        import lightgbm as lgb

        mdl = lgb.LGBMClassifier(**C._params(fixed, seed))
        mdl.fit(C.prepare(Xa, cat_live), ya)
        return C.predict_pd(mdl, Xb, cat_live)

    oof_cl = _oof(gb_fp, Xtr, ytr, m.calibration.cv_folds, seed)
    cal_cl = decide(ytr, oof_cl, m.calibration)
    pd_cl = cal_cl.apply(C.predict_pd(lgbm, features[cands], cat_live))

    bounds = m.master_scale.pd_upper_bounds
    scored = pd.DataFrame({
        "partition": a["partition"].astype(str).reindex(features.index),
        "TARGET": y.reindex(features.index),
        "score": sc.score(W), "pd_champion_raw": pd_ch_raw, "pd_champion": pd_ch,
        "grade_champion": assign_grade(pd_ch, bounds),
        "pd_challenger": pd_cl, "grade_challenger": assign_grade(pd_cl, bounds),
        "pd_origination_proxy": pd_proxy, "grade_origination_proxy": assign_grade(pd_proxy, bounds),
    }, index=features.index)  # fmt: skip

    out = artefact_dir(view, "model")
    bin_tables = B.binning_tables(bp, sel)
    files = {
        "scored": out / "scored.parquet", "scorecard": out / "champion_scorecard.pkl",
        "coefficients": out / "champion_coefficients.csv", "points": out / "scorecard_points.csv",
        "proxy": out / "origination_proxy_scorecard.pkl", "challenger": out / "challenger_lgbm.txt",
        "importance": out / "challenger_importance.csv", "master_scale": out / "master_scale.csv",
        "calibration": out / "calibration.json",
    }  # fmt: skip
    external = scored.drop(index=a.index).drop(columns=["partition", "TARGET"])
    scored = scored.loc[a.index]
    scored.to_parquet(files["scored"])
    ext_info = view["split"].get("external")
    if len(external):
        files["scored_external"] = out / "scored_external.parquet"
        external.to_parquet(files["scored_external"])
    kaggle = {}
    if ext_info and ext_info["complete"]:
        # Kaggle needs every application_test applicant, so only complete (non-sample) runs export.
        kdir = Path(view["run_dir"]) / "kaggle"
        kdir.mkdir(exist_ok=True)
        for name, col in (("champion", "pd_champion"), ("challenger", "pd_challenger")):
            path = kdir / f"submission_{name}.csv"
            external[[col]].rename(columns={col: "TARGET"}).reset_index().to_csv(path, index=False)
            kaggle[name] = str(path)
    with open(files["scorecard"], "wb") as f:
        pickle.dump(sc, f)
    with open(files["proxy"], "wb") as f:
        pickle.dump(proxy, f)
    sc.coef_table.to_csv(files["coefficients"], index=False)
    sc.points_table(bin_tables).to_csv(files["points"], index=False)
    lgbm.booster_.save_model(str(files["challenger"]))
    C.importance(lgbm).to_csv(files["importance"], index=False)
    scale_table(bounds).to_csv(files["master_scale"], index=False)
    calib = {"champion": cal_ch.__dict__, "challenger": cal_cl.__dict__}
    files["calibration"].write_text(json.dumps(calib, indent=2))

    tr = scored.loc[train_ids]
    return {
        "model": {
            "champion": {
                "type": "WoE logistic regression scorecard",
                "n_features": len(sel),
                "intercept": round(sc.intercept, 6),
                "coefficients": {k: round(v, 6) for k, v in sc.coefs.items()},
                "calibration": cal_ch.reason,
                "calibration_applied": cal_ch.applied,
                "train_gini": round(gini(tr["TARGET"], tr["pd_champion"]), 4),
                "train_auc": round((gini(tr["TARGET"], tr["pd_champion"]) + 1) / 2, 4),
                "train_ks": round(ks(tr["TARGET"], tr["pd_champion"]), 4),
                "oof_gini": round(gini(ytr, oof_ch), 4),
                "oof_auc": round((gini(ytr, oof_ch) + 1) / 2, 4),
                "scaling": {"pdo": sc.pdo, "base_score": sc.base_score, "base_odds": sc.base_odds},
            },
            "challenger": {
                "type": "LightGBM gradient boosting",
                "n_features": len(cands),
                "n_trees": n_trees,
                "calibration": cal_cl.reason,
                "calibration_applied": cal_cl.applied,
                "train_gini": round(gini(tr["TARGET"], tr["pd_challenger"]), 4),
                "train_auc": round((gini(tr["TARGET"], tr["pd_challenger"]) + 1) / 2, 4),
                "oof_gini": round(gini(ytr, oof_cl), 4),
                "oof_auc": round((gini(ytr, oof_cl) + 1) / 2, 4),
            },
            "origination_proxy": {
                "type": "WoE logistic regression, application leg only (ECL SICR proxy, D6)",
                "features": proxy_feats,
                "train_gini": round(gini(tr["TARGET"], tr["pd_origination_proxy"]), 4),
                "train_auc": round((gini(tr["TARGET"], tr["pd_origination_proxy"]) + 1) / 2, 4),
            },
            "external": {
                "n_scored": int(len(external)),
                "kaggle_submissions": kaggle,
                "note": "application_test has no TARGET; scored for a score-PSI check "
                "and Kaggle submission only",
            },  # fmt: skip
            "master_scale": {
                "pd_upper_bounds": bounds,
                "train_grade_counts": {
                    int(k): int(v)
                    for k, v in tr["grade_champion"].value_counts().sort_index().items()
                },
            },  # fmt: skip
            "paths": {k: str(v) for k, v in files.items()},
            "checksums": {k: sha256(v) for k, v in files.items()},
        }
    }
