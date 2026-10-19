"""Node 5: modelling. WoE logistic scorecard and monotone LightGBM, calibration, master scale.

Both models are always built. ``model.champion`` in config assigns the roles: the champion
drives gates, ECL, documents and reason codes; the other model is the benchmark challenger and
documented fallback (decision D33). Scored columns ``pd_champion`` / ``pd_challenger`` follow
the roles; ``pd_scorecard`` / ``pd_lightgbm`` always name the models.

Also fits the application-leg-only "origination proxy" in the champion's model family, used by
the ECL layer's relative SICR test (decision D6). All fitting uses train only.
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
    role="Fit the WoE logistic scorecard and monotone LightGBM on train; assign champion and "
    "challenger roles from config; decide calibration from out-of-fold train predictions; map PD "
    "to the master scale; fit an application-only origination proxy in the champion's family.",
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

    # --- WoE logistic scorecard ---
    sel = meta["selected"]
    sc = fit_scorecard(Wtr, ytr, sel, m.scorecard)

    def lr_fp(Xa, ya, Xb):
        return fit_scorecard(Xa, ya, sel, m.scorecard).predict_pd(Xb)

    oof_ch = _oof(lr_fp, Wtr, ytr, m.calibration.cv_folds, seed)
    cal_ch = decide(ytr, oof_ch, m.calibration)
    pd_ch_raw = sc.predict_pd(W)
    pd_ch = cal_ch.apply(pd_ch_raw)

    # --- monotone LightGBM ---
    excluded = set(c.base.features.exclude_features)
    cands = [f for f in meta["candidates"] if f in live and f not in excluded]
    Xtr = require_partition(features.loc[train_ids, cands], a.reset_index(), "train")
    # Monotonic constraints: directions from the train-only WoE binning trends.
    signs = C.monotone_signs(bp, cands, cat_live) if m.lightgbm_fit.monotone else None
    lgbm = C.fit_challenger(Xtr, ytr, cat_live, dict(m.lightgbm), m.lightgbm_fit, seed, signs)
    n_trees = int(lgbm.n_estimators)
    fixed = C.constrained({**dict(m.lightgbm), "n_estimators": n_trees}, cands, signs,
                          m.lightgbm_fit.monotone_method)  # fmt: skip

    def gb_fp(Xa, ya, Xb):
        import lightgbm as lgb

        mdl = lgb.LGBMClassifier(**C._params(fixed, seed))
        mdl.fit(C.prepare(Xa, cat_live), ya)
        return C.predict_pd(mdl, Xb, cat_live)

    oof_cl = _oof(gb_fp, Xtr, ytr, m.calibration.cv_folds, seed)
    cal_cl = decide(ytr, oof_cl, m.calibration)
    pd_cl = cal_cl.apply(C.predict_pd(lgbm, features[cands], cat_live))

    # --- origination proxy: application leg only, in the champion's model family ---
    champ = m.champion
    if champ == "lightgbm":
        app_cands = [f for f in cands if dictionary.loc[f, "leg"] == "application"]
        app_signs = {f: signs[f] for f in app_cands} if signs else None
        proxy = C.fit_challenger(Xtr[app_cands], ytr, [f for f in cat_live if f in app_cands],
                                 dict(m.lightgbm), m.lightgbm_fit, seed, app_signs)  # fmt: skip
        pd_proxy = C.predict_pd(proxy, features[app_cands], [f for f in cat_live if f in app_cands])
        proxy_feats, proxy_type = app_cands, "monotone LightGBM, application leg only"
    else:
        app_feats = [f for f in meta["screened"] if dictionary.loc[f, "leg"] == "application"]
        iv = B.iv_table(bp, live).set_index("feature")["iv"]
        proxy_feats = prune(Wtr, ytr, app_feats, iv, c.base.features, [])
        proxy = fit_scorecard(Wtr, ytr, proxy_feats, m.scorecard)
        pd_proxy = proxy.predict_pd(W)
        proxy_type = "WoE logistic regression, application leg only"

    bounds = m.master_scale.pd_upper_bounds
    pds = {"scorecard": pd_ch, "lightgbm": pd_cl}
    chal = "scorecard" if champ == "lightgbm" else "lightgbm"
    scored = pd.DataFrame({
        "partition": a["partition"].astype(str).reindex(features.index),
        "TARGET": y.reindex(features.index),
        "score": sc.score(W), "pd_scorecard_raw": pd_ch_raw,
        "pd_scorecard": pd_ch, "pd_lightgbm": pd_cl,
        "pd_champion": pds[champ], "grade_champion": assign_grade(pds[champ], bounds),
        "pd_challenger": pds[chal], "grade_challenger": assign_grade(pds[chal], bounds),
        "pd_origination_proxy": pd_proxy, "grade_origination_proxy": assign_grade(pd_proxy, bounds),
    }, index=features.index)  # fmt: skip

    out = artefact_dir(view, "model")
    bin_tables = B.binning_tables(bp, sel)
    files = {
        "scored": out / "scored.parquet", "scorecard": out / "scorecard.pkl",
        "coefficients": out / "scorecard_coefficients.csv", "points": out / "scorecard_points.csv",
        "proxy": out / "origination_proxy.pkl", "lightgbm": out / "lightgbm.txt",
        "importance": out / "lightgbm_gain_importance.csv", "master_scale": out / "master_scale.csv",
        "calibration": out / "calibration.json",
        "monotone": out / "lightgbm_monotone_constraints.json",
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
    lgbm.booster_.save_model(str(files["lightgbm"]))
    C.importance(lgbm).to_csv(files["importance"], index=False)
    scale_table(bounds).to_csv(files["master_scale"], index=False)
    calib = {"scorecard": cal_ch.__dict__, "lightgbm": cal_cl.__dict__}
    calib = {**calib, "champion": calib[champ], "challenger": calib[chal]}
    files["calibration"].write_text(json.dumps(calib, indent=2))
    files["monotone"].write_text(json.dumps({"enabled": bool(signs), "method": m.lightgbm_fit.monotone_method,
                                             "signs": signs or {}}, indent=2))  # fmt: skip

    tr = scored.loc[train_ids]

    def _g(col):
        return round(gini(tr["TARGET"], tr[col]), 4)

    summaries = {
        "scorecard": {
            "model": "scorecard",
            "type": "WoE logistic regression scorecard",
            "n_features": len(sel),
            "intercept": round(sc.intercept, 6),
            "coefficients": {k: round(v, 6) for k, v in sc.coefs.items()},
            "calibration": cal_ch.reason, "calibration_applied": cal_ch.applied,
            "train_gini": _g("pd_scorecard"), "train_auc": round((_g("pd_scorecard") + 1) / 2, 4),
            "train_ks": round(ks(tr["TARGET"], tr["pd_scorecard"]), 4),
            "oof_gini": round(gini(ytr, oof_ch), 4), "oof_auc": round((gini(ytr, oof_ch) + 1) / 2, 4),
            "scaling": {"pdo": sc.pdo, "base_score": sc.base_score, "base_odds": sc.base_odds},
        },
        "lightgbm": {
            "model": "lightgbm",
            "type": "monotone LightGBM gradient boosting" if signs else "LightGBM gradient boosting",
            "n_features": len(cands),
            "n_trees": n_trees,
            "monotone": {
                "enabled": bool(signs),
                "method": m.lightgbm_fit.monotone_method,
                "increasing": sum(v == 1 for v in (signs or {}).values()),
                "decreasing": sum(v == -1 for v in (signs or {}).values()),
                "free": sum(v == 0 for v in (signs or {}).values()),
            },
            "calibration": cal_cl.reason, "calibration_applied": cal_cl.applied,
            "train_gini": _g("pd_lightgbm"), "train_auc": round((_g("pd_lightgbm") + 1) / 2, 4),
            "oof_gini": round(gini(ytr, oof_cl), 4), "oof_auc": round((gini(ytr, oof_cl) + 1) / 2, 4),
        },
    }  # fmt: skip
    return {
        "model": {
            "roles": {"champion": champ, "challenger": chal},
            **summaries,
            "champion": summaries[champ],
            "challenger": summaries[chal],
            "origination_proxy": {
                "type": proxy_type + " (ECL SICR proxy, D6)",
                "features": proxy_feats,
                "train_gini": _g("pd_origination_proxy"),
                "train_auc": round((_g("pd_origination_proxy") + 1) / 2, 4),
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
