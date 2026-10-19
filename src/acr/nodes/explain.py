"""Explain node: SHAP explanations and reason codes for the LightGBM model (first-line explainability).

Reads the saved booster and features; never refits. Produces global and per-leg importance,
per-applicant reason codes for the whole OOT book, stability of the top drivers, a direct
test of the monotonic constraints, and the challenger's calibration by gender.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from acr.audit import node
from acr.core import challenger as C
from acr.core import explain as E
from acr.nodes._util import artefact_dir, cfg

EXTERNAL = {"APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3", "APP_EXT_SOURCE_MEAN",
            "APP_EXT_SOURCE_MIN", "APP_EXT_SOURCE_MISSING_COUNT"}  # fmt: skip


@node(
    "explain",
    role="Exact TreeSHAP explanations of the LightGBM model (champion under D33): global and per-leg "
    "importance, top-4 reason codes for every OOT applicant, stability of the top drivers, "
    "and a direct test that the monotonic constraints hold. Reads the saved model; never refits.",
    reads=["config", "split", "selection", "model"],
    writes=["explain"],
)
def explain(view):
    c = cfg(view)
    k, top_n = c.base.explain.reason_codes, c.base.explain.stability_top_n
    paths = view["model"]["paths"]
    booster = lgb.Booster(model_file=paths["lightgbm"])
    names = booster.feature_name()
    meta = json.loads(Path(view["selection"]["paths"]["meta"]).read_text())
    cat = [f for f in names if f in meta["categorical"]]
    feats = pd.read_parquet(view["selection"]["paths"]["features_all"])
    a = pd.read_parquet(view["split"]["assignment_path"])
    ids = {p: a.loc[a["partition"] == p, "SK_ID_CURR"] for p in ("train", "test", "oot")}
    leg_of = pd.read_csv(view["selection"]["paths"]["dictionary"]).set_index("name")["leg"]

    def shap_for(idx):
        X = C.prepare(feats.loc[idx, names], cat)
        return X, E.shap_frame(booster, X)

    X_oot, sh_oot = shap_for(ids["oot"])
    add_err = E.additivity_error(booster, X_oot, sh_oot)
    imp = E.global_importance(sh_oot)
    legs = E.leg_importance(imp, leg_of)
    _, sh_test = shap_for(ids["test"])
    train_sample = ids["train"].sample(n=min(50_000, len(ids["train"])), random_state=c.base.seed)
    _, sh_train = shap_for(train_sample)
    half = sh_oot.sample(frac=0.5, random_state=c.base.seed)
    stab = {
        "oot_half_vs_half": E.stability(E.global_importance(half), E.global_importance(sh_oot.drop(half.index)), top_n),
        "test_vs_oot": E.stability(E.global_importance(sh_test), imp, top_n),
        "train_vs_oot": E.stability(E.global_importance(sh_train), imp, top_n),
    }  # fmt: skip
    reasons = E.reason_codes(sh_oot, feats.loc[ids["oot"], names], k)
    fams = [f.model_dump() for f in c.base.explain.families]
    fmap = E.family_map(names, fams)
    labels = {f["family"]: f["label"] for f in fams}
    fam_reasons = E.family_reason_codes(sh_oot, fmap, labels, k)
    fam_imp = E.family_shap(sh_oot, fmap).abs().mean()
    fam_share = (fam_imp / fam_imp.sum()).sort_values(ascending=False) if fam_imp.sum() else fam_imp
    scored_oot = pd.read_parquet(paths["scored"]).loc[ids["oot"]]
    high = scored_oot.index[scored_oot["pd_lightgbm"] >= scored_oot["pd_lightgbm"].quantile(0.9)]
    primary = fam_reasons[(fam_reasons["rank"] == 1) & fam_reasons["SK_ID_CURR"].isin(high)]
    primary_share = primary["reason"].value_counts(normalize=True)

    mono = (
        json.loads(Path(paths["monotone"]).read_text())
        if "monotone" in paths
        else {"enabled": False}
    )
    sweep_rows = X_oot.sample(n=min(500, len(X_oot)), random_state=c.base.seed)
    viol = (E.monotonicity_violations(booster, sweep_rows, mono.get("signs", {}))
            if mono.get("enabled") else pd.DataFrame(columns=["feature", "violating_rows"]))  # fmt: skip

    scored = scored_oot
    gender = feats.loc[ids["oot"], "APP_CODE_GENDER"].astype("string")
    by_gender = (scored.groupby(gender)[["pd_lightgbm", "TARGET"]].mean()
                 .query("TARGET > 0").assign(pd_to_dr=lambda d: d["pd_lightgbm"] / d["TARGET"]))  # fmt: skip
    by_gender = by_gender[scored.groupby(gender).size().reindex(by_gender.index) >= 1000]

    out = artefact_dir(view, "explain")
    files = {"shap_oot": out / "shap_oot.parquet", "importance": out / "shap_importance.csv",
             "legs": out / "shap_by_leg.csv", "reason_codes": out / "reason_codes_oot.parquet",
             "monotonicity": out / "monotonicity_check.csv",
             "reason_families": out / "reason_families_oot.parquet",
             "family_map": out / "reason_family_map.json"}  # fmt: skip
    sh_oot.astype("float32").to_parquet(files["shap_oot"])
    imp.to_csv(files["importance"])
    legs.rename("share").to_csv(files["legs"])
    reasons.to_parquet(files["reason_codes"], index=False)
    viol.to_csv(files["monotonicity"], index=False)
    fam_reasons.to_parquet(files["reason_families"], index=False)
    files["family_map"].write_text(json.dumps(fmap, indent=2))

    top = imp.head(10)
    return {
        "explain": {
            "method": "exact TreeSHAP (LightGBM pred_contrib), log-odds",
            "applicants_explained": int(len(sh_oot)),
            "additivity_max_error": add_err,
            "top_features": [
                {
                    "feature": f,
                    "share": float(r["share"]),
                    "leg": str(leg_of.get(f, "?")),
                    "constraint": {
                        1: "PD can only rise with it",
                        -1: "PD can only fall with it",
                        0: "unconstrained",
                    }[mono.get("signs", {}).get(f, 0)],
                }
                for f, r in top.iterrows()
            ],  # fmt: skip
            "leg_shares": {str(k_): float(v) for k_, v in legs.items()},
            "external_score_share": float(imp.loc[imp.index.isin(EXTERNAL), "share"].sum()),
            "stability": stab,
            "monotone": {
                "enabled": bool(mono.get("enabled")),
                "method": mono.get("method"),
                "features_tested": int(len(viol)),
                "violating_rows": int(viol["violating_rows"].sum()) if len(viol) else 0,
                "rows_swept": int(len(sweep_rows)),
            },  # fmt: skip
            "reason_codes_per_applicant": k,
            "family_shares": {labels[f]: float(v) for f, v in fam_share.items()},
            "primary_reason_top_decile": {str(r): float(v) for r, v in primary_share.items()},
            "families": len(fams),
            "gender_pd_to_dr": {str(g): float(v) for g, v in by_gender["pd_to_dr"].items()},
            "paths": {k_: str(v) for k_, v in files.items()},
        }
    }
