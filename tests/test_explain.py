"""Explainability: monotonic constraints, exact TreeSHAP, reason codes, stability, findings."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from acr.core import explain as E
from acr.graph import build_graph


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    # Non-monotone truth: risk dips in the middle. An unconstrained model will follow the dip.
    p = 0.05 + 0.25 * (x - 0.5) ** 2 + 0.15 * x
    y = (rng.uniform(size=n) < p).astype(int)
    X = pd.DataFrame({"x": x.astype("float32"), "noise": rng.normal(size=n).astype("float32")})
    return X, y


def _fit(X, y, constraints=None):
    params = dict(n_estimators=80, num_leaves=15, min_child_samples=50, learning_rate=0.1,
                  random_state=0, deterministic=True, force_row_wise=True, verbose=-1)  # fmt: skip
    if constraints:
        params |= {
            "monotone_constraints": constraints,
            "monotone_constraints_method": "intermediate",
        }
    return lgb.LGBMClassifier(**params).fit(X, y).booster_


def test_violation_check_catches_unconstrained_and_passes_constrained():
    X, y = _toy()
    free = E.monotonicity_violations(_fit(X, y), X.head(300), {"x": 1})
    held = E.monotonicity_violations(_fit(X, y, [1, 0]), X.head(300), {"x": 1})
    assert free["violating_rows"].iloc[0] > 0
    assert held["violating_rows"].iloc[0] == 0


def test_shap_is_exact_and_reason_codes_are_positive_drivers():
    X, y = _toy()
    booster = _fit(X, y, [1, 0])
    sh = E.shap_frame(booster, X)
    assert E.additivity_error(booster, X, sh) < 1e-9
    rc = E.reason_codes(sh, X, k=2)
    assert (rc["shap_logodds"] > 0).all()
    assert rc.groupby("SK_ID_CURR")["rank"].max().max() <= 2
    imp = E.global_importance(sh)
    assert imp.index[0] == "x" and imp["share"].sum() == pytest.approx(1.0)


def test_stability_identical_samples():
    X, y = _toy()
    imp = E.global_importance(E.shap_frame(_fit(X, y), X))
    s = E.stability(imp, imp, top_n=2)
    assert s["top_overlap"] == 1.0 and s["spearman"] == pytest.approx(1.0)


def test_explain_node_end_to_end(init_state):
    state = init_state(sample_frac=None)
    state["config"]["base"]["model"]["lightgbm"]["min_child_samples"] = 50  # small synthetic data
    final = build_graph().invoke(state, {"recursion_limit": 60})
    ex = final["explain"]
    a = pd.read_parquet(final["split"]["assignment_path"])
    assert ex["applicants_explained"] == (a["partition"] == "oot").sum()
    assert ex["additivity_max_error"] < 1e-9
    assert ex["monotone"]["enabled"] and ex["monotone"]["violating_rows"] == 0
    assert sum(ex["leg_shares"].values()) == pytest.approx(1.0)
    titles = {f["title"] for f in final["validation"]["findings"]}
    assert "Monotonic constraints verified" in titles
    rc = pd.read_parquet(ex["paths"]["reason_codes"])
    assert set(rc["SK_ID_CURR"]) <= set(a.loc[a["partition"] == "oot", "SK_ID_CURR"])


def test_constraint_violation_is_a_high_finding():
    from acr.config import load_config
    from acr.core.findings import build_findings
    from tests.test_validation import EMPTY_CSI, EMPTY_GRADES, EMPTY_SEG, _metrics

    cfg = load_config()
    m = _metrics()
    m["explain"] = {"monotone": {"enabled": True, "violating_rows": 3, "features_tested": 5, "rows_swept": 10},
                    "additivity_max_error": 0.0, "external_score_share": 0.2, "gender_pd_to_dr": {},
                    "stability": {"test_vs_oot": {"top_n": 10, "top_overlap": 0.9, "spearman": 0.95}}}  # fmt: skip
    f = build_findings(
        m, EMPTY_GRADES, EMPTY_CSI, EMPTY_SEG, cfg.gates.validation, cfg.gate_set(False), False
    )
    assert any(x.severity == "high" and x.title == "Monotonic constraints violated" for x in f)


def test_degenerate_lightgbm_is_flagged_not_called_stable(init_state):
    final = build_graph().invoke(init_state(), {"recursion_limit": 60})  # tiny sample: no splits
    titles = {f["title"] for f in final["validation"]["findings"]}
    assert "LightGBM explanations are stable" not in titles
    assert "LightGBM explanations could not be assessed" in titles
