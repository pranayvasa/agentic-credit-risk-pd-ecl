"""Selection and modelling: train-only fitting (leakage), reproducibility and scorecard maths."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from acr.config import load_config
from acr.core.guards import LeakageError, require_partition
from acr.core.master_scale import assign_grade
from acr.core.metrics import ece, gini, hosmer_lemeshow, ks, psi
from acr.graph import build_graph
from synthetic import make_tables, write_raw


@pytest.fixture(scope="module")
def base_run(tmp_path_factory):
    from acr.config import load_config as lc

    tmp = tmp_path_factory.mktemp("base")
    raw = write_raw(tmp / "raw", make_tables(n=8000))
    cfg = lc().model_dump(mode="json")
    cfg["base"]["paths"].update(raw_dir=str(raw), interim_dir=str(tmp / "interim"),
                                outputs_dir=str(tmp / "out"))  # fmt: skip
    return build_graph().invoke({"config": cfg, "sample_frac": None, "llm_mode": "dry-run"},
                                {"recursion_limit": 60})  # fmt: skip


def _scored(run):
    return pd.read_parquet(run["model"]["paths"]["scored"])


# --- leakage controls -------------------------------------------------------------------


def test_guard_rejects_non_train_rows():
    a = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "partition": ["train", "test", "oot"]})
    frame = pd.DataFrame({"x": [0, 1]}, index=pd.Index([1, 2], name="SK_ID_CURR"))
    with pytest.raises(LeakageError, match="test"):
        require_partition(frame, a, "train")
    require_partition(frame.loc[[1]], a, "train")


def test_permuting_test_and_oot_targets_changes_nothing_learned(tmp_path, base_run):
    """Re-run selection and modelling on the same split with test and OOT targets shuffled.
    If any fitter saw holdout targets, bins, selection, coefficients or PDs would change."""
    from acr.nodes.model import model as model_node
    from acr.nodes.select import select as select_node

    a = pd.read_parquet(base_run["split"]["assignment_path"])
    rng = np.random.default_rng(5)
    for part in ("test", "oot"):
        m = a["partition"] == part
        a.loc[m, "TARGET"] = rng.permutation(a.loc[m, "TARGET"].to_numpy())
    assert not a["TARGET"].equals(pd.read_parquet(base_run["split"]["assignment_path"])["TARGET"])
    a.to_parquet(tmp_path / "assignment.parquet", index=False)

    state = {"run_id": "perm", "run_dir": str(tmp_path), "config": base_run["config"],
             "split": {**base_run["split"], "assignment_path": str(tmp_path / "assignment.parquet")},
             "legs": base_run["legs"]}  # fmt: skip
    (tmp_path / "artefacts").mkdir()
    state.update(select_node(state))
    state.update(model_node(state))

    pd.testing.assert_frame_equal(pd.read_csv(base_run["selection"]["paths"]["binning_tables"]),
                                  pd.read_csv(state["selection"]["paths"]["binning_tables"]))  # fmt: skip
    assert base_run["selection"]["selected"] == state["selection"]["selected"]
    assert (
        base_run["model"]["champion"]["coefficients"] == state["model"]["champion"]["coefficients"]
    )
    s1, s2 = _scored(base_run), _scored(state)
    np.testing.assert_allclose(s1["pd_champion"], s2["pd_champion"])
    np.testing.assert_allclose(s1["pd_challenger"], s2["pd_challenger"])
    np.testing.assert_allclose(s1["pd_origination_proxy"], s2["pd_origination_proxy"])


def test_exclusion_log_has_reasons_and_protected_features(base_run):
    ex = pd.read_csv(base_run["selection"]["paths"]["exclusions"])
    assert ex["reason"].notna().all() and ex["stage"].notna().all()
    role = ex[ex["stage"] == "role"]["feature"].tolist()
    assert set(role) == {"APP_CODE_GENDER", "APP_NAME_FAMILY_STATUS"}
    selected = {s["feature"] for s in base_run["selection"]["selected"]}
    assert not selected & set(ex["feature"])


# --- reproducibility --------------------------------------------------------------------


def test_same_inputs_give_identical_results(base_run, tmp_path):
    cfg = json.loads(json.dumps(base_run["config"]))
    cfg["base"]["paths"]["outputs_dir"] = str(tmp_path / "out2")
    run2 = build_graph().invoke({"config": cfg, "sample_frac": None, "llm_mode": "dry-run"},
                                {"recursion_limit": 60})  # fmt: skip
    pd.testing.assert_frame_equal(_scored(base_run), _scored(run2))
    for k in ("scored", "coefficients", "points", "challenger", "master_scale"):
        assert base_run["model"]["checksums"][k] == run2["model"]["checksums"][k], k


# --- scorecard behaviour ----------------------------------------------------------------


def test_champion_coefficients_negative_on_woe(base_run):
    coefs = base_run["model"]["champion"]["coefficients"]
    assert coefs and all(v < 0 for v in coefs.values())


def test_points_table_reproduces_scores(base_run):
    import pickle

    from acr.core import binning as B

    paths = base_run["model"]["paths"]
    with open(paths["scorecard"], "rb") as f:
        sc = pickle.load(f)
    pts = pd.read_csv(paths["points"])
    feats = pd.read_parquet(base_run["selection"]["paths"]["features_all"])
    feats = feats.loc[
        _scored(base_run).index
    ]  # labelled rows (features also cover application_test)
    with open(base_run["selection"]["paths"]["binning"], "rb") as f:
        bp = pickle.load(f)
    with open(base_run["selection"]["paths"]["meta"]) as f:
        meta = json.load(f)
    live = list(bp.variable_names)
    idx = B.bins(bp, feats[live], [c for c in live if c in meta["categorical"]])
    total = np.zeros(len(feats))
    for f in sc.features:
        lut = pts[pts["feature"] == f].set_index("bin_index")["points"]
        total += idx[f].map(lut).to_numpy()
    np.testing.assert_allclose(total, _scored(base_run)["score"].to_numpy(), rtol=1e-6)


def test_higher_score_means_lower_pd(base_run):
    s = _scored(base_run)
    assert s["score"].corr(s["pd_champion_raw"], method="spearman") == pytest.approx(-1.0)


def test_models_discriminate_on_synthetic_signal(base_run):
    s = _scored(base_run)
    oot = s[s["partition"] == "oot"]
    # True signal is weak by design (max achievable Gini about 0.2); the point is that it is found.
    assert gini(oot["TARGET"], oot["pd_champion"]) > 0.05
    selected = {x["feature"] for x in base_run["selection"]["selected"]}
    assert selected & {"APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3",
                       "APP_EXT_SOURCE_MEAN", "APP_EXT_SOURCE_MIN"}  # fmt: skip


def test_origination_proxy_uses_application_leg_only(base_run):
    feats = base_run["model"]["origination_proxy"]["features"]
    assert feats and all(f.startswith("APP_") for f in feats)


def test_challenger_uses_fixed_config_hyperparameters(base_run):
    cfg = load_config().base.model.lightgbm
    import lightgbm as lgb

    booster = lgb.Booster(model_file=base_run["model"]["paths"]["challenger"])
    params = booster.dump_model()["tree_info"]
    assert len(params) == base_run["model"]["challenger"]["n_trees"] <= cfg["n_estimators"]


# --- master scale and metrics -----------------------------------------------------------


def test_master_scale_boundaries():
    bounds = [0.01, 0.05, 1.0]
    assert assign_grade(np.array([0.0, 0.01, 0.0100001, 0.05, 0.9, 1.0]), bounds).tolist() == [
        1, 1, 2, 2, 3, 3]  # fmt: skip


def test_metric_sanity():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.5, 20_000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    assert gini(y, p) > 0.3 and 0 < ks(y, p) < 1
    assert ece(y, p) < 0.01
    assert hosmer_lemeshow(y, p)[1] > 0.001
    same = pd.Series(rng.integers(0, 5, 10_000))
    assert psi(same, same) == pytest.approx(0.0, abs=1e-12)
    shifted = pd.Series(np.r_[np.zeros(8000), np.ones(2000)])
    assert psi(same.clip(0, 1), shifted) > 0.25


def test_excluded_features_never_reach_either_model(init_state):
    import lightgbm as lgb

    ext = ["APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3", "APP_EXT_SOURCE_MEAN",
           "APP_EXT_SOURCE_MIN", "APP_EXT_SOURCE_MISSING_COUNT"]  # fmt: skip
    state = init_state(sample_frac=None)
    state["config"]["base"]["features"]["exclude_features"] = ext
    final = build_graph().invoke(state, {"recursion_limit": 60})
    ex = pd.read_csv(final["selection"]["paths"]["exclusions"])
    assert set(ex.loc[ex["stage"] == "config", "feature"]) == set(ext)
    assert not {s["feature"] for s in final["selection"]["selected"]} & set(ext)
    assert not set(final["model"]["origination_proxy"]["features"]) & set(ext)
    booster = lgb.Booster(model_file=final["model"]["paths"]["challenger"])
    assert not set(booster.feature_name()) & set(ext)


def test_exclude_list_defaults_to_empty():
    assert load_config().base.features.exclude_features == []
