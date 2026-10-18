"""Offline tuning: train-only, deterministic, and the one-standard-error winner rule."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from acr.config import load_config
from acr.graph import build_graph
from acr.tuning import LGBMGrid, ScorecardGrid, TuningCfg, load_train, one_se, run
from synthetic import make_tables, write_raw

TINY = TuningCfg(
    folds=2, selection_rule="one_se",
    scorecard=ScorecardGrid(max_n_bins=[4, 6], min_bin_size=[0.05], max_features=[5, 10],
                            iv_min=[0.02]),
    lightgbm=LGBMGrid(max_depth=[3, 10], num_leaves=[7, 15], min_child_samples=[50],
                      colsample_bytree=[0.7], reg_lambda=[1.0]),
)  # fmt: skip


@pytest.fixture(scope="module")
def source_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tune")
    raw = write_raw(tmp / "raw", make_tables(n=4000))
    cfg = load_config().model_dump(mode="json")
    cfg["base"]["paths"].update(raw_dir=str(raw), interim_dir=str(tmp / "interim"),
                                outputs_dir=str(tmp / "out"))  # fmt: skip
    final = build_graph().invoke({"config": cfg, "sample_frac": None, "llm_mode": "dry-run"},
                                 {"recursion_limit": 60})  # fmt: skip
    from acr.config import AppConfig

    return Path(final["run_dir"]), AppConfig.model_validate(cfg)


def test_one_se_picks_simplest_within_one_standard_error():
    res = pd.DataFrame({
        "n_selected": [20, 12, 8], "cv_gini_mean": [0.50, 0.495, 0.40], "cv_gini_se": [0.01, 0.01, 0.01],
    })  # fmt: skip
    win = one_se(res, ["n_selected"], [True])
    assert win["n_selected"] == 12  # within 1 SE of 0.50; the 8-feature model is not


def test_load_train_reads_train_partition_only(source_run):
    run_dir, _ = source_run
    td = load_train(run_dir)
    a = pd.read_parquet(run_dir / "artefacts/split/assignment.parquet")
    assert set(td.features.index) == set(a.loc[a["partition"] == "train", "SK_ID_CURR"])
    assert td.y.index.equals(td.features.index)


def test_tuning_ignores_holdout_targets_and_is_deterministic(source_run, tmp_path):
    run_dir, cfg = source_run
    out1 = run(run_dir, cfg, TINY, log=lambda *_: None)
    # Shuffle test/OOT targets in the source assignment: tuning results must not move.
    ap = run_dir / "artefacts/split/assignment.parquet"
    a = pd.read_parquet(ap)
    original = a.copy()
    hold = a["partition"] != "train"
    a.loc[hold, "TARGET"] = a.loc[hold, "TARGET"].sample(frac=1, random_state=3).to_numpy()
    a.to_parquet(ap, index=False)
    try:
        out2 = run(run_dir, cfg, TINY, log=lambda *_: None)
    finally:
        original.to_parquet(ap, index=False)
    for name in ("scorecard", "lightgbm"):
        pd.testing.assert_frame_equal(pd.read_csv(out1 / f"{name}_results.csv"),
                                      pd.read_csv(out2 / f"{name}_results.csv"))  # fmt: skip
    best = yaml.safe_load((out1 / "best_params.yaml").read_text())
    assert set(best["winner"]) == {"scorecard", "lightgbm"}
    assert "train partition only" in (out1 / "tuning_report.md").read_text()


def test_report_includes_stability_and_depth_views(source_run):
    run_dir, cfg = source_run
    out = run(run_dir, cfg, TINY, log=lambda *_: None)
    sc = pd.read_csv(out / "scorecard_results.csv")
    assert sc["feature_jaccard"].between(0, 1).all()
    gb = pd.read_csv(out / "lightgbm_results.csv")
    assert set(gb["max_depth"]) == {3, 10}
    assert (gb.loc[gb["max_depth"] == 3, "leaves_eff"] <= 8).all()
    report = (out / "tuning_report.md").read_text()
    assert "Best configuration per feature cap" in report and "per max_depth" in report


def test_grids_cannot_exceed_hard_limits():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="15-variable limit"):
        ScorecardGrid(max_n_bins=[4], min_bin_size=[0.05], max_features=[10, 16], iv_min=[0.02])
    for bad in (-1, 0, 11):
        with pytest.raises(ValidationError, match="max_depth"):
            LGBMGrid(max_depth=[bad], num_leaves=[15], min_child_samples=[100],
                     colsample_bytree=[0.7], reg_lambda=[1.0])  # fmt: skip


def test_equivalent_lightgbm_configs_are_not_rerun(source_run):
    run_dir, cfg = source_run
    grid = TINY.model_copy(update={"lightgbm": LGBMGrid(
        max_depth=[3], num_leaves=[15, 31], min_child_samples=[50], colsample_bytree=[0.7],
        reg_lambda=[1.0])})  # fmt: skip
    out = run(run_dir, cfg, grid, which="lightgbm", log=lambda *_: None)
    assert len(pd.read_csv(out / "lightgbm_results.csv")) == 1  # 15 and 31 leaves cap at 8
