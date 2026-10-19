"""Grade-level monotone recalibration: preserves rank order, improves calibration, train-OOF only."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acr.core.calibration import fit_grade_recalibrator
from acr.core.metrics import ece, gini
from acr.graph import build_graph

BOUNDS = [0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.10, 0.15, 0.25, 1.0]


def _miscalibrated(n, seed):
    rng = np.random.default_rng(seed)
    true_p = 1 / (1 + np.exp(-rng.normal(-2.6, 1.0, n)))
    y = (rng.uniform(size=n) < true_p).astype(int)
    shown = true_p**1.15  # systematically under-predicts risk
    return y, shown


def test_rank_order_is_preserved_exactly():
    y, p = _miscalibrated(60_000, 0)
    rec = fit_grade_recalibrator(y, p, BOUNDS)
    q = rec.apply(p)
    assert pd.Series(p).corr(pd.Series(q), method="spearman") == pytest.approx(1.0)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= 0)
    assert gini(y, q) == pytest.approx(gini(y, p))


def test_calibration_improves_on_held_out_data():
    y_fit, p_fit = _miscalibrated(60_000, 1)
    y_new, p_new = _miscalibrated(60_000, 2)
    rec = fit_grade_recalibrator(y_fit, p_fit, BOUNDS)
    assert ece(y_new, rec.apply(p_new)) < ece(y_new, p_new) / 2


def test_sparse_grades_get_no_anchor_and_extrapolation_is_monotone():
    y, p = _miscalibrated(20_000, 3)
    rec = fit_grade_recalibrator(y, p, BOUNDS, min_n=500)
    assert rec.table and all(row["n"] >= 500 for row in rec.table)
    extremes = rec.apply(np.array([1e-5, 1e-4, 0.5, 0.9, 0.999]))
    assert np.all(np.diff(extremes) > 0)


def test_pipeline_recalibrates_champion_without_changing_gini(init_state):
    state = init_state(sample_frac=None)
    state["config"]["base"]["model"]["lightgbm"]["min_child_samples"] = 50  # small synthetic data
    state["config"]["base"]["model"]["calibration"]["grade_min_n"] = 50
    final = build_graph().invoke(state, {"recursion_limit": 60})
    s = pd.read_parquet(final["model"]["paths"]["scored"])
    assert final["model"]["grade_recalibration"]["applied"]
    assert s["pd_champion"].corr(s["pd_champion_uncalibrated"], method="spearman") == pytest.approx(
        1.0
    )
    rc = final["validation"]["metrics"]["recalibration"]
    assert rc["before"]["gini"] == pytest.approx(rc["after"]["gini"], abs=1e-9)
    assert any(
        f["title"].startswith("Grade-level recalibration") for f in final["validation"]["findings"]
    )


def test_too_little_data_leaves_pds_unchanged():
    y, p = _miscalibrated(800, 4)
    rec = fit_grade_recalibrator(y, p, BOUNDS, min_n=500)
    assert rec.identity
    np.testing.assert_allclose(rec.apply(p), p)
