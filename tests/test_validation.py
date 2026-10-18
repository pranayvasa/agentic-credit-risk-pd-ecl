"""Independent validation: gate logic, read-only enforcement, findings and statistical helpers."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import acr.nodes.validate as validate_mod
from acr.config import load_config
from acr.core import validation as V
from acr.core.findings import build_findings, gate_decision
from acr.graph import build_graph
from acr.state import Finding

CFG = load_config()


def _metrics(test_gini=0.50, oot_gini=0.50, psi=0.001, integrity=True):
    perf = {"n": 1000, "gini": 0.52}
    return {
        "integrity": {"ok": integrity, "mismatches": [] if integrity else ["scored"]},
        "champion": {
            "performance": {"train": {**perf, "gini": 0.52}, "test": {**perf, "gini": test_gini},
                            "oot": {**perf, "gini": oot_gini}},
            "calibration": {"oot": {"hl_p_value": 0.5, "hl_stat": 5.0, "ece": 0.002, "slope": 1.0}},
            "rank_breaks_grade_oot": 0, "rank_breaks_decile_oot": 0,
        },
        "challenger": {"performance": {"train": {"gini": 0.60}, "oot": {"gini": 0.56}}},
        "comparison": {"diff": 0.01, "ci_low": -0.01, "ci_high": 0.03},
        "stability": {
            "score_psi": {"train_vs_test": psi, "train_vs_oot": psi},
            "stress_psi": {"under30_x3": 0.02, "under30_x10": 0.13, "under30_x50": 0.34},
        },
        "external_score_share": 0.3,
    }  # fmt: skip


EMPTY_GRADES = pd.DataFrame({"grade": [5], "n": [5000], "share": [0.2], "default_rate": [0.05],
                             "mean_pd": [0.05], "traffic_light": ["green"]})  # fmt: skip
EMPTY_CSI = pd.DataFrame(columns=["feature", "csi_test", "csi_oot"])
EMPTY_SEG = pd.DataFrame(columns=["segment", "value", "n", "gini", "pd_to_dr"])


def _findings(m, sample=False):
    gate = CFG.gate_set(sample)
    return build_findings(m, EMPTY_GRADES, EMPTY_CSI, EMPTY_SEG, CFG.gates.validation, gate, sample)


@pytest.mark.parametrize("kwargs, area", [
    ({"test_gini": 0.30}, "Performance"),
    ({"oot_gini": 0.44}, "Performance"),
    ({"psi": 0.30}, "Stability"),
    ({"integrity": False}, "Integrity"),
])  # fmt: skip
def test_red_conditions_halt_when_enforced(kwargs, area):
    f = _findings(_metrics(**kwargs))
    assert any(x.severity == "red" and x.area == area for x in f)
    assert gate_decision(f, CFG.gate_set(False))[0] == "halt"
    assert gate_decision(f, CFG.gate_set(True))[0] == "warn"  # --sample: warn only


def test_healthy_model_continues():
    f = _findings(_metrics())
    assert not [x for x in f if x.severity == "red"]
    assert gate_decision(f, CFG.gate_set(False))[0] == "continue"


def test_gini_just_above_gate_is_amber():
    f = _findings(_metrics(test_gini=0.47))
    assert any(x.severity == "medium" and "close to gate" in x.title for x in f)


def test_psi_amber_band():
    f = _findings(_metrics(psi=0.15))
    assert any(x.severity == "medium" and x.title == "Score distribution shift" for x in f)


def test_standing_findings_always_present():
    titles = {x.title for x in _findings(_metrics())}
    assert "Stability testing is not temporal (pseudo-OOT)" in titles
    assert "Target is not the regulatory default definition" in titles


def test_finding_ids_unique_and_schema_valid():
    f = _findings(_metrics(test_gini=0.3, psi=0.3))
    assert len({x.id for x in f}) == len(f)
    assert all(isinstance(x, Finding) for x in f)


# --- read-only / independence ------------------------------------------------------------


def test_validator_imports_no_fitting_code():
    src = inspect.getsource(validate_mod)
    for forbidden in ("fit_scorecard", "fit_challenger", "select_features", "fit_binning",
                      "acr.core.scorecard", "acr.core.challenger", "acr.core.calibration",
                      "LGBMClassifier", ".fit("):  # fmt: skip
        assert forbidden not in src, forbidden


def test_tampered_model_artefact_halts_run(init_state, monkeypatch):
    """Simulate an artefact changed between modelling and validation: the validator's
    checksum control must raise a red finding, and a full (enforced) run must halt."""
    import acr.graph as graph_mod
    from acr.nodes.model import model as real_model

    def tampering_model(state):
        out = real_model(state)
        p = out["model"]["paths"]["scored"]
        df = pd.read_parquet(p)
        df["pd_champion"] = df["pd_champion"] * 1.01
        df.to_parquet(p)
        return out

    monkeypatch.setattr(graph_mod, "model", tampering_model)
    final = build_graph().invoke(init_state(sample_frac=None), {"recursion_limit": 60})
    assert final["halted"] is True
    reds = [f for f in final["validation"]["findings"] if f["severity"] == "red"]
    assert any(f["area"] == "Integrity" for f in reds)
    assert "ecl" not in final


def test_validation_leaves_artefacts_unchanged(init_state):
    from acr.nodes._util import sha256

    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    m = final["model"]
    assert all(sha256(p) == m["checksums"][k] for k, p in m["paths"].items())
    assert final["validation"]["metrics"]["integrity"]["ok"] is True


# --- statistical helpers ------------------------------------------------------------------


def test_grade_binomial_flags_under_prediction():
    df = pd.DataFrame({"grade": [1] * 2000 + [2] * 2000,
                       "pd": [0.02] * 2000 + [0.05] * 2000,
                       "TARGET": [1] * 40 + [0] * 1960 + [1] * 160 + [0] * 1840})  # fmt: skip
    g = V.grade_table(df, "pd", "grade", 0.01, 0.05).set_index("grade")
    assert g.loc[1, "traffic_light"] == "green"  # 2.0% observed vs 2% PD
    assert g.loc[2, "traffic_light"] == "red"  # 8.0% observed vs 5% PD


def test_rank_order_breaks_counts_reversals():
    assert V.rank_order_breaks(pd.Series([0.01, 0.02, 0.015, 0.05]), pd.Series([500] * 4)) == 1
    assert V.rank_order_breaks(pd.Series([0.01, 0.02, 0.015]), pd.Series([500, 500, 50])) == 0


def test_stress_psi_exceeds_unstressed():
    rng = np.random.default_rng(0)
    n = 20_000
    age = pd.Series(rng.uniform(20, 70, n))
    score = 500 + 2 * age + rng.normal(0, 20, n)
    part = np.where(rng.uniform(size=n) < 0.5, "train", "oot")
    scored = pd.DataFrame({"score": score, "partition": part})
    base = V.score_psi(scored)["train_vs_oot"]
    assert V.stress_psi(scored, age, 3.0) > max(0.05, 10 * base)


def test_bootstrap_is_deterministic():
    rng = np.random.default_rng(1)
    y = pd.Series((rng.uniform(size=3000) < 0.1).astype(int))
    a, b = pd.Series(rng.uniform(size=3000)), pd.Series(rng.uniform(size=3000))
    assert V.bootstrap_gini_diff(y, a, b, 50, 7) == V.bootstrap_gini_diff(y, a, b, 50, 7)


def test_stress_ladder_reported_with_bands():
    f = [x for x in _findings(_metrics()) if x.title.startswith("Stability testing")][0]
    assert "3x: 0.0200 (green)" in f.detail
    assert "10x: 0.1300 (amber)" in f.detail
    assert "50x: 0.3400 (red)" in f.detail
