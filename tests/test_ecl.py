"""ECL layer: annuity maths, exposure, staging, lifetime PD, engine, floors and the graph."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acr.config import load_config
from acr.core.ecl import lifetime as L
from acr.core.ecl import rates as R
from acr.core.ecl.engine import Book, floors_for, unit_ecl
from acr.core.ecl.staging import allocate
from acr.graph import build_graph

E = load_config().ecl


# --- annuity maths and exposure -----------------------------------------------------------


def test_rate_and_term_round_trip():
    P, r, n = np.array([100_000.0, 50_000.0]), np.array([0.02, 0.035]), np.array([24, 12])
    A = R.annuity(P, r, n)
    np.testing.assert_allclose(R.solve_rate(P, A, n), r, rtol=1e-6)
    assert R.term_months(P, A, r, 3, 120).tolist() == [24, 12]


def test_zero_interest_and_uncovered_interest():
    assert R.solve_rate(np.array([1200.0]), np.array([100.0]), np.array([12]))[0] == 0.0
    # Instalment below the monthly interest never repays: term goes to the maximum.
    assert (
        R.term_months(np.array([100_000.0]), np.array([1_000.0]), np.array([0.02]), 3, 120)[0]
        == 120
    )


def test_balance_schedule_amortises_to_zero():
    P, r, n = np.array([10_000.0]), np.array([0.03]), np.array([12])
    A = R.annuity(P, r, n)
    b = R.balance_schedule(P, A, r, n, horizon=15)[0]
    assert b[0] == pytest.approx(10_000.0)
    assert np.all(np.diff(b[:12]) < 0)
    assert b[11] == pytest.approx(A[0] / (1 + r[0]), rel=1e-6)  # last instalment clears it
    assert np.all(b[12:] == 0)


def test_implied_rates_use_approved_loans_with_terms():
    prev = pd.DataFrame({
        "NAME_CONTRACT_TYPE": ["Cash loans"] * 3 + ["Consumer loans"],
        "NAME_CONTRACT_STATUS": ["Approved", "Approved", "Refused", "Approved"],
        "AMT_CREDIT": [10_000.0] * 4,
        "AMT_ANNUITY": list(R.annuity(np.full(4, 10_000.0), np.array([0.02, 0.02, 0.5, 0.01]), np.full(4, 12))),
        "CNT_PAYMENT": [12, 12, 12, 12],
    })  # fmt: skip
    rates = R.implied_rates(prev)
    assert rates["Cash loans"] == pytest.approx(0.02, rel=1e-5)  # refused loan ignored
    assert rates["Consumer loans"] == pytest.approx(0.01, rel=1e-5)


# --- staging ------------------------------------------------------------------------------


def _stage_frame(**over):
    base = {"BEH_INS_DPD_MAX_12M": 0.0, "BEH_POS_DPD_MAX_12M": 0.0, "BUR_BB_WORST_BUCKET_12M": 0.0,
            "BUR_DAYS_OVERDUE_MAX": 0.0, "grade_champion": 5, "grade_origination_proxy": 5,
            "pd_champion": 0.04, "pd_origination_proxy": 0.04}  # fmt: skip
    return pd.DataFrame([{**base, **over}])


@pytest.mark.parametrize("over, stage, trigger", [
    ({}, 1, ""),
    ({"BEH_INS_DPD_MAX_12M": 95}, 3, "hc_90dpd"),
    ({"BEH_POS_DPD_MAX_12M": 45}, 2, "hc_30dpd"),
    ({"BUR_BB_WORST_BUCKET_12M": 2}, 2, "bureau_30dpd"),
    ({"BUR_DAYS_OVERDUE_MAX": 31}, 2, "bureau_30dpd"),
    ({"grade_champion": 8, "grade_origination_proxy": 5}, 2, "grade_downgrade"),
    ({"pd_champion": 0.12, "pd_origination_proxy": 0.05}, 2, "pd_ratio"),
    ({"pd_champion": 0.04, "pd_origination_proxy": 0.01}, 1, ""),  # ratio met, absolute floor not
    ({"BEH_INS_DPD_MAX_12M": 30}, 1, ""),                           # exactly 30 is not "more than 30"
])  # fmt: skip
def test_stage_rules(over, stage, trigger):
    out = allocate(_stage_frame(**over), E.staging)
    assert out["stage"].iloc[0] == stage
    assert trigger in out["triggers"].iloc[0]


def test_staging_lookback_must_match_feature_windows():
    s = {**E.staging, "lookback_months": 24}
    with pytest.raises(ValueError, match="12-month"):
        allocate(_stage_frame(), s)


# --- lifetime PD --------------------------------------------------------------------------


def test_constant_hazard_reproduces_12m_pd():
    pd12 = np.array([0.01, 0.1, 0.5])
    m = L.marginal(L.constant_hazards(pd12, 36))
    np.testing.assert_allclose(m[:, :12].sum(axis=1), pd12, rtol=1e-9)
    assert np.all(m.sum(axis=1) <= 1)


def _toy_bb():
    rng = np.random.default_rng(0)
    rows = []
    for b in range(400):
        s = 0
        for t in range(-24, 0):
            rows.append((b, t, float(s)))
            s = (
                min(5, s + 1)
                if rng.uniform() < 0.1
                else (0 if s < 4 and rng.uniform() < 0.5 else s)
            )
    return pd.DataFrame(rows, columns=["SK_ID_BUREAU", "MONTHS_BALANCE", "bucket"])


def test_transition_matrix_rows_sum_to_one_and_default_absorbs():
    P = L.transition_matrix(_toy_bb(), default_bucket=4)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    assert P.loc["D", "D"] == 1.0
    assert list(P.index) == ["0", "1", "2", "3", "D"]


def test_markov_scaling_matches_12m_pd():
    profile = L.markov_profile(L.transition_matrix(_toy_bb(), 4), 60)
    pd12 = np.array([0.02, 0.08, 0.3])
    m = L.marginal(L.scaled_hazards(profile, pd12))
    np.testing.assert_allclose(m[:, :12].sum(axis=1), pd12, rtol=1e-6)


# --- engine and floors --------------------------------------------------------------------


def _book(stage, term=24):
    n = len(stage)
    P, r = np.full(n, 10_000.0), np.full(n, 0.03)
    A = R.annuity(P, r, np.full(n, term))
    ead = R.balance_schedule(P, A, r, np.full(n, term), term)
    return Book(stage=np.array(stage), pd12=np.full(n, 0.10), ead=ead, rate=r,
                term=np.full(n, term), pd_floor=0.0003)  # fmt: skip


def test_stage3_unit_ecl_is_full_exposure():
    res = unit_ecl(_book([3]), 1.0, "constant_hazard", None)
    assert res["unit_ecl"][0] == pytest.approx(10_000.0) and res["pd_horizon"][0] == 1.0


def test_stage2_lifetime_exceeds_stage1_12m():
    res = unit_ecl(_book([1, 2]), 1.0, "constant_hazard", None)
    s1, s2 = res["unit_ecl"]
    assert s2 > s1 > 0
    assert res["pd_horizon"][0] == pytest.approx(0.10, rel=1e-9)  # 12-month horizon
    assert res["pd_horizon"][1] > 0.10  # lifetime over 24 months


def test_downside_scenario_raises_ecl_and_pd_floor_applies():
    b = _book([1, 2])
    base = unit_ecl(b, 1.0, "constant_hazard", None)["unit_ecl"]
    down = unit_ecl(b, 1.35, "constant_hazard", None)["unit_ecl"]
    up = unit_ecl(b, 0.85, "constant_hazard", None)["unit_ecl"]
    assert np.all(down > base) and np.all(base > up)
    tiny = Book(**{**b.__dict__, "pd12": np.full(2, 1e-6)})
    assert np.all(unit_ecl(tiny, 1.0, "constant_hazard", None)["pd12_used"] == 0.0003)


def test_rbi_floors_by_stage():
    assert floors_for(np.array([1, 2, 3]), E.floors).tolist() == [0.01, 0.05, 0.25]


# --- graph --------------------------------------------------------------------------------


def test_ecl_node_end_to_end(init_state):
    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    e = final["ecl"]
    assert e["illustrative"] is True and e["regulation"] == "RBI/DOR/2026-27/398"
    acct = pd.read_parquet(e["paths"]["accounts"])
    a = pd.read_parquet(final["split"]["assignment_path"])
    assert set(acct.index) == set(a.loc[a["partition"] == "oot", "SK_ID_CURR"])
    assert (acct["ecl_final"] >= acct["ecl_model"] - 1e-9).all()
    assert (acct["ecl_final"] >= acct["ecl_floor"] - 1e-9).all()
    assert e["checks"]["scenario_weights_sum_to_1"] and e["checks"]["ecl_non_negative"]
    lgd = pd.read_csv(e["paths"]["lgd"]).set_index("lgd")["ecl_model"]
    assert lgd.loc[0.45] / lgd.loc[0.70] == pytest.approx(0.45 / 0.70)  # ECL linear in LGD
    assert sum(e["stage_counts"].values()) == len(acct)
