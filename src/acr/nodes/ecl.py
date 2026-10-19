"""Node 7: illustrative RBI ECL simulation on the pseudo-OOT book.

ILLUSTRATIVE ONLY. Home Credit is an application-level dataset, not a live portfolio. The OOT
partition is treated as a book at a simulated reporting date (seasoning in config). Regulatory
parameters come from config/ecl_rbi.yaml with paragraph citations to RBI/DOR/2026-27/398.

Data-learned inputs (implied rates, the DPD transition matrix) use train customers only.
Nothing here fits or changes the PD models: the ECL layer consumes their scored outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from acr.audit import node
from acr.core.ecl import lifetime as L
from acr.core.ecl import rates as R
from acr.core.ecl.engine import Book, floors_for, unit_ecl
from acr.core.ecl.staging import INPUTS, allocate
from acr.core.io import read
from acr.core.legs.bureau import dpd_buckets
from acr.core.schema import ID
from acr.nodes._util import artefact_dir, cfg

METHODS = ("constant_hazard", "markov")


def _r(x, nd=6):
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _records(df: pd.DataFrame) -> list[dict]:
    return _r(df.reset_index().to_dict(orient="records"))


@node(
    "ecl",
    role="Illustrative RBI ECL on the OOT book: stage allocation with audited triggers, "
    "12-month and lifetime PD (constant hazard and bureau DPD Markov chain), amortising EAD, "
    "EIR discounting, weighted macro scenarios, LGD backstop and sensitivity, RBI stage floors.",
    reads=["config", "split", "dq", "selection", "model"],
    writes=["ecl"],
)
def ecl(view):
    c = cfg(view)
    e = c.ecl
    tables = {k: v["path"] for k, v in view["dq"]["tables"].items()}
    a = pd.read_parquet(view["split"]["assignment_path"]).set_index(ID)
    book_ids = a.index[a["partition"] == e.book.partition]
    train_ids = a.index[a["partition"] == "train"]

    # --- PDs and grades from the champion (consumed, never refitted) ---
    scored = pd.read_parquet(view["model"]["paths"]["scored"]).loc[book_ids]
    feats = pd.read_parquet(view["selection"]["paths"]["features_all"], columns=INPUTS).loc[
        book_ids
    ]
    df = scored.join(feats)

    # --- contract data at float64 precision ---
    app = read(tables["application"], [ID, "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY"],
               book_ids).set_index(ID).loc[book_ids]  # fmt: skip
    ctype = app["NAME_CONTRACT_TYPE"].astype("string")
    revolving = (ctype == "Revolving loans").to_numpy()

    # --- implied monthly rates from prior approved loans of TRAIN customers ---
    prev = read(tables["previous"], ["NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "AMT_CREDIT",
                                     "AMT_ANNUITY", "CNT_PAYMENT"], train_ids)  # fmt: skip
    implied = R.implied_rates(prev)
    del prev
    rate = (
        ctype.map(lambda t: implied.get(t, e.term.fallback_monthly_rate)).astype(float).to_numpy()
    )

    # --- term and exposure schedule ---
    P = app["AMT_CREDIT"].to_numpy(dtype="float64")
    A = app["AMT_ANNUITY"].to_numpy(dtype="float64")
    term = R.term_months(P, np.nan_to_num(A, nan=1.0), rate, e.term.min_months, e.term.max_months)
    median_cash_term = int(np.median(term[~revolving & ~np.isnan(A)])) if (~revolving).any() else 12
    term = np.where(np.isnan(A), median_cash_term, term)
    term = np.where(revolving, e.revolving.life_months, term)
    s0 = e.book.seasoning_months
    H = int(max(term.max() - s0, 1))
    sched = R.balance_schedule(P, np.nan_to_num(A, nan=0.0), rate, term, H + s0)[:, s0:]
    rev_ead = np.where(
        np.arange(H)[None, :] < (term - s0)[:, None], (P * e.revolving.ccf)[:, None], 0.0
    )
    ead = np.where(revolving[:, None], rev_ead, sched)
    remaining = np.maximum(term - s0, 1)

    # --- Markov transition matrix from bureau_balance of TRAIN customers ---
    bur = read(tables["bureau"], ["SK_ID_BUREAU", ID], train_ids)
    bb = read(tables["bureau_balance"], ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
              ids=bur["SK_ID_BUREAU"].unique(), id_col="SK_ID_BUREAU")  # fmt: skip
    bb["bucket"] = dpd_buckets(bb.pop("STATUS"))
    tm = L.transition_matrix(bb, e.lifetime.markov_default_bucket)
    n_transitions = int(len(bb))
    del bb, bur
    profile = L.markov_profile(tm, H)

    # --- staging ---
    st = allocate(df, e.staging)
    stage = st["stage"].to_numpy()
    book = Book(stage=stage, pd12=df["pd_champion"].to_numpy(), ead=ead, rate=rate,
                term=remaining, pd_floor=e.pd["floor_12m"])  # fmt: skip

    # --- ECL per method x scenario (unit ECL at LGD = 1) ---
    lgd = float(e.lgd["base"])
    weights = {k: s.weight for k, s in e.scenarios.items()}
    acct = pd.DataFrame({"stage": stage, "triggers": st["triggers"].to_numpy(),
                         "contract_type": ctype.to_numpy(), "grade": df["grade_champion"].to_numpy(),
                         "grade_origination_proxy": df["grade_origination_proxy"].to_numpy(),
                         "pd12": book.pd12, "rate_monthly": rate, "term_months": remaining,
                         "ead0": book.ead0, "TARGET": df["TARGET"].to_numpy()},
                        index=book_ids)  # fmt: skip
    for method in METHODS:
        weighted = np.zeros(len(acct))
        for name, sc in e.scenarios.items():
            res = unit_ecl(book, sc.pd_scalar, method, profile)
            acct[f"ecl_{method}_{name}"] = res["unit_ecl"] * lgd
            if name == "base":
                acct[f"pd_horizon_{method}"] = res["pd_horizon"]
            weighted += weights[name] * res["unit_ecl"] * lgd
        acct[f"ecl_{method}_weighted"] = weighted

    prim = e.lifetime.primary
    acct["ecl_model"] = acct[f"ecl_{prim}_weighted"]
    acct["floor_rate"] = floors_for(stage, e.floors)
    acct["ecl_floor"] = acct["floor_rate"] * acct["ead0"]
    acct["ecl_final"] = np.maximum(acct["ecl_model"], acct["ecl_floor"])
    acct["floor_binds"] = acct["ecl_floor"] > acct["ecl_model"]

    # --- summaries ---
    def agg(g):
        ead0 = g["ead0"].sum()
        return pd.Series({
            "accounts": len(g), "ead": ead0, "mean_pd12": g["pd12"].mean(),
            "mean_pd_horizon": g[f"pd_horizon_{prim}"].mean(),
            "ecl_model": g["ecl_model"].sum(), "ecl_final": g["ecl_final"].sum(),
            "coverage_model": g["ecl_model"].sum() / ead0 if ead0 else 0.0,
            "coverage_final": g["ecl_final"].sum() / ead0 if ead0 else 0.0,
            "floor_binding_share": g["floor_binds"].mean(),
            "realised_default_rate": g["TARGET"].mean(),
        })  # fmt: skip

    by_stage = acct.groupby("stage").apply(agg, include_groups=False)
    total = agg(acct).rename("total")
    by_stage_t = pd.concat([by_stage, total.to_frame().T])
    by_stage_t.index = [f"stage_{i}" if i != "total" else "total" for i in by_stage_t.index]
    by_grade = acct.groupby("grade").apply(agg, include_groups=False)
    by_scenario = pd.DataFrame({
        f"{name} (w={weights[name]:.2f}, PD x{sc.pd_scalar:.2f})":
            acct.groupby("stage")[f"ecl_{prim}_{name}"].sum() for name, sc in e.scenarios.items()
    })  # fmt: skip
    by_scenario["probability_weighted"] = acct.groupby("stage")["ecl_model"].sum()
    by_scenario.loc["total"] = by_scenario.sum()
    methods = pd.DataFrame({
        m: {"ecl_weighted": acct[f"ecl_{m}_weighted"].sum(),
            "stage1_ecl": acct.loc[acct["stage"] == 1, f"ecl_{m}_weighted"].sum(),
            "stage2_ecl": acct.loc[acct["stage"] == 2, f"ecl_{m}_weighted"].sum(),
            "stage2_mean_lifetime_pd": acct.loc[acct["stage"] == 2, f"pd_horizon_{m}"].mean()}
        for m in METHODS
    }).T  # fmt: skip
    unit_model = acct["ecl_model"] / lgd
    lgd_sens = pd.DataFrame([{
        "lgd": x, "ecl_model": (unit_model * x).sum(),
        "ecl_final": np.maximum(unit_model * x, acct["ecl_floor"]).sum(),
        "coverage_final": np.maximum(unit_model * x, acct["ecl_floor"]).sum() / acct["ead0"].sum(),
    } for x in e.lgd["sensitivity"]])  # fmt: skip
    trig_cols = ["hc_90dpd", "hc_30dpd", "bureau_30dpd", "grade_downgrade", "pd_ratio"]
    triggers = st[trig_cols].sum().astype(int).to_dict()
    migration = pd.crosstab(acct["grade_origination_proxy"], acct["grade"])
    migration.index.name, migration.columns.name = "origination_proxy_grade", "reporting_grade"

    cov = by_stage["coverage_final"]
    # Lifetime PD can only be compared with 12-month PD where the loan runs at least 12 months.
    s2l = acct.loc[(acct["stage"] == 2) & (acct["term_months"] >= 12)].assign(
        pd_lifetime=lambda d: d[f"pd_horizon_{prim}"]
    )
    checks = {
        "scenario_weights_sum_to_1": abs(sum(weights.values()) - 1) < 1e-9,
        "ecl_non_negative": bool((acct["ecl_final"] >= 0).all()),
        "coverage_increases_with_stage": bool(cov.is_monotonic_increasing),
        "realised_default_rate_increases_with_stage": bool(by_stage["realised_default_rate"].is_monotonic_increasing),
        "stage2_lifetime_pd_at_least_12m_pd": bool(
            (s2l["pd_lifetime"] >= s2l["pd12"].clip(lower=e.pd["floor_12m"]) - 1e-9).all()),
    }  # fmt: skip

    out = artefact_dir(view, "ecl")
    paths = {
        "accounts": out / "ecl_accounts.parquet", "by_stage": out / "ecl_by_stage.csv",
        "by_grade": out / "ecl_by_grade.csv", "by_scenario": out / "ecl_by_scenario.csv",
        "methods": out / "lifetime_method_comparison.csv", "lgd": out / "lgd_sensitivity.csv",
        "migration": out / "grade_migration.csv", "transition_matrix": out / "dpd_transition_matrix.csv",
    }  # fmt: skip
    acct.to_parquet(paths["accounts"])
    by_stage_t.to_csv(paths["by_stage"])
    by_grade.to_csv(paths["by_grade"])
    by_scenario.to_csv(paths["by_scenario"])
    methods.to_csv(paths["methods"])
    lgd_sens.to_csv(paths["lgd"], index=False)
    migration.to_csv(paths["migration"])
    tm.to_csv(paths["transition_matrix"])

    return {
        "ecl": {
            "illustrative": True,
            "regulation": e.regulation["reference"],
            "book": {
                "partition": e.book.partition,
                "accounts": int(len(acct)),
                "ead_total": float(acct["ead0"].sum()),
                "seasoning_months": s0,
                "revolving_share": float(revolving.mean()),
            },  # fmt: skip
            "assumptions": {
                "lgd_base": lgd,
                "lgd_source": "para 97 backstop, unsecured portion",
                "pd_floor_12m": e.pd["floor_12m"],
                "floors": e.floors,
                "lifetime_primary": prim,
                "scenarios": {k: v.model_dump() for k, v in e.scenarios.items()},
                "implied_monthly_rates": implied,
                "implied_annual_rates": {k: (1 + v) ** 12 - 1 for k, v in implied.items()},
                "term_months": {"median": float(np.median(remaining)), "max": int(remaining.max())},
                "markov_transitions_source_rows": n_transitions,
            },  # fmt: skip
            "stage_counts": {
                int(k): int(v) for k, v in acct["stage"].value_counts().sort_index().items()
            },
            "stage_triggers": triggers,
            "by_stage": _r(by_stage_t.to_dict(orient="index")),
            "by_scenario": _r(by_scenario.to_dict(orient="index")),
            "lifetime_methods": _r(methods.to_dict(orient="index")),
            "lgd_sensitivity": _records(lgd_sens.set_index("lgd")),
            "transition_matrix": _r(tm.round(6).to_dict(orient="index")),
            "floor_uplift": float(acct["ecl_final"].sum() - acct["ecl_model"].sum()),
            "pd_floor_binding_accounts": int((acct["pd12"] < e.pd["floor_12m"]).sum()),
            "checks": checks,
            "paths": {k: str(v) for k, v in paths.items()},
        }
    }
