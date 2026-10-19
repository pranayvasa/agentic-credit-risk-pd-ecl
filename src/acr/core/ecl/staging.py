"""Stage allocation for the simulated reporting-date book (decisions D5, D6).

Stage 3 (credit-impaired, para 25): more than 90 days past due within the lookback on the
  borrower's prior Home Credit loans (same lender, borrower level).
Stage 2 (SICR, paras 26-34): any of
  * rebuttable presumption (para 33): more than 30 DPD within the lookback on prior Home
    Credit loans, or on bureau credits with other lenders (bureau bucket 2+, or currently
    more than 30 days overdue);
  * relative deterioration: reporting grade at least N notches worse than the
    origination-proxy grade; or PD at least X times the proxy PD *and* above an absolute
    floor. The proxy stands in for an origination-date PD, which this data does not have.
Stage 1: everything else.
Every account records the triggers that fired, so the allocation can be audited.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Leg features the rules read. All use 12-month windows built in the legs.
INPUTS = ["BEH_INS_DPD_MAX_12M", "BEH_POS_DPD_MAX_12M", "BUR_BB_WORST_BUCKET_12M",
          "BUR_DAYS_OVERDUE_MAX"]  # fmt: skip
FEATURE_WINDOW_MONTHS = 12


def allocate(df: pd.DataFrame, s: dict) -> pd.DataFrame:
    """``df`` holds the INPUTS plus grade_champion, grade_origination_proxy, pd_champion and
    pd_origination_proxy. ``s`` is the staging config. Returns stage and trigger columns."""
    if int(s["lookback_months"]) != FEATURE_WINDOW_MONTHS:
        raise ValueError("staging lookback must match the 12-month leg feature windows")
    hc_dpd = df[["BEH_INS_DPD_MAX_12M", "BEH_POS_DPD_MAX_12M"]].max(axis=1).fillna(0)
    t = pd.DataFrame(index=df.index)
    t["hc_90dpd"] = hc_dpd > s["credit_impaired_dpd"]
    t["hc_30dpd"] = hc_dpd > s["sicr_dpd_presumption"]
    t["bureau_30dpd"] = (df["BUR_BB_WORST_BUCKET_12M"].fillna(0) >= 2) | (
        df["BUR_DAYS_OVERDUE_MAX"].fillna(0) > s["sicr_dpd_presumption"])  # fmt: skip
    t["grade_downgrade"] = (df["grade_champion"] - df["grade_origination_proxy"]) >= s[
        "sicr_grade_notches"
    ]
    t["pd_ratio"] = (df["pd_champion"] >= s["sicr_pd_ratio"] * df["pd_origination_proxy"]) & (
        df["pd_champion"] >= s["sicr_abs_pd_min"])  # fmt: skip
    stage = np.where(t["hc_90dpd"], 3, np.where(t[["hc_30dpd", "bureau_30dpd", "grade_downgrade",
                                                    "pd_ratio"]].any(axis=1), 2, 1))  # fmt: skip
    out = t.copy()
    out["stage"] = stage
    names = np.array(t.columns)
    out["triggers"] = [",".join(names[row]) for row in t.to_numpy()]
    return out
