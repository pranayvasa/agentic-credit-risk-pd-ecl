"""Bureau leg: credit with other lenders (bureau) and its monthly status history (bureau_balance)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acr.core.io import read
from acr.core.legs.common import DAYS_PER_MONTH, FeatureDict, LegResult, finalise, ratio
from acr.core.schema import ID

B, BB = "bureau", "bureau_balance"
# bureau_balance STATUS: 0 = no DPD, 1 = 1-30, 2 = 31-60, 3 = 61-90, 4 = 91-120,
# 5 = 120+ or written off; C = closed, X = unknown (both treated as missing bucket).
STATUS_TO_BUCKET = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def dpd_buckets(status: pd.Series) -> pd.Series:
    return status.astype("string").map(STATUS_TO_BUCKET).astype("float32")


def build(paths: dict[str, str], ids: np.ndarray) -> LegResult:
    cols = ["SK_ID_BUREAU", ID, "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
            "DAYS_CREDIT_ENDDATE", "AMT_CREDIT_MAX_OVERDUE", "CNT_CREDIT_PROLONG",
            "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE", "CREDIT_TYPE"]  # fmt: skip
    b = read(paths["bureau"], cols, ids)
    fd = FeatureDict("bureau")
    active = b["CREDIT_ACTIVE"].astype("string") == "Active"
    ctype = b["CREDIT_TYPE"].astype("string")
    g = b.groupby(ID)
    ga = b[active].groupby(ID)
    f = pd.DataFrame(index=pd.Index(g.size().index, name=ID))
    counts = []

    def cnt(name, definition, s):
        counts.append(fd.add(name, definition, B))
        f[name] = s

    cnt("BUR_COUNT", "Number of bureau credits", g.size())
    cnt("BUR_ACTIVE_COUNT", "Number of active bureau credits", active.groupby(b[ID]).sum())
    cnt(
        "BUR_OPENED_12M_COUNT",
        "Bureau credits opened in the 12 months before application",
        (b["DAYS_CREDIT"] >= -365).groupby(b[ID]).sum(),
    )
    cnt(
        "BUR_MICROLOAN_COUNT",
        "Number of bureau microloans",
        (ctype == "Microloan").groupby(b[ID]).sum(),
    )
    cnt(
        "BUR_CREDIT_CARD_COUNT",
        "Number of bureau credit cards",
        (ctype == "Credit card").groupby(b[ID]).sum(),
    )
    cnt(
        "BUR_PROLONG_SUM",
        "Total number of bureau credit prolongations",
        g["CNT_CREDIT_PROLONG"].sum(),
    )
    f[fd.add("BUR_ACTIVE_SHARE", "Active / all bureau credits", B)] = ratio(
        f["BUR_ACTIVE_COUNT"], f["BUR_COUNT"]
    )
    f[fd.add("BUR_MONTHS_SINCE_FIRST", "Months since the oldest bureau credit was opened", B)] = (
        -g["DAYS_CREDIT"].min() / DAYS_PER_MONTH
    )
    f[fd.add("BUR_MONTHS_SINCE_LAST", "Months since the newest bureau credit was opened", B)] = (
        -g["DAYS_CREDIT"].max() / DAYS_PER_MONTH
    )
    debt = ga["AMT_CREDIT_SUM_DEBT"].sum()
    limit = ga["AMT_CREDIT_SUM"].sum()
    f[fd.add("BUR_ACTIVE_DEBT_SUM", "Outstanding debt on active bureau credits", B)] = debt
    f[fd.add("BUR_ACTIVE_DEBT_TO_CREDIT", "Active outstanding debt / active credit amount", B)] = (
        ratio(debt.reindex(f.index), limit.reindex(f.index))
    )
    f[fd.add("BUR_OVERDUE_SUM", "Current overdue amount across bureau credits", B)] = g[
        "AMT_CREDIT_SUM_OVERDUE"
    ].sum()
    f[fd.add("BUR_DAYS_OVERDUE_MAX", "Current maximum days overdue on any bureau credit", B)] = g[
        "CREDIT_DAY_OVERDUE"
    ].max()
    f[fd.add("BUR_AMT_MAX_OVERDUE_MAX", "Largest overdue amount ever on any bureau credit", B)] = g[
        "AMT_CREDIT_MAX_OVERDUE"
    ].max()
    f[
        fd.add(
            "BUR_ACTIVE_REMAINING_MONTHS_MEAN", "Mean remaining months on active bureau credits", B
        )
    ] = ga["DAYS_CREDIT_ENDDATE"].mean() / DAYS_PER_MONTH

    # bureau_balance: monthly DPD buckets per bureau credit, rolled up to the customer.
    bb = read(
        paths["bureau_balance"],
        ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
        ids=b["SK_ID_BUREAU"].unique(),
        id_col="SK_ID_BUREAU",
    )
    bb["bucket"] = dpd_buckets(bb.pop("STATUS"))
    bb = bb.merge(b[["SK_ID_BUREAU", ID]], on="SK_ID_BUREAU", how="inner")
    m = bb["MONTHS_BALANCE"]
    r12, r24 = bb[m >= -12], bb[m >= -24]
    gb = bb.groupby(ID)
    f[
        fd.add(
            "BUR_BB_WORST_BUCKET_EVER", "Worst monthly DPD bucket ever (0 none ... 5 120+ DPD)", BB
        )
    ] = gb["bucket"].max()
    f[fd.add("BUR_BB_WORST_BUCKET_12M", "Worst monthly DPD bucket in the last 12 months", BB)] = (
        r12.groupby(ID)["bucket"].max()
    )
    f[fd.add("BUR_BB_DPD_MONTHS_12M", "Months with any DPD in the last 12 months", BB)] = (
        (r12["bucket"] >= 1).groupby(r12[ID]).sum()
    )
    known24 = r24["bucket"].notna().groupby(r24[ID]).sum()
    dpd24 = (r24["bucket"] >= 1).groupby(r24[ID]).sum()
    f[
        fd.add("BUR_BB_DPD_SHARE_24M", "Share of known months with DPD in the last 24 months", BB)
    ] = ratio(dpd24.reindex(f.index), known24.reindex(f.index))
    late = bb[bb["bucket"] >= 1]
    f[
        fd.add("BUR_BB_MONTHS_SINCE_DPD", "Months since the most recent DPD month", BB)
    ] = -late.groupby(ID)["MONTHS_BALANCE"].max()
    f[
        fd.add(
            "BUR_BB_30DPD_12M_FLAG",
            "1 if any bureau credit was more than 30 DPD (bucket 2+) in the last 12 months",
            BB,
        )
    ] = (r12.groupby(ID)["bucket"].max() >= 2).astype(float)
    f[
        fd.add(
            "BUR_BB_90DPD_12M_FLAG",
            "1 if any bureau credit was more than 90 DPD (bucket 4+) in the last 12 months",
            BB,
        )
    ] = (r12.groupby(ID)["bucket"].max() >= 4).astype(float)

    has = pd.Series(pd.Index(ids).isin(f.index), index=pd.Index(ids, name=ID))
    return LegResult(finalise(f, fd, ids, counts), fd.frame(), has)
