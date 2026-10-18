"""Behavioural leg: repayment behaviour on the customer's earlier Home Credit loans.

Sources: installments_payments, POS_CASH_balance, credit_card_balance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from acr.core.io import read
from acr.core.legs.common import FeatureDict, LegResult, finalise, ratio
from acr.core.schema import ID

INS, POS, CC = "installments_payments", "POS_CASH_balance", "credit_card_balance"
INST_KEY = ["SK_ID_PREV", "NUM_INSTALMENT_VERSION", "NUM_INSTALMENT_NUMBER"]


def instalment_level(ins: pd.DataFrame) -> pd.DataFrame:
    """Collapse payment rows to one row per scheduled instalment.

    An instalment can be paid in several part-payments. Days past due is measured from the
    due date to the *last* payment, so a partly paid instalment counts as late until it is
    fully settled. Instalments with no payment record have unknown DPD (NaN).
    """
    agg = ins.groupby(INST_KEY, observed=True, sort=False).agg(
        cust=(ID, "first"), due=("DAYS_INSTALMENT", "min"), amt=("AMT_INSTALMENT", "max"),
        paid=("AMT_PAYMENT", "sum"), last_pay=("DAYS_ENTRY_PAYMENT", "max"),
    )  # fmt: skip
    agg["dpd"] = (agg["last_pay"] - agg["due"]).clip(lower=0)
    agg["early"] = (agg["due"] - agg["last_pay"]).clip(lower=0)
    agg["paid_ratio"] = ratio(agg["paid"], agg["amt"])
    agg["shortfall"] = (agg["amt"] - agg["paid"]).clip(lower=0)
    return agg.reset_index(drop=True).rename(columns={"cust": ID})


def build(paths: dict[str, str], ids: np.ndarray) -> LegResult:
    fd = FeatureDict("behavioural")
    idx = pd.Index(ids, name=ID)
    f = pd.DataFrame(index=idx)
    counts: list[str] = []

    # --- installments ---
    ins = read(paths["installments"], [*INST_KEY, ID, "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT",
                                       "AMT_INSTALMENT", "AMT_PAYMENT"], ids)  # fmt: skip
    il = instalment_level(ins)
    del ins
    g = il.groupby(ID)
    recent = il[il["due"] >= -365]
    gr = recent.groupby(ID)
    counts.append(fd.add("BEH_INS_COUNT", "Scheduled instalments on prior Home Credit loans", INS))
    f["BEH_INS_COUNT"] = g.size()
    f[fd.add("BEH_INS_LATE_SHARE", "Share of instalments paid after the due date", INS)] = (
        (il["dpd"] > 0).groupby(il[ID]).mean()
    )
    f[fd.add("BEH_INS_DPD_MAX", "Maximum days past due on any instalment", INS)] = g["dpd"].max()
    f[fd.add("BEH_INS_DPD_MEAN", "Mean days past due per instalment", INS)] = g["dpd"].mean()
    f[
        fd.add("BEH_INS_DPD_MAX_12M", "Maximum DPD on instalments due in the last 12 months", INS)
    ] = gr["dpd"].max()
    counts.append(
        fd.add(
            "BEH_INS_30DPD_COUNT_12M",
            "Instalments more than 30 DPD, due in the last 12 months",
            INS,
        )
    )
    f["BEH_INS_30DPD_COUNT_12M"] = (recent["dpd"] > 30).groupby(recent[ID]).sum()
    f[
        fd.add(
            "BEH_INS_90DPD_12M_FLAG",
            "1 if any instalment due in the last 12 months was more than 90 DPD",
            INS,
        )
    ] = (gr["dpd"].max() > 90).astype(float)
    f[fd.add("BEH_INS_PAID_RATIO_MEAN", "Mean of paid / scheduled amount per instalment", INS)] = g[
        "paid_ratio"
    ].mean()
    f[
        fd.add(
            "BEH_INS_UNDERPAID_SHARE",
            "Share of instalments paid below 99% of the scheduled amount",
            INS,
        )
    ] = (il["paid_ratio"] < 0.99).groupby(il[ID]).mean()
    f[fd.add("BEH_INS_SHORTFALL_SUM", "Total unpaid amount across instalments", INS)] = g[
        "shortfall"
    ].sum()
    f[fd.add("BEH_INS_EARLY_DAYS_MEAN", "Mean days paid before the due date", INS)] = g[
        "early"
    ].mean()
    has_ins = idx.isin(g.size().index)
    del il, recent

    # --- POS / cash loans, monthly ---
    pos = read(paths["pos_cash"], ["SK_ID_PREV", ID, "MONTHS_BALANCE", "CNT_INSTALMENT_FUTURE",
                                   "NAME_CONTRACT_STATUS", "SK_DPD_DEF"], ids)  # fmt: skip
    gp = pos.groupby(ID)
    p12 = pos[pos["MONTHS_BALANCE"] >= -12]
    counts.append(fd.add("BEH_POS_CONTRACT_COUNT", "Number of prior POS / cash loans", POS))
    f["BEH_POS_CONTRACT_COUNT"] = gp["SK_ID_PREV"].nunique()
    f[fd.add("BEH_POS_DPD_MAX", "Maximum DPD (with tolerance) in any month", POS)] = gp[
        "SK_DPD_DEF"
    ].max()
    f[fd.add("BEH_POS_DPD_MAX_12M", "Maximum DPD (with tolerance) in the last 12 months", POS)] = (
        p12.groupby(ID)["SK_DPD_DEF"].max()
    )
    f[fd.add("BEH_POS_DPD_MONTH_SHARE", "Share of months with DPD > 0", POS)] = (
        (pos["SK_DPD_DEF"] > 0).groupby(pos[ID]).mean()
    )
    status = pos["NAME_CONTRACT_STATUS"].astype("string")
    completed = pos.loc[status == "Completed"].groupby(ID)["SK_ID_PREV"].nunique()
    f[
        fd.add("BEH_POS_COMPLETED_SHARE", "Share of POS / cash loans that reached Completed", POS)
    ] = ratio(completed.reindex(idx), f["BEH_POS_CONTRACT_COUNT"])
    latest = pos.sort_values("MONTHS_BALANCE").groupby("SK_ID_PREV").tail(1)
    f[
        fd.add(
            "BEH_POS_REMAINING_INSTALMENTS",
            "Remaining instalments across POS / cash loans, latest month",
            POS,
        )
    ] = latest.groupby(ID)["CNT_INSTALMENT_FUTURE"].sum()
    has_pos = idx.isin(gp.size().index)
    del pos, p12, latest

    # --- credit cards, monthly ---
    cc = read(paths["credit_card"], ["SK_ID_PREV", ID, "MONTHS_BALANCE", "AMT_BALANCE",
                                     "AMT_CREDIT_LIMIT_ACTUAL", "AMT_DRAWINGS_ATM_CURRENT",
                                     "AMT_PAYMENT_CURRENT", "AMT_INST_MIN_REGULARITY",
                                     "SK_DPD_DEF"], ids)  # fmt: skip
    cc["util"] = ratio(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"])
    cc["pay_to_min"] = ratio(cc["AMT_PAYMENT_CURRENT"], cc["AMT_INST_MIN_REGULARITY"]).clip(
        upper=10
    )
    gc = cc.groupby(ID)
    c3 = cc[cc["MONTHS_BALANCE"] >= -3]
    counts.append(fd.add("BEH_CC_COUNT", "Number of prior Home Credit credit cards", CC))
    f["BEH_CC_COUNT"] = gc["SK_ID_PREV"].nunique()
    f[fd.add("BEH_CC_UTIL_MEAN", "Mean monthly balance / limit", CC)] = gc["util"].mean()
    f[fd.add("BEH_CC_UTIL_MAX", "Maximum monthly balance / limit", CC)] = gc["util"].max()
    f[fd.add("BEH_CC_UTIL_MEAN_3M", "Mean balance / limit over the last 3 months", CC)] = (
        c3.groupby(ID)["util"].mean()
    )
    f[fd.add("BEH_CC_DPD_MAX", "Maximum card DPD (with tolerance)", CC)] = gc["SK_DPD_DEF"].max()
    f[fd.add("BEH_CC_DPD_MONTH_SHARE", "Share of card months with DPD > 0", CC)] = (
        (cc["SK_DPD_DEF"] > 0).groupby(cc[ID]).mean()
    )
    f[fd.add("BEH_CC_ATM_DRAWINGS_MEAN", "Mean monthly ATM drawings", CC)] = gc[
        "AMT_DRAWINGS_ATM_CURRENT"
    ].mean()
    f[fd.add("BEH_CC_PAY_TO_MIN_MEAN", "Mean payment / minimum due (capped at 10)", CC)] = gc[
        "pay_to_min"
    ].mean()
    has_cc = idx.isin(gc.size().index)

    has = pd.Series(has_ins | has_pos | has_cc, index=idx)
    return LegResult(finalise(f, fd, ids, counts), fd.frame(), has)
