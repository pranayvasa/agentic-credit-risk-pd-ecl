"""Previous applications leg: the customer's earlier applications to Home Credit."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acr.core.io import read
from acr.core.legs.common import DAYS_PER_MONTH, FeatureDict, LegResult, clean_days, finalise, ratio
from acr.core.schema import ID

SRC = "previous_application"


def build(paths: dict[str, str], ids: np.ndarray) -> LegResult:
    cols = ["SK_ID_PREV", ID, "NAME_CONTRACT_TYPE", "AMT_ANNUITY", "AMT_APPLICATION",
            "AMT_CREDIT", "RATE_DOWN_PAYMENT", "NAME_CONTRACT_STATUS", "DAYS_DECISION",
            "CNT_PAYMENT", "NAME_YIELD_GROUP", "DAYS_TERMINATION"]  # fmt: skip
    p = read(paths["previous"], cols, ids)
    fd = FeatureDict("previous")
    status = p["NAME_CONTRACT_STATUS"].astype("string")
    ctype = p["NAME_CONTRACT_TYPE"].astype("string")
    approved, refused = status == "Approved", status == "Refused"
    recent = p["DAYS_DECISION"] >= -365
    term = p["DAYS_TERMINATION"].astype("float64")
    # 365243 in DAYS_TERMINATION means "no termination date yet"; positive means in the future.
    still_running = approved & (term.isna() | (term == 365243) | (term > 0))
    g = p.groupby(ID)
    by = p[ID]
    f = pd.DataFrame(index=pd.Index(g.size().index, name=ID))
    counts: list[str] = []

    def cnt(name, definition, s):
        counts.append(fd.add(name, definition, SRC))
        f[name] = s

    cnt("PRV_COUNT", "Number of previous applications", g.size())
    cnt("PRV_APPROVED_COUNT", "Approved previous applications", approved.groupby(by).sum())
    cnt("PRV_REFUSED_COUNT", "Refused previous applications", refused.groupby(by).sum())
    cnt(
        "PRV_COUNT_12M",
        "Previous applications decided in the last 12 months",
        recent.groupby(by).sum(),
    )
    cnt("PRV_REFUSED_12M", "Refusals in the last 12 months", (refused & recent).groupby(by).sum())
    cnt(
        "PRV_RUNNING_COUNT",
        "Approved previous loans with no termination yet",
        still_running.groupby(by).sum(),
    )
    cnt(
        "PRV_REVOLVING_COUNT",
        "Previous revolving-loan applications",
        (ctype == "Revolving loans").groupby(by).sum(),
    )
    f[fd.add("PRV_REFUSED_SHARE", "Refused / all previous applications", SRC)] = ratio(
        f["PRV_REFUSED_COUNT"], f["PRV_COUNT"]
    )
    f[fd.add("PRV_CANCELED_SHARE", "Cancelled / all previous applications", SRC)] = ratio(
        (status == "Canceled").groupby(by).sum(), f["PRV_COUNT"]
    )
    dec = clean_days(p["DAYS_DECISION"])
    f[fd.add("PRV_MONTHS_SINCE_LAST", "Months since the latest previous decision", SRC)] = (
        -dec.groupby(by).max() / DAYS_PER_MONTH
    )
    f[fd.add("PRV_MONTHS_SINCE_REFUSAL", "Months since the latest refusal", SRC)] = (
        -dec[refused].groupby(by[refused]).max() / DAYS_PER_MONTH
    )
    a = p[approved]
    ga = a.groupby(ID)
    f[
        fd.add(
            "PRV_CREDIT_TO_APPLICATION_MEAN",
            "Mean granted / requested amount, approved applications",
            SRC,
        )
    ] = ratio(a["AMT_CREDIT"], a["AMT_APPLICATION"]).groupby(a[ID]).mean()
    f[fd.add("PRV_AMT_CREDIT_MEAN", "Mean credit amount, approved applications", SRC)] = ga[
        "AMT_CREDIT"
    ].mean()
    f[fd.add("PRV_ANNUITY_MEAN", "Mean annuity, approved applications", SRC)] = ga[
        "AMT_ANNUITY"
    ].mean()
    f[fd.add("PRV_CNT_PAYMENT_MEAN", "Mean term in instalments, approved applications", SRC)] = ga[
        "CNT_PAYMENT"
    ].mean()
    f[fd.add("PRV_DOWN_PAYMENT_RATE_MEAN", "Mean down-payment rate", SRC)] = g[
        "RATE_DOWN_PAYMENT"
    ].mean()
    f[
        fd.add("PRV_HIGH_YIELD_SHARE", "Share of applications in the high interest-rate group", SRC)
    ] = (p["NAME_YIELD_GROUP"].astype("string") == "high").groupby(by).mean()
    f[fd.add("PRV_CASH_LOAN_SHARE", "Share of applications that were cash loans", SRC)] = (
        (ctype == "Cash loans").groupby(by).mean()
    )

    has = pd.Series(pd.Index(ids).isin(f.index), index=pd.Index(ids, name=ID))
    return LegResult(finalise(f, fd, ids, counts), fd.frame(), has)
