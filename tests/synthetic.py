"""Small synthetic Home Credit-shaped dataset for tests. No real data is used in the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acr.core.schema import APP_CATEGORICAL, APP_NUMERIC, REQUIRED_COLUMNS

FILES = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "installments": "installments_payments.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "credit_card": "credit_card_balance.csv",
    "previous": "previous_application.csv",
}


def make_tables(n: int = 3000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100_001, 100_001 + n)
    ext = rng.uniform(0, 1, n)
    # Realistic signal strength: IV of the EXT_SOURCE features lands around 0.2-0.4.
    p = 1 / (1 + np.exp(-(-2.5 - 1.6 * (ext - 0.5))))
    target = (rng.uniform(size=n) < p).astype(int)

    app = pd.DataFrame({"SK_ID_CURR": ids, "TARGET": target})
    for c in APP_NUMERIC:
        app[c] = rng.normal(0, 1, n).round(3)
    app["EXT_SOURCE_1"] = np.where(
        rng.uniform(size=n) < 0.5, np.nan, (ext + rng.normal(0, 0.3, n)).clip(0, 1)
    )
    app["EXT_SOURCE_2"] = (ext + rng.normal(0, 0.3, n)).clip(0, 1)
    app["EXT_SOURCE_3"] = (ext + rng.normal(0, 0.4, n)).clip(0, 1)
    app["AMT_INCOME_TOTAL"] = rng.uniform(5e4, 5e5, n).round(1)
    app["AMT_CREDIT"] = rng.uniform(5e4, 1e6, n).round(1)
    app["AMT_ANNUITY"] = (app["AMT_CREDIT"] / rng.integers(12, 60, n) * 1.2).round(1)
    app["AMT_GOODS_PRICE"] = (app["AMT_CREDIT"] * 0.9).round(1)
    app["DAYS_BIRTH"] = -rng.integers(21 * 365, 68 * 365, n)
    app["DAYS_EMPLOYED"] = np.where(rng.uniform(size=n) < 0.15, 365243, -rng.integers(0, 15000, n))
    for c in ("DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"):
        app[c] = -rng.integers(0, 8000, n).astype(float)
    for c in APP_CATEGORICAL:
        app[c] = rng.choice(["A", "B", "C"], n)
    app["CODE_GENDER"] = rng.choice(["F", "M"], n)
    app["NAME_FAMILY_STATUS"] = rng.choice(["Married", "Single / not married"], n)

    # bureau: 0-4 credits per customer
    nb = rng.integers(0, 5, n)
    bcust = np.repeat(ids, nb)
    m = len(bcust)
    bureau = pd.DataFrame({
        "SK_ID_CURR": bcust,
        "SK_ID_BUREAU": np.arange(5_000_000, 5_000_000 + m),
        "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], m),
        "DAYS_CREDIT": -rng.integers(1, 3000, m),
        "CREDIT_DAY_OVERDUE": rng.choice([0, 0, 0, 10, 40], m),
        "DAYS_CREDIT_ENDDATE": rng.integers(-2000, 2000, m).astype(float),
        "AMT_CREDIT_MAX_OVERDUE": rng.uniform(0, 1e4, m),
        "CNT_CREDIT_PROLONG": rng.integers(0, 2, m),
        "AMT_CREDIT_SUM": rng.uniform(1e4, 5e5, m),
        "AMT_CREDIT_SUM_DEBT": rng.uniform(0, 2e5, m),
        "AMT_CREDIT_SUM_OVERDUE": rng.choice([0.0, 0.0, 500.0], m),
        "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Microloan"], m),
    })  # fmt: skip
    months = 15
    bb = pd.DataFrame({
        "SK_ID_BUREAU": np.repeat(bureau["SK_ID_BUREAU"].to_numpy(), months),
        "MONTHS_BALANCE": np.tile(-np.arange(months), m),
        "STATUS": rng.choice(["0", "0", "0", "1", "2", "4", "C", "X"], m * months),
    })  # fmt: skip

    # prior Home Credit loans
    npv = rng.integers(0, 3, n)
    pcust = np.repeat(ids, npv)
    k = len(pcust)
    prev_ids = np.arange(2_000_000, 2_000_000 + k)
    previous = pd.DataFrame({
        "SK_ID_PREV": prev_ids, "SK_ID_CURR": pcust,
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Consumer loans", "Revolving loans"], k),
        "AMT_ANNUITY": rng.uniform(1e3, 3e4, k), "AMT_APPLICATION": rng.uniform(1e4, 5e5, k),
        "AMT_CREDIT": rng.uniform(1e4, 5e5, k), "RATE_DOWN_PAYMENT": rng.uniform(0, 0.3, k),
        "NAME_CONTRACT_STATUS": rng.choice(["Approved", "Refused", "Canceled"], k),
        "DAYS_DECISION": -rng.integers(1, 2500, k), "CNT_PAYMENT": rng.integers(6, 48, k),
        "NAME_YIELD_GROUP": rng.choice(["low_normal", "middle", "high"], k),
        "DAYS_TERMINATION": rng.choice([365243.0, -100.0, -500.0], k),
    })  # fmt: skip
    ninst = 6
    inst = pd.DataFrame({
        "SK_ID_PREV": np.repeat(prev_ids, ninst), "SK_ID_CURR": np.repeat(pcust, ninst),
        "NUM_INSTALMENT_VERSION": 1, "NUM_INSTALMENT_NUMBER": np.tile(np.arange(1, ninst + 1), k),
        "DAYS_INSTALMENT": -np.tile(np.arange(ninst, 0, -1) * 30, k).astype(float),
    })  # fmt: skip
    inst["DAYS_ENTRY_PAYMENT"] = inst["DAYS_INSTALMENT"] + rng.choice(
        [-5, 0, 0, 3, 45, 100], len(inst)
    )
    inst["AMT_INSTALMENT"] = 1000.0
    inst["AMT_PAYMENT"] = rng.choice([1000.0, 1000.0, 600.0], len(inst))
    pos = pd.DataFrame({
        "SK_ID_PREV": np.repeat(prev_ids, 4), "SK_ID_CURR": np.repeat(pcust, 4),
        "MONTHS_BALANCE": np.tile([-4, -3, -2, -1], k), "CNT_INSTALMENT_FUTURE": np.tile([3, 2, 1, 0], k),
        "NAME_CONTRACT_STATUS": np.tile(["Active", "Active", "Active", "Completed"], k),
        "SK_DPD_DEF": rng.choice([0, 0, 0, 5, 40], 4 * k),
    })  # fmt: skip
    cc = pd.DataFrame({
        "SK_ID_PREV": np.repeat(prev_ids, 3), "SK_ID_CURR": np.repeat(pcust, 3),
        "MONTHS_BALANCE": np.tile([-3, -2, -1], k), "AMT_BALANCE": rng.uniform(0, 9e4, 3 * k),
        "AMT_CREDIT_LIMIT_ACTUAL": 9e4, "AMT_DRAWINGS_ATM_CURRENT": rng.uniform(0, 5e3, 3 * k),
        "AMT_PAYMENT_CURRENT": rng.uniform(0, 5e3, 3 * k), "AMT_INST_MIN_REGULARITY": 2000.0,
        "SK_DPD_DEF": rng.choice([0, 0, 10], 3 * k),
    })  # fmt: skip
    tables = {"application": app, "bureau": bureau, "bureau_balance": bb, "installments": inst,
              "pos_cash": pos, "credit_card": cc, "previous": previous}  # fmt: skip
    for name, df in tables.items():
        missing = set(REQUIRED_COLUMNS[name]) - set(df.columns)
        assert not missing, (name, missing)
    return tables


def write_raw(raw_dir: Path, tables: dict[str, pd.DataFrame]) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(raw_dir / FILES[name], index=False)
    return raw_dir
