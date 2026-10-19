"""Data-quality checks. Each check reports a severity that applies when it fails.

Severities: critical (halts the run), high, medium, low, info.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from acr.config import DQCfg
from acr.core.io import column_null_counts, read
from acr.core.schema import (
    BUREAU_STATUS_DOMAIN,
    DAYS_SENTINEL,
    ID,
    OPTIONAL_TABLES,
    REQUIRED_COLUMNS,
    TARGET,
)

SEVERITY_ORDER = ["info", "low", "medium", "high", "critical"]


@dataclass
class Check:
    name: str
    table: str
    severity: str
    passed: bool
    detail: str
    value: Any = None
    extra: dict = field(default_factory=dict)


def run_checks(tables: dict[str, dict], cfg: DQCfg) -> list[Check]:
    checks: list[Check] = []

    for name, info in tables.items():
        present = info.get("path") is not None
        sev = "medium" if name in OPTIONAL_TABLES else "critical"
        checks.append(Check("table_present", name, sev, present,
                            "loaded" if present else f"missing file {info.get('missing')}"))  # fmt: skip
    present = {n: i["path"] for n, i in tables.items() if i.get("path")}

    for name, path in present.items():
        cols = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in cols]
        checks.append(Check("required_columns", name, "critical", not missing,
                            f"missing: {missing}" if missing else "all required columns present",
                            value=len(missing)))  # fmt: skip
    if any(c.name == "required_columns" and not c.passed for c in checks):
        return checks  # downstream checks would fail on missing columns

    if "application" in present:
        checks += _application_checks(present["application"], cfg)
        app_ids = set(read(present["application"], [ID])[ID])
    else:
        app_ids = set()

    for name, path in present.items():
        rows = pq.ParquetFile(path).metadata.num_rows
        nulls = column_null_counts(path)
        high = sorted(c for c, n in nulls.items() if rows and n / rows > cfg.high_missing_share)
        checks.append(Check("high_missing_columns", name, "low", not high,
                            f"{len(high)} column(s) over {cfg.high_missing_share:.0%} missing",
                            value=len(high), extra={"columns": high}))  # fmt: skip
        if name == "application":
            continue
        if name == "application_test":
            ext = read(path, [ID])[ID]
            checks.append(Check("id_unique", name, "high", ext.is_unique,
                                f"{int(ext.duplicated().sum())} duplicate {ID}"))  # fmt: skip
            overlap = int(ext.isin(app_ids).sum())
            checks.append(Check("disjoint_from_train", name, "high", overlap == 0,
                                f"{overlap} {ID} also appear in application_train"))  # fmt: skip
            checks.append(Check("unlabelled", name, "info", True,
                                f"{len(ext)} applicants with no TARGET: used for an external "
                                "score-PSI check and a Kaggle submission only"))  # fmt: skip
            continue
        df = read(path, REQUIRED_COLUMNS[name])
        dup = int(df.duplicated().sum())
        share = dup / max(len(df), 1)
        checks.append(Check("duplicate_rows", name, "medium", share <= cfg.duplicate_share_max,
                            f"{dup} duplicate rows ({share:.4%})", value=dup))  # fmt: skip
        if ID in df.columns:
            null_ids = int(df[ID].isna().sum())
            checks.append(Check("key_not_null", name, "high", null_ids == 0,
                                f"{null_ids} null {ID}", value=null_ids))  # fmt: skip
            orphan = 1 - df[ID].isin(app_ids).mean() if app_ids else 0.0
            checks.append(Check("orphan_key_share", name, "info", True,
                                f"{orphan:.1%} of rows belong to customers outside application_train "
                                "(expected: application_test customers)", value=round(orphan, 4)))  # fmt: skip
        if name == "bureau_balance":
            bad = sorted(set(df["STATUS"].astype(str).unique()) - BUREAU_STATUS_DOMAIN)
            checks.append(Check("status_domain", name, "high", not bad,
                                f"unexpected STATUS values: {bad}" if bad else "STATUS within domain"))  # fmt: skip
        del df
    return checks


def _application_checks(path: str, cfg: DQCfg) -> list[Check]:
    app = read(path, [ID, TARGET, "AMT_CREDIT", "AMT_ANNUITY", "DAYS_EMPLOYED"])
    n = len(app)
    out = [
        Check("min_rows", "application", "critical", n >= cfg.min_rows, f"{n} rows", value=n),
        Check("id_unique", "application", "critical", app[ID].is_unique,
              f"{int(app[ID].duplicated().sum())} duplicate {ID}"),
    ]  # fmt: skip
    t = app[TARGET]
    binary = t.notna().all() and set(pd.unique(t)) <= {0, 1}
    out.append(Check("target_binary", "application", "critical", bool(binary),
                     "TARGET is 0/1 with no nulls" if binary else "TARGET has nulls or non-binary values"))  # fmt: skip
    if binary:
        rate = float(t.mean())
        lo, hi = cfg.target_rate_band
        out.append(Check("target_rate", "application", "high", lo <= rate <= hi,
                         f"default rate {rate:.4%} (band {lo:.0%}-{hi:.0%})", value=round(rate, 6)))  # fmt: skip
    bad_amt = int(((app["AMT_CREDIT"] <= 0) | app["AMT_CREDIT"].isna()).sum())
    out.append(Check("credit_amount_positive", "application", "high", bad_amt == 0,
                     f"{bad_amt} non-positive or missing AMT_CREDIT", value=bad_amt))  # fmt: skip
    no_annuity = int(app["AMT_ANNUITY"].isna().sum())
    out.append(Check("annuity_present", "application", "medium", no_annuity == 0,
                     f"{no_annuity} missing AMT_ANNUITY (ECL term and EIR need it)", value=no_annuity))  # fmt: skip
    sentinel = int((app["DAYS_EMPLOYED"] == DAYS_SENTINEL).sum())
    out.append(Check("days_employed_sentinel", "application", "info", True,
                     f"{sentinel} rows use the 365243 sentinel; set to missing with a flag in the application leg",
                     value=sentinel))  # fmt: skip
    return out


def summarise(checks: list[Check], halt_on: list[str]) -> dict[str, Any]:
    failed = [c for c in checks if not c.passed]
    halting = [c for c in failed if c.severity in halt_on]
    by_sev = {s: sum(1 for c in failed if c.severity == s) for s in SEVERITY_ORDER}
    return {
        "status": "halt" if halting else "pass",
        "failed_by_severity": by_sev,
        "halting_checks": [f"{c.table}.{c.name}: {c.detail}" for c in halting],
        "checks": [asdict(c) for c in checks],
    }
