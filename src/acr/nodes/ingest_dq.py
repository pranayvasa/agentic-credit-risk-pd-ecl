"""Node 1: ingestion and data quality."""

from __future__ import annotations

from acr.audit import node
from acr.config import resolve_path
from acr.core.dq import run_checks, summarise
from acr.nodes._util import cfg, is_sample


@node(
    "ingest_dq",
    role="Load raw tables into a downcast parquet cache; run schema, key, duplicate, missingness "
    "and target-rate checks.",
    reads=["config", "sample_frac"],
    writes=["dq", "gate_log"],
    on_failure="Any failed critical check sets dq.status='halt'; the gate routes to the halt node.",
)
def ingest_dq(view):
    from acr.core.io import ensure_parquet

    c = cfg(view)
    tables = ensure_parquet(
        resolve_path(c.base.paths.raw_dir),
        resolve_path(c.base.paths.interim_dir),
        c.base.data.tables,
    )
    checks = run_checks(tables, c.base.dq)
    dq = summarise(checks, c.gate_set(is_sample(view)).dq_halt_on)
    dq["tables"] = tables
    rate = next((x["value"] for x in dq["checks"] if x["name"] == "target_rate"), None)
    dq["target_rate"] = rate
    if dq["status"] == "halt":
        gate = {
            "gate": "data_quality",
            "decision": "halt",
            "reason": "; ".join(dq["halting_checks"]),
        }
    else:
        fails = sum(dq["failed_by_severity"].values())
        gate = {"gate": "data_quality", "decision": "continue",
                "reason": f"no critical failures; {fails} non-critical finding(s)"}  # fmt: skip
    return {"dq": dq, "gate_log": [gate]}
