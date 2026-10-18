"""Node 1: ingestion and data quality."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "ingest_dq",
    role="Load raw tables, run schema, missingness, duplicate and target-rate checks.",
    reads=["config", "sample_frac"],
    writes=["dq", "gate_log"],
    on_failure="Any critical check sets dq.status='halt'; the gate routes to the halt node.",
)
def ingest_dq(view):
    return {
        "dq": {"status": "pass", "note": STUB, "checks": []},
        "gate_log": [{"gate": "data_quality", "decision": "continue", "reason": STUB}],
    }
