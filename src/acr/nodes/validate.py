"""Node 6: independent validation (read-only over model artefacts) and the performance gate."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "validate",
    role="Independent second-line checks: Gini/KS/PSI/CSI, calibration, rank-ordering, segments, "
    "champion vs challenger; issue findings; apply gates.",
    reads=["config", "sample_frac", "split", "model", "selection"],
    writes=["validation", "gate_log"],
    on_failure="A red gate (with enforcement on) routes to the halt node with a halt report.",
)
def validate(view):
    return {
        "validation": {"note": STUB, "status": "pass", "findings": [], "metrics": {}},
        "gate_log": [{"gate": "validation", "decision": "continue", "reason": STUB}],
    }
