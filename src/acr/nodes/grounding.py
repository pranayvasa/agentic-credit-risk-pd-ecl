"""Node 10: deterministic grounding check (anti-hallucination control)."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "grounding_check",
    role="Extract every number from generated documents and verify it against the run's metrics.",
    reads=["config", "documents", "validation", "ecl", "model", "selection", "split", "dq"],
    writes=["grounding"],
)
def grounding_check(view):
    return {"grounding": {"status": "stub", "note": STUB, "unverified": []}}
