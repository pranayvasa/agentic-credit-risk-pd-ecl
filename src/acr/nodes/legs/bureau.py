"""Leg agent: bureau (bureau, bureau_balance)."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "leg_bureau",
    role="Aggregate bureau, bureau_balance to SK_ID_CURR; leg-level DQ; write feature table and feature dictionary.",
    reads=["config", "split"],
    writes=["legs"],
)
def leg_bureau(view):
    return {"legs": {"bureau": {"note": STUB, "n_features": 0}}}
