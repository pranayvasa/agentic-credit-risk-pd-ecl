"""Leg agent: previous (previous_application)."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "leg_previous",
    role="Aggregate previous_application to SK_ID_CURR; leg-level DQ; write feature table and feature dictionary.",
    reads=["config", "split"],
    writes=["legs"],
)
def leg_previous(view):
    return {"legs": {"previous": {"note": STUB, "n_features": 0}}}
