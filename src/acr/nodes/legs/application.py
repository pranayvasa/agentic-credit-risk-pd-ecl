"""Leg agent: application (application_train)."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "leg_application",
    role="Aggregate application_train to SK_ID_CURR; leg-level DQ; write feature table and feature dictionary.",
    reads=["config", "split"],
    writes=["legs"],
)
def leg_application(view):
    return {"legs": {"application": {"note": STUB, "n_features": 0}}}
