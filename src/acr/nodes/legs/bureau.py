"""Leg agent: bureau (bureau, bureau_balance)."""

from __future__ import annotations

from acr.audit import node
from acr.core.legs import bureau as builder
from acr.nodes.legs._base import run_leg


@node(
    "leg_bureau",
    role="Credit with other lenders and monthly bureau DPD history. Aggregates bureau, bureau_balance to SK_ID_CURR; leg-level quality checks; writes a feature "
    "table and feature dictionary.",
    reads=["config", "split", "dq"],
    writes=["legs"],
    on_failure="A critical leg quality failure (e.g. duplicate IDs) raises and stops the run.",
)
def leg_bureau(view):
    return {"legs": run_leg(view, "bureau", builder.build)}
