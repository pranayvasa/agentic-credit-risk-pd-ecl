"""Leg agent: previous (previous_application)."""

from __future__ import annotations

from acr.audit import node
from acr.core.legs import previous as builder
from acr.nodes.legs._base import run_leg


@node(
    "leg_previous",
    role="Earlier applications to Home Credit. Aggregates previous_application to SK_ID_CURR; leg-level quality checks; writes a feature "
    "table and feature dictionary.",
    reads=["config", "split", "dq"],
    writes=["legs"],
    on_failure="A critical leg quality failure (e.g. duplicate IDs) raises and stops the run.",
)
def leg_previous(view):
    return {"legs": run_leg(view, "previous", builder.build)}
