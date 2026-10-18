"""Leg agent: application (application_train)."""

from __future__ import annotations

from acr.audit import node
from acr.core.legs import application as builder
from acr.nodes.legs._base import run_leg


@node(
    "leg_application",
    role="Applicant and loan attributes at application. Aggregates application_train to SK_ID_CURR; leg-level quality checks; writes a feature "
    "table and feature dictionary.",
    reads=["config", "split", "dq"],
    writes=["legs"],
    on_failure="A critical leg quality failure (e.g. duplicate IDs) raises and stops the run.",
)
def leg_application(view):
    return {"legs": run_leg(view, "application", builder.build)}
