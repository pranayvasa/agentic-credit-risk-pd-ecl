"""Leg agent: behavioural (installments_payments, POS_CASH_balance, credit_card_balance)."""

from __future__ import annotations

from acr.audit import node
from acr.core.legs import behavioural as builder
from acr.nodes.legs._base import run_leg


@node(
    "leg_behavioural",
    role="Repayment behaviour on prior Home Credit loans. Aggregates installments_payments, POS_CASH_balance, credit_card_balance to SK_ID_CURR; leg-level quality checks; writes a feature "
    "table and feature dictionary.",
    reads=["config", "split", "dq"],
    writes=["legs"],
    on_failure="A critical leg quality failure (e.g. duplicate IDs) raises and stops the run.",
)
def leg_behavioural(view):
    return {"legs": run_leg(view, "behavioural", builder.build)}
