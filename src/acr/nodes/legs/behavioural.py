"""Leg agent: behavioural (installments_payments, POS_CASH_balance, credit_card_balance)."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "leg_behavioural",
    role="Aggregate installments_payments, POS_CASH_balance, credit_card_balance to SK_ID_CURR; leg-level DQ; write feature table and feature dictionary.",
    reads=["config", "split"],
    writes=["legs"],
)
def leg_behavioural(view):
    return {"legs": {"behavioural": {"note": STUB, "n_features": 0}}}
