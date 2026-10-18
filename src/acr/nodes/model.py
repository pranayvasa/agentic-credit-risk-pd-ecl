"""Node 5: modelling. Champion WoE scorecard, challenger LightGBM, calibration, master scale."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "model",
    role="Fit champion WoE logistic scorecard and LightGBM challenger on train; calibrate; map to master scale.",
    reads=["config", "split", "selection"],
    writes=["model"],
)
def model(view):
    return {"model": {"note": STUB, "champion": "woe_logit_scorecard", "challenger": "lightgbm"}}
