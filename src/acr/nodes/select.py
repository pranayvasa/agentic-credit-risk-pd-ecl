"""Node 4: feature assembly and selection (train only)."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "select",
    role="Join legs; select on train only by IV, correlation, VIF and train-vs-test stability; log exclusions.",
    reads=["config", "split", "legs"],
    writes=["selection"],
)
def select(view):
    legs = sorted(view["legs"])
    return {"selection": {"note": STUB, "legs_joined": legs, "selected": [], "excluded": []}}
