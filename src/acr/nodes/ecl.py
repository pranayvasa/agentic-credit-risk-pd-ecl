"""Node 7: illustrative RBI ECL simulation on the pseudo-OOT book."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "ecl",
    role="Stage the OOT book, build lifetime PD, EAD, EIR discounting, scenarios, LGD, RBI floors.",
    reads=["config", "split", "model", "validation"],
    writes=["ecl"],
)
def ecl(view):
    return {"ecl": {"note": STUB, "illustrative": True}}
