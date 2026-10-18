"""Node 2: train / test / pseudo-OOT split on SK_ID_CURR."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "split",
    role="Stratified 75:25 (dev:OOT) then 75:25 (train:test) split on customer ID; persist ID lists.",
    reads=["config", "sample_frac", "dq"],
    writes=["split"],
)
def split(view):
    return {"split": {"note": STUB, "proportions": {"train": 0.5625, "test": 0.1875, "oot": 0.25}}}
