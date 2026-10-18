"""Deterministic executor for critic requests. Runs only whitelisted analyses."""

from __future__ import annotations

from acr.audit import node
from acr.nodes._util import STUB


@node(
    "run_extra_analyses",
    role="Run the critic's whitelisted, deterministic analysis requests; never touches model artefacts.",
    reads=["config", "critic_requests", "critic_iteration", "model", "ecl", "split"],
    writes=["extra_analyses", "critic_requests"],
)
def run_extra_analyses(view):
    done = [
        {
            "iteration": view["critic_iteration"] - 1,
            "kind": r["kind"],
            "params": r["params"],
            "result": {"note": STUB},
        }
        for r in view.get("critic_requests", [])
    ]
    return {"extra_analyses": done, "critic_requests": []}
