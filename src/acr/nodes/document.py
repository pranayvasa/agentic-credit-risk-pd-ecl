"""Node 9: LLM documentation agent."""

from __future__ import annotations

from acr.audit import node
from acr.llm import get_provider
from acr.nodes._util import STUB, cfg

DOCS = ("model_development", "independent_validation", "ongoing_monitoring", "ecl_methodology")


@node(
    "document",
    role="Draft MDD, validation report, monitoring plan and ECL note section by section.",
    reads=[
        "config",
        "dq",
        "split",
        "selection",
        "model",
        "validation",
        "ecl",
        "critic_reviews",
        "extra_analyses",
    ],
    writes=["documents", "llm_calls"],
    uses_llm=True,
    on_failure="If no LLM is available the node records a skip and the pipeline continues.",
)
def document(view):
    provider, why = get_provider(cfg(view).llm)
    if provider is None:
        return {"documents": {"skipped": why}, "llm_calls": []}
    return {"documents": {d: STUB for d in DOCS}, "llm_calls": []}
