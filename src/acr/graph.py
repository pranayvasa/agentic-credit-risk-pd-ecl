"""LangGraph wiring: nodes, gates (conditional edges), leg fan-out/join and the critic loop."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from acr.audit import write_event
from acr.config import AppConfig
from acr.nodes.critic import critic
from acr.nodes.document import document
from acr.nodes.ecl import ecl
from acr.nodes.extras import run_extra_analyses
from acr.nodes.grounding import grounding_check
from acr.nodes.halt import halt
from acr.nodes.ingest_dq import ingest_dq
from acr.nodes.legs import LEG_NODES
from acr.nodes.model import model
from acr.nodes.report import assemble_report
from acr.nodes.select import select
from acr.nodes.setup import setup
from acr.nodes.split import split
from acr.nodes.validate import validate
from acr.state import PipelineState


def _log_route(state: PipelineState, gate: str, decision: str, detail: str = "") -> str:
    write_event(
        state["run_dir"], {"event": "route", "gate": gate, "to": decision, "detail": detail}
    )
    return decision


def route_after_dq(state: PipelineState) -> str:
    status = state.get("dq", {}).get("status")
    return _log_route(state, "data_quality", "halt" if status == "halt" else "split", str(status))


def route_after_validation(state: PipelineState) -> str:
    status = state.get("validation", {}).get("status")
    return _log_route(state, "validation", "halt" if status == "halt" else "ecl", str(status))


def route_after_critic(state: PipelineState) -> str:
    """Bounded critic loop.

    ``critic_iteration`` counts completed critic reviews. Extra analyses run only while the
    number of completed request rounds is below ``max_critic_iterations``; after that, any
    new requests are dropped and logged, and the run moves on to documentation.
    """
    max_rounds = AppConfig.model_validate(state["config"]).llm.max_critic_iterations
    requests = state.get("critic_requests") or []
    rounds_done = state.get("critic_iteration", 0) - 1  # extra-analysis rounds already run
    if requests and rounds_done < max_rounds:
        return _log_route(state, "critic_loop", "run_extra_analyses", f"{len(requests)} request(s)")
    detail = "no requests" if not requests else f"bound reached; {len(requests)} request(s) dropped"
    return _log_route(state, "critic_loop", "document", detail)


def build_graph():
    g = StateGraph(PipelineState)

    g.add_node("setup", setup)
    g.add_node("ingest_dq", ingest_dq)
    g.add_node("split", split)
    for name, fn in LEG_NODES.items():
        g.add_node(name, fn)
    g.add_node("select", select)
    g.add_node("model", model)
    g.add_node("validate", validate)
    g.add_node("ecl", ecl)
    g.add_node("critic", critic)
    g.add_node("run_extra_analyses", run_extra_analyses)
    g.add_node("document", document)
    g.add_node("grounding_check", grounding_check)
    g.add_node("assemble_report", assemble_report)
    g.add_node("halt", halt)

    g.add_edge(START, "setup")
    g.add_edge("setup", "ingest_dq")
    g.add_conditional_edges("ingest_dq", route_after_dq, {"split": "split", "halt": "halt"})

    # Fan-out: split -> four legs in parallel. Join: select waits for all four.
    for name in LEG_NODES:
        g.add_edge("split", name)
    g.add_edge(list(LEG_NODES), "select")

    g.add_edge("select", "model")
    g.add_edge("model", "validate")
    g.add_conditional_edges("validate", route_after_validation, {"ecl": "ecl", "halt": "halt"})
    g.add_edge("ecl", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"run_extra_analyses": "run_extra_analyses", "document": "document"},
    )
    g.add_edge("run_extra_analyses", "critic")
    g.add_edge("document", "grounding_check")
    g.add_edge("grounding_check", "assemble_report")
    g.add_edge("assemble_report", END)
    g.add_edge("halt", END)

    return g.compile()


def mermaid() -> str:
    return build_graph().get_graph().draw_mermaid()
