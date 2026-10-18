"""Gate routing: red results route to halt; the critic loop is bounded and whitelisted."""

from __future__ import annotations

import acr.graph as graph_mod
from acr.audit import node, read_events
from acr.graph import build_graph
from acr.llm import dryrun


def test_dq_halt_routes_to_halt_node(init_state, monkeypatch):
    @node("ingest_dq", role="test", reads=[], writes=["dq", "gate_log"])
    def failing_dq(view):
        return {
            "dq": {"status": "halt"},
            "gate_log": [{"gate": "data_quality", "decision": "halt", "reason": "duplicate IDs"}],
        }

    monkeypatch.setattr(graph_mod, "ingest_dq", failing_dq)
    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    assert final["halted"] is True
    assert "duplicate IDs" in final["halt_reason"]
    assert "split" not in final
    ended = [e["node"] for e in read_events(final["run_dir"]) if e["event"] == "node_end"]
    assert "split" not in ended and ended[-1] == "halt"


def test_validation_halt_routes_to_halt_node(init_state, monkeypatch):
    @node("validate", role="test", reads=[], writes=["validation", "gate_log"])
    def red_validation(view):
        return {
            "validation": {"status": "halt"},
            "gate_log": [{"gate": "validation", "decision": "halt", "reason": "test Gini 0.30"}],
        }

    monkeypatch.setattr(graph_mod, "validate", red_validation)
    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    assert final["halted"] is True
    assert "ecl" not in final


def test_critic_loop_is_bounded(init_state, monkeypatch):
    # A critic that always asks for more analysis must still stop after the configured bound.
    monkeypatch.setitem(dryrun._CANNED, "critic_iter1", dryrun._critic_first)
    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    max_rounds = final["config"]["llm"]["max_critic_iterations"]
    assert max_rounds == 2
    assert len(final["extra_analyses"]) == max_rounds
    assert len(final["critic_reviews"]) == max_rounds + 1
    routes = [
        e
        for e in read_events(final["run_dir"])
        if e["event"] == "route" and e["gate"] == "critic_loop"
    ]
    assert routes[-1]["to"] == "document" and "bound reached" in routes[-1]["detail"]


def test_critic_cannot_request_off_menu_analysis(init_state, monkeypatch):
    def rogue(_):
        return (
            '{"challenges": [], "conclusion": "x", "requests": ['
            '{"kind": "retune_model", "params": {"C": 10}, "rationale": "better Gini"}]}'
        )

    monkeypatch.setitem(dryrun._CANNED, "critic_iter0", rogue)
    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    assert final["extra_analyses"] == []
    assert final["critic_reviews"][0]["rejected"]
    assert any(e["event"] == "critic_request_rejected" for e in read_events(final["run_dir"]))


def test_llm_nodes_skip_without_provider(init_state, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    final = build_graph().invoke(init_state(llm_mode="gemini"), {"recursion_limit": 60})
    assert final["report"]["status"] == "complete"
    assert final["critic_reviews"][0]["skipped"] is True
    assert "skipped" in final["documents"]
    assert final["llm_calls"] == []
