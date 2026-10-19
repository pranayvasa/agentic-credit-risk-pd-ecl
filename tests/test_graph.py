"""Graph wiring: compiles, fan-out/join shape, end-to-end dry-run and Mermaid export."""

from __future__ import annotations

from acr.audit import NODE_REGISTRY, read_events
from acr.graph import build_graph, mermaid
from acr.nodes.legs import LEG_NODES


def test_graph_compiles_with_all_nodes():
    nodes = set(build_graph().get_graph().nodes)
    expected = {
        "setup",
        "ingest_dq",
        "split",
        *LEG_NODES,
        "select",
        "model",
        "explain",
        "validate",
        "ecl",
        "critic",
        "run_extra_analyses",
        "document",
        "grounding_check",
        "assemble_report",
        "halt",
    }
    assert expected <= nodes


def test_legs_fan_out_from_split_and_join_at_select():
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    for leg in LEG_NODES:
        assert ("split", leg) in edges
        assert (leg, "select") in edges


def test_every_node_declares_a_contract():
    build_graph()
    for name, spec in NODE_REGISTRY.items():
        assert spec.role, name
        assert spec.writes, name


def test_only_critic_and_document_use_llm():
    build_graph()
    assert {n for n, s in NODE_REGISTRY.items() if s.uses_llm} == {"critic", "document"}


def test_dry_run_end_to_end(init_state):
    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    assert final["report"]["status"] == "complete"
    assert set(final["legs"]) == {"application", "bureau", "behavioural", "previous"}
    assert final["report"]["llm_totals"]["cost_usd"] == 0.0
    events = read_events(final["run_dir"])
    ended = [e["node"] for e in events if e["event"] == "node_end"]
    assert ended[0] == "setup" and ended[-1] == "assemble_report"
    # select runs only after all four legs have finished (the join)
    sel = ended.index("select")
    assert all(ended.index(leg) < sel for leg in LEG_NODES)


def test_mermaid_export_contains_conditional_edges():
    text = mermaid()
    assert "ingest_dq -.-> halt" in text
    assert "validate -.-> halt" in text
    assert "critic -.-> run_extra_analyses" in text


def test_split_and_leg_artefacts_persisted(init_state):
    from pathlib import Path

    import pandas as pd

    final = build_graph().invoke(init_state(), {"recursion_limit": 60})
    s = final["split"]
    for p in ("train", "test", "oot"):
        assert Path(s["id_paths"][p]).exists()
    assert s["pseudo_oot"] is True
    ids = set(pd.read_parquet(s["assignment_path"])["SK_ID_CURR"])
    ext = set(pd.read_parquet(s["external"]["path"])["SK_ID_CURR"])
    assert not ids & ext
    ids |= ext  # legs build features for unlabelled applicants too, so the model can score them
    for leg, info in final["legs"].items():
        f = pd.read_parquet(info["features_path"])
        d = pd.read_csv(info["dictionary_path"])
        assert set(f.index) == ids, leg
        assert list(d["name"]) == list(f.columns), leg
