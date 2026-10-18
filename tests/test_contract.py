"""The node read/write contract is enforced at runtime."""

from __future__ import annotations

import pytest

from acr.audit import ContractViolation, node, read_events


def test_undeclared_write_is_rejected(tmp_path):
    @node("t_write", role="test", reads=[], writes=["dq"])
    def bad(view):
        return {"dq": {}, "model": {"sneaky": True}}

    with pytest.raises(ContractViolation, match="model"):
        bad({"run_id": "r", "run_dir": str(tmp_path)})


def test_undeclared_read_is_invisible(tmp_path):
    @node("t_read", role="test", reads=["split"], writes=["dq"])
    def peeks(view):
        return {"dq": {"saw": view["model"]}}

    with pytest.raises(KeyError):
        peeks({"run_id": "r", "run_dir": str(tmp_path), "split": {}, "model": {"x": 1}})


def test_view_is_read_only(tmp_path):
    @node("t_mutate", role="test", reads=["split"], writes=["dq"])
    def mutates(view):
        view["split"] = {}
        return {}

    with pytest.raises(TypeError):
        mutates({"run_id": "r", "run_dir": str(tmp_path), "split": {"a": 1}})


def test_errors_are_audited(tmp_path):
    @node("t_err", role="test", reads=[], writes=["dq"])
    def boom(view):
        raise ValueError("bad data")

    with pytest.raises(ValueError):
        boom({"run_id": "r", "run_dir": str(tmp_path)})
    events = read_events(tmp_path)
    assert events[-1]["event"] == "node_error" and "bad data" in events[-1]["error"]
