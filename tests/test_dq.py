"""Data-quality checks and the DQ gate."""

from __future__ import annotations

from acr.audit import read_events
from acr.config import load_config
from acr.core.dq import run_checks, summarise
from acr.core.io import ensure_parquet
from acr.graph import build_graph
from synthetic import make_tables, write_raw


def _dq(tmp_path, tables):
    raw = write_raw(tmp_path / "raw", tables)
    cfg = load_config().base
    loaded = ensure_parquet(raw, tmp_path / "interim", cfg.data.tables)
    return summarise(run_checks(loaded, cfg.dq), ["critical"])


def _failed(dq):
    return {(c["table"], c["name"]) for c in dq["checks"] if not c["passed"]}


def test_clean_synthetic_data_passes(tmp_path):
    dq = _dq(tmp_path, make_tables(n=1500))
    assert dq["status"] == "pass"
    assert dq["failed_by_severity"]["critical"] == 0


def test_duplicate_application_id_is_critical(tmp_path):
    t = make_tables(n=1500)
    t["application"].loc[1, "SK_ID_CURR"] = t["application"].loc[0, "SK_ID_CURR"]
    dq = _dq(tmp_path, t)
    assert dq["status"] == "halt"
    assert ("application", "id_unique") in _failed(dq)


def test_missing_required_column_is_critical(tmp_path):
    t = make_tables(n=1500)
    t["bureau"] = t["bureau"].drop(columns=["CREDIT_ACTIVE"])
    dq = _dq(tmp_path, t)
    assert dq["status"] == "halt"
    assert ("bureau", "required_columns") in _failed(dq)


def test_non_binary_target_is_critical(tmp_path):
    t = make_tables(n=1500)
    t["application"].loc[0, "TARGET"] = 2
    assert ("application", "target_binary") in _failed(_dq(tmp_path, t))


def test_target_rate_outside_band_flags_but_does_not_halt(tmp_path):
    t = make_tables(n=1500)
    t["application"]["TARGET"] = 0
    t["application"].loc[:600, "TARGET"] = 1  # 40% default rate
    dq = _dq(tmp_path, t)
    assert ("application", "target_rate") in _failed(dq)
    assert dq["status"] == "pass"


def test_unexpected_bureau_status_flagged(tmp_path):
    t = make_tables(n=1500)
    t["bureau_balance"].loc[0, "STATUS"] = "Z"
    assert ("bureau_balance", "status_domain") in _failed(_dq(tmp_path, t))


def test_graph_halts_on_critical_dq(tmp_path, init_state):
    t = make_tables(n=1500)
    t["application"].loc[1, "SK_ID_CURR"] = t["application"].loc[0, "SK_ID_CURR"]
    raw = write_raw(tmp_path / "bad_raw", t)
    final = build_graph().invoke(init_state(raw_dir=raw), {"recursion_limit": 60})
    assert final["halted"] is True
    assert "id_unique" in final["halt_reason"]
    ended = [e["node"] for e in read_events(final["run_dir"]) if e["event"] == "node_end"]
    assert "split" not in ended
