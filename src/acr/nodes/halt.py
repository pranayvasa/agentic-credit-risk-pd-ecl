"""Halt node: reached when a gate returns red. Writes a clear halt report."""

from __future__ import annotations

import json
from pathlib import Path

from acr.audit import node
from acr.nodes.report import SUMMARY_KEYS, _llm_totals


@node(
    "halt",
    role="Stop the run after a red gate; write halt_report.md explaining why, and "
    "run_summary.json with everything computed before the halt.",
    reads=list(SUMMARY_KEYS),
    writes=["halted", "halt_reason", "report"],
)
def halt(view):
    red = [g for g in view.get("gate_log", []) if g["decision"] == "halt"]
    reason = "; ".join(f"{g['gate']}: {g['reason']}" for g in red) or "unspecified"
    lines = [
        "# Run halted",
        "",
        f"Run `{view['run_id']}` stopped at a red gate.",
        "",
        "| Gate | Decision | Reason |",
        "|---|---|---|",
    ]
    lines += [
        f"| {g['gate']} | {g['decision']} | {g['reason']} |" for g in view.get("gate_log", [])
    ]
    summary = {k: view.get(k) for k in SUMMARY_KEYS}
    summary.update(
        status="halted", halt_reason=reason, llm_totals=_llm_totals(view.get("llm_calls", []))
    )
    (Path(view["run_dir"]) / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    path = Path(view["run_dir"]) / "halt_report.md"
    path.write_text("\n".join(lines) + "\n")
    return {
        "halted": True,
        "halt_reason": reason,
        "report": {"status": "halted", "halt_report": str(path)},
    }
