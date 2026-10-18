"""Halt node: reached when a gate returns red. Writes a clear halt report."""

from __future__ import annotations

from pathlib import Path

from acr.audit import node


@node(
    "halt",
    role="Stop the run after a red gate and write halt_report.md explaining why.",
    reads=["gate_log", "dq", "validation"],
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
    path = Path(view["run_dir"]) / "halt_report.md"
    path.write_text("\n".join(lines) + "\n")
    return {
        "halted": True,
        "halt_reason": reason,
        "report": {"status": "halted", "halt_report": str(path)},
    }
