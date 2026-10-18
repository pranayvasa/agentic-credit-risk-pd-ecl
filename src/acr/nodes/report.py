"""Node 11: report assembly. Writes run summary and metrics to outputs/<run_id>/."""

from __future__ import annotations

import json
from pathlib import Path

from acr.audit import node

SUMMARY_KEYS = (
    "run_id",
    "config_hash",
    "sample_frac",
    "llm_mode",
    "dq",
    "split",
    "legs",
    "selection",
    "model",
    "validation",
    "ecl",
    "critic_reviews",
    "extra_analyses",
    "documents",
    "grounding",
    "gate_log",
    "llm_calls",
)


def _llm_totals(calls: list[dict]) -> dict:
    return {
        "calls": len(calls),
        "input_tokens": sum(c["input_tokens"] for c in calls),
        "output_tokens": sum(c["output_tokens"] for c in calls),
        "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
    }


@node(
    "assemble_report",
    role="Write run_summary.json and print a console summary.",
    reads=list(SUMMARY_KEYS),
    writes=["report"],
)
def assemble_report(view):
    summary = {k: view.get(k) for k in SUMMARY_KEYS}
    summary["llm_totals"] = _llm_totals(view.get("llm_calls", []))
    path = Path(view["run_dir"]) / "run_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    return {
        "report": {
            "status": "complete",
            "summary_path": str(path),
            "llm_totals": summary["llm_totals"],
        }
    }
