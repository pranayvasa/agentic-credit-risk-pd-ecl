"""Node 10: deterministic grounding check (anti-hallucination control).

Every number in every generated document must exist in the run's facts, the critic's
deterministic extra analyses or the config, within the rounding shown. Numbers the critic
or documentation LLM wrote themselves are deliberately *not* part of the reference, so an
invented figure in either is flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

from acr.audit import node
from acr.core.facts import build_facts, numeric_leaves
from acr.core.grounding import check


@node(
    "grounding_check",
    role="Extract every number from the generated documents and verify it against the run's "
    "facts, extra analyses and config within rounding; flag anything unverified.",
    reads=[
        "config",
        "config_hash",
        "sample_frac",
        "dq",
        "split",
        "selection",
        "model",
        "explain",
        "validation",
        "ecl",
        "documents",
        "extra_analyses",
    ],
    writes=["grounding"],
)
def grounding_check(view):
    docs = {k: v for k, v in (view.get("documents") or {}).items() if not k.startswith("_")}
    if not docs or "skipped" in docs:
        return {"grounding": {"status": "skipped", "reason": "no documents generated"}}
    reference = [x for _, x in numeric_leaves(build_facts(view))]
    reference += [x for _, x in numeric_leaves(view.get("extra_analyses", []))]
    reference += [x for _, x in numeric_leaves(view["config"])]
    per_doc, unverified = {}, []
    for name, path in docs.items():
        rep = check(Path(path).read_text(), reference)
        per_doc[name] = rep.counts
        unverified += [
            {"document": name, "number": t.text, "context": t.context} for t in rep.unverified
        ]
    status = "pass" if not unverified else "flagged"
    totals = {
        k: sum(d[k] for d in per_doc.values()) for k in ("verified", "allowlisted", "unverified")
    }
    lines = ["# Grounding check", "", f"Status: **{status}**", "",
             "| Document | Verified | Allowlisted | Unverified |", "|---|---|---|---|"]  # fmt: skip
    lines += [
        f"| {n} | {c['verified']} | {c['allowlisted']} | {c['unverified']} |"
        for n, c in per_doc.items()
    ]
    if unverified:
        lines += [
            "",
            "## Unverified numbers",
            "",
            "| Document | Number | Context |",
            "|---|---|---|",
        ]
        lines += [f"| {u['document']} | {u['number']} | {u['context']} |" for u in unverified]
    out = Path(view["run_dir"]) / "documents"
    (out / "grounding_report.md").write_text("\n".join(lines) + "\n")
    (out / "grounding_report.json").write_text(json.dumps({"status": status, "per_document": per_doc,
                                                           "unverified": unverified}, indent=2))  # fmt: skip
    return {"grounding": {"status": status, "totals": totals, "per_document": per_doc,
                          "unverified": unverified[:50], "report": str(out / "grounding_report.md")}}  # fmt: skip
