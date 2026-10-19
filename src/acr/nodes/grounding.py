"""Node 10: deterministic grounding check (anti-hallucination control).

Three checks on every generated document:
1. Existence: every number must match a fact, an extra-analysis result or a config value
   within the rounding shown.
2. Attribution: a decimal or percentage in prose must match the fact its label names
   ("challenger PD PSI of 0.015" must equal the challenger PD PSI, not some other 0.015).
3. Consistency: prose must not contradict the validation findings or the run's facts.

Numbers or claims written by the critic or documentation LLM are never part of the reference,
so anything they invent or mislabel is flagged. A flag does not halt the run; it marks the
documents as needing correction before use.
"""

from __future__ import annotations

import json
from pathlib import Path

from acr.audit import node
from acr.core import consistency
from acr.core.facts import build_facts, numeric_leaves
from acr.core.grounding import check_labelled


@node(
    "grounding_check",
    role="Check every generated document: each number must exist in the run's facts (existence), "
    "match the fact its label names (attribution), and no sentence may contradict the validation "
    "findings (consistency). Anything else is flagged.",
    reads=["config", "config_hash", "sample_frac", "dq", "split", "selection", "model", "explain",
           "validation", "ecl", "documents", "extra_analyses", "critic_reviews"],
    writes=["grounding"],
)  # fmt: skip
def grounding_check(view):
    docs = {k: v for k, v in (view.get("documents") or {}).items() if not k.startswith("_")}
    if not docs or "skipped" in docs:
        return {"grounding": {"status": "skipped", "reason": "no documents generated"}}
    facts = build_facts(view)
    labelled = numeric_leaves(facts)
    labelled += [("extra." + p, x) for p, x in numeric_leaves(view.get("extra_analyses", []))]
    labelled += [("config." + p, x) for p, x in numeric_leaves(view["config"])]
    per_doc, issues = {}, []
    for name, path in docs.items():
        text = Path(path).read_text()
        rep = check_labelled(text, labelled)
        contra = consistency.check(text, facts, view.get("extra_analyses", []),
                                   view.get("critic_reviews", []))  # fmt: skip
        per_doc[name] = {**rep.counts, "contradictions": len(contra)}
        issues += [{"document": name, "kind": t.status, "number": t.text, "context": t.context}
                   for t in rep.tokens if t.status in ("unverified", "misattributed")]  # fmt: skip
        issues += [{"document": name, "kind": "contradiction", "number": "", "context": c["sentence"],
                    "rule": c["rule"]} for c in contra]  # fmt: skip
    status = "pass" if not issues else "flagged"
    keys = ("verified", "allowlisted", "unverified", "misattributed", "contradictions")
    totals = {k: sum(d[k] for d in per_doc.values()) for k in keys}
    lines = ["# Grounding check", "", f"Status: **{status}**", "",
             "Existence: every number is in the run's results. Attribution: each number matches the "
             "fact its label names. Consistency: no sentence contradicts a validation finding.", "",
             "| Document | Verified | Allowlisted | Unverified | Misattributed | Contradictions |",
             "|---|---|---|---|---|---|"]  # fmt: skip
    lines += [f"| {n} | {c['verified']} | {c['allowlisted']} | {c['unverified']} | {c['misattributed']} | "
              f"{c['contradictions']} |" for n, c in per_doc.items()]  # fmt: skip
    if issues:
        lines += [
            "",
            "## Issues",
            "",
            "| Document | Kind | Number | Context |",
            "|---|---|---|---|",
        ]
        lines += [f"| {u['document']} | {u['kind']}{' (' + u['rule'] + ')' if u.get('rule') else ''} | "
                  f"{u['number']} | {u['context'].replace('|', '/')} |" for u in issues]  # fmt: skip
    out = Path(view["run_dir"]) / "documents"
    (out / "grounding_report.md").write_text("\n".join(lines) + "\n")
    (out / "grounding_report.json").write_text(json.dumps({"status": status, "per_document": per_doc,
                                                           "issues": issues}, indent=2))  # fmt: skip
    return {"grounding": {"status": status, "totals": totals, "per_document": per_doc,
                          "unverified": issues[:50], "report": str(out / "grounding_report.md")}}  # fmt: skip
