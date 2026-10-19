"""Node 9: LLM documentation agent.

Writes four documents section by section. For every section the agent gets the section's
guidance and the run's facts; it writes prose only. Tables are rendered by code from the
same facts. In dry-run mode the prose comes from the spec's templates, filled from facts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from acr.audit import node
from acr.core.docs import TABLES, fill, load_spec
from acr.core.facts import assert_summary_only, build_facts
from acr.llm import get_provider, load_prompt
from acr.llm.cost import call_cost
from acr.nodes._util import cfg
from acr.state import LLMCall

DOCS = ("model_development", "independent_validation", "ongoing_monitoring", "ecl_methodology")


def _to_pdf(md_path: Path) -> str | None:
    try:
        import markdown
        from weasyprint import HTML
    except ImportError:
        return None
    html = markdown.markdown(md_path.read_text(), extensions=["tables"])
    pdf = md_path.with_suffix(".pdf")
    HTML(
        string=f"<html><body style='font-family:sans-serif;font-size:10pt'>{html}</body></html>"
    ).write_pdf(pdf)
    return str(pdf)


@node(
    "document",
    role="Draft the Model Development Document, Independent Validation Report, Ongoing "
    "Monitoring Plan and ECL Methodology Note section by section: LLM prose from curated "
    "facts, tables rendered by code.",
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
        "critic_reviews",
        "extra_analyses",
    ],
    writes=["documents", "llm_calls"],
    uses_llm=True,
    on_failure="If no LLM is available the node records a skip and the pipeline continues.",
)
def document(view):
    c = cfg(view)
    provider, why = get_provider(c.llm)
    if provider is None:
        return {"documents": {"skipped": why}, "llm_calls": []}
    facts = build_facts(view)
    assert_summary_only(facts)
    ctx = {"critic_reviews": view.get("critic_reviews", []),
           "extra_analyses": view.get("extra_analyses", [])}  # fmt: skip
    spec = load_spec()
    prompt_tpl = load_prompt("doc_section", c.llm.prompts.get("doc_section", "v1"))
    reviews = [r for r in view.get("critic_reviews", []) if not r.get("skipped")]
    critic_summary = json.dumps({
        "reviews": len(reviews),
        "challenges": [ch.get("title") for r in reviews for ch in r.get("challenges", [])],
        "analyses_requested": len(view.get("extra_analyses", [])),
        "analyses": [e["kind"] for e in view.get("extra_analyses", [])],
    })  # fmt: skip
    facts_json = json.dumps(facts, separators=(",", ":"), default=str)
    out_dir = Path(view["run_dir"]) / "documents"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    calls, docs, pdfs = [], {}, {}

    for name in DOCS:
        d = spec["documents"][name]
        if name == "ecl_methodology" and "ecl" not in facts:
            continue
        parts = [f"# {d['title']}", "",
                 f"> Run `{view['run_id']}` · generated {stamp} · LLM mode: {provider.name} · "
                 "portfolio project on public Home Credit data, not a production model. "
                 "Every number in this document is checked by the grounding check "
                 "(`grounding_report.md`).", ""]  # fmt: skip
        for i, sec in enumerate(d["sections"], 1):
            prompt = (prompt_tpl.replace("{{DOCUMENT}}", d["title"]).replace("{{PURPOSE}}", d["purpose"])
                      .replace("{{SECTION}}", sec["title"]).replace("{{GUIDANCE}}", sec["guidance"])
                      .replace("{{CRITIC}}", critic_summary).replace("{{FACTS}}", facts_json))  # fmt: skip
            resp = provider.generate(prompt, task="doc_section",
                                     context={"dry_run_text": fill(sec["template"], facts)})  # fmt: skip
            calls.append(LLMCall(node="document", provider=resp.provider, model=resp.model,
                                 input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
                                 cost_usd=call_cost(c.llm, resp.model, resp.provider,
                                                    resp.input_tokens, resp.output_tokens)).model_dump())  # fmt: skip
            parts += [f"## {i}. {sec['title']}", "", resp.text.strip(), ""]
            for t in sec.get("tables", []):
                parts += [TABLES[t](facts, ctx), ""]
        path = out_dir / f"{name}.md"
        path.write_text("\n".join(parts))
        docs[name] = str(path)
        pdf = _to_pdf(path)
        if pdf:
            pdfs[name] = pdf
    (out_dir / "facts.json").write_text(json.dumps(facts, indent=2, default=str))
    docs_state = {**docs, "_facts": str(out_dir / "facts.json")}
    if pdfs:
        docs_state["_pdf"] = json.dumps(pdfs)
    return {"documents": docs_state, "llm_calls": calls}
