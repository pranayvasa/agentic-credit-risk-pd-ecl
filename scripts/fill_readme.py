"""Regenerate the machine-written parts of README.md and docs/sample_outputs/.

Blocks between <!-- BEGIN:name --> and <!-- END:name --> markers are replaced:
  graph    Mermaid diagram generated from the compiled LangGraph
  nodes    per-node contract table from the @node registry (reads, writes, LLM, failure)
  results  results tables from a completed full run's run_summary.json
  findings validation findings summary from the same run
Then the run's generated documents and grounding report are copied to docs/sample_outputs/.

Usage:  .venv/bin/python scripts/fill_readme.py [--run RUN_ID]   (default: latest full run)
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
ORDER = ["setup", "ingest_dq", "split", "leg_application", "leg_bureau", "leg_behavioural",
         "leg_previous", "select", "model", "explain", "validate", "ecl", "critic", "run_extra_analyses",
         "document", "grounding_check", "assemble_report", "halt"]  # fmt: skip


def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{name} -->)(.*?)(<!-- END:{name} -->)", re.S)
    if not pat.search(text):
        raise SystemExit(f"README has no {name} block")
    return pat.sub(lambda m: f"{m.group(1)}\n{body.strip()}\n{m.group(3)}", text)


def graph_block() -> str:
    from acr.graph import mermaid

    text = mermaid()
    (ROOT / "docs" / "graph.mmd").write_text(text)
    return f"```mermaid\n{text.strip()}\n```"


def nodes_block() -> str:
    import acr.graph  # noqa: F401  (registers nodes)
    from acr.audit import NODE_REGISTRY

    rows = [
        "| # | Node | What it does | Reads | Writes | LLM | On failure |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, name in enumerate(ORDER):
        s = NODE_REGISTRY[name]
        reads = ", ".join(f"`{r}`" for r in s.reads) or "—"
        writes = ", ".join(f"`{w}`" for w in s.writes)
        rows.append(
            f"| {i} | `{name}` | {s.role} | {reads} | {writes} | {'yes' if s.uses_llm else 'no'} | {s.on_failure} |"
        )
    return "\n".join(rows)


def _pct(x, nd=2):
    return f"{x:.{nd}%}"


def results_block(s: dict) -> str:
    v = s["validation"]["metrics"]
    out = [
        f"From run `{s['run_id']}` (full population, dry-run LLM mode, config hash `{s['config_hash']}`).",
        "",
    ]
    out += ["**Discrimination and calibration** (Gini = 2 × AUC − 1)", "",
            "| Model | Partition | Gini | AUC | KS | Mean PD | Default rate |", "|---|---|---|---|---|---|---|"]  # fmt: skip
    for m in ("champion", "challenger"):
        for p in ("train", "test", "oot"):
            x = v[m]["performance"][p]
            out.append(
                f"| {m} ({s['model']['roles'][m]}) | {p} | {x['gini']:.4f} | {x['auc']:.4f} | {x['ks']:.4f} | {_pct(x['mean_pd'])} | {_pct(x['default_rate'])} |"
            )
    c, st = v["comparison"], v["stability"]
    cal = v["champion"]["calibration"]["oot"]
    out += ["", "| Measure | Value |", "|---|---|",
            f"| Champion OOT decile ECE / calibration slope | {cal['ece']:.4f} / {cal['slope']:.3f} |",
            f"| Champion lead over benchmark, OOT Gini (95% bootstrap CI) | {-c['diff']:.4f} ({-c['ci_high']:.4f} to {-c['ci_low']:.4f}) |",
            f"| Score PSI train→test / train→OOT | {st['score_psi']['train_vs_test']:.4f} / {st['score_psi']['train_vs_oot']:.4f} |",
            f"| Largest selected-feature CSI | {st['csi_max']:.4f} |",
            f"| Stress PSI ladder (under-30s ×3 / ×10 / ×50) | {' / '.join(f'{x:.4f}' for x in st['stress_psi'].values())} |"]  # fmt: skip
    ex = s.get("explain")
    if ex:
        out.append(
            f"| LightGBM monotonic constraints: features tested / violations | "
            f"{ex['monotone']['features_tested']} / {ex['monotone']['violating_rows']} |"
        )
        out.append(
            f"| SHAP top-10 driver overlap, test vs OOT | {ex['stability']['test_vs_oot']['top_overlap']:.0%} |"
        )
    if st.get("external"):
        out.append(
            f"| Score PSI, unlabelled application_test ({st['external']['n']:,}) vs train | {st['external']['score_psi']:.4f} |"
        )
    e = s["ecl"]
    out += ["", f"**Illustrative RBI ECL** on the OOT book ({e['book']['accounts']:,} loans; LGD "
            f"{_pct(e['assumptions']['lgd_base'], 0)}; floors {_pct(e['assumptions']['floors']['stage1'], 0)} / "
            f"{_pct(e['assumptions']['floors']['stage2'], 0)} / {_pct(e['assumptions']['floors']['stage3_0_1y'], 0)})", "",
            "| Stage | Accounts | Exposure (m) | Final ECL (m) | Coverage | Realised default rate |",
            "|---|---|---|---|---|---|"]  # fmt: skip
    for k, x in e["by_stage"].items():
        out.append(f"| {k.replace('_', ' ')} | {x['accounts']:,.0f} | {x['ead'] / 1e6:,.1f} | {x['ecl_final'] / 1e6:,.1f} | "
                   f"{_pct(x['coverage_final'])} | {_pct(x['realised_default_rate'])} |")  # fmt: skip
    g, calls = s["grounding"], s["llm_calls"]
    out += ["", "| Control | Result |", "|---|---|",
            f"| Validation gate | {s['validation']['gate_decision']} ({', '.join(f'{n} {k}' for k, n in s['validation']['findings_by_severity'].items())}) |",
            f"| Critic loop | {len(s['critic_reviews'])} reviews, {len(s['extra_analyses'])} whitelisted analyses run |",
            f"| Grounding check | {g['status']}: {g['totals']['verified']} numbers verified, {g['totals']['unverified']} unverified |",
            f"| LLM calls (dry-run) | {len(calls)} calls, ~{sum(x['input_tokens'] for x in calls):,} input tokens |",
            f"| ECL QC checks | {sum(e['checks'].values())} of {len(e['checks'])} pass |"]  # fmt: skip
    return "\n".join(out)


def findings_block(s: dict) -> str:
    rows = ["| ID | Severity | Area | Finding |", "|---|---|---|---|"]
    rows += [
        f"| {f['id']} | {f['severity']} | {f['area']} | {f['title']} |"
        for f in s["validation"]["findings"]
    ]
    return "\n".join(rows)


def copy_samples(run_dir: Path) -> None:
    dst = ROOT / "docs" / "sample_outputs"
    dst.mkdir(exist_ok=True)
    for p in (run_dir / "documents").glob("*.md"):
        shutil.copy(p, dst / p.name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    runs = ROOT / "outputs"
    run_dir = runs / args.run if args.run else sorted(
        p for p in runs.glob("*_full_*") if (p / "documents" / "grounding_report.md").exists())[-1]  # fmt: skip
    s = json.loads((run_dir / "run_summary.json").read_text())
    text = README.read_text()
    for name, body in (("graph", graph_block()), ("nodes", nodes_block()),
                       ("results", results_block(s)), ("findings", findings_block(s))):  # fmt: skip
        text = replace_block(text, name, body)
    README.write_text(text)
    copy_samples(run_dir)
    print(f"README and docs/sample_outputs filled from {run_dir.name}")


if __name__ == "__main__":
    main()
