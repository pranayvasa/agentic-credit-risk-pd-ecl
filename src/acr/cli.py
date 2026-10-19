"""Command-line entry point: ``acr run`` and ``acr graph``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from acr.config import REPO_ROOT, load_config


def _run(args: argparse.Namespace) -> int:
    load_dotenv(REPO_ROOT / ".env")
    from acr.graph import build_graph

    cfg = load_config(llm_mode=args.llm)
    if args.gemini_model:
        cfg = cfg.model_copy(
            update={
                "llm": cfg.llm.model_copy(
                    update={"gemini": cfg.llm.gemini.model_copy(update={"use": args.gemini_model})}
                )
            }
        )
    if args.exclude_features:
        feats = cfg.base.features.model_copy(
            update={"exclude_features": [f.strip() for f in args.exclude_features.split(",")]}
        )
        cfg = cfg.model_copy(update={"base": cfg.base.model_copy(update={"features": feats})})
    sample = None
    if args.sample is not None:
        sample = args.sample if args.sample > 0 else cfg.base.sample.default_frac
    init = {"config": cfg.model_dump(mode="json"), "sample_frac": sample, "llm_mode": cfg.llm.mode}
    final = build_graph().invoke(
        init, {"recursion_limit": 60, "max_concurrency": cfg.base.runtime.max_concurrency}
    )

    report = final.get("report", {})
    print(f"\nRun {final['run_id']}  status={report.get('status')}  outputs={final['run_dir']}")
    for g in final.get("gate_log", []):
        print(f"  gate {g['gate']:<14} {g['decision']:<9} {g['reason']}")
    for r in final.get("critic_reviews", []):
        tag = "skipped: " + r["reason"] if r.get("skipped") else r.get("conclusion", "")
        print(f"  critic iter {r['iteration']}: {tag}")
    docs = {k: v for k, v in (final.get("documents") or {}).items() if not k.startswith("_")}
    if docs and "skipped" not in docs:
        print(f"  documents: {len(docs)} written to {Path(final['run_dir']) / 'documents'}")
    elif docs:
        print(f"  documents: skipped ({docs['skipped']})")
    g = final.get("grounding") or {}
    if g.get("status") in ("pass", "flagged"):
        print(
            f"  grounding: {g['status']} ({g['totals']['verified']} verified, "
            f"{g['totals']['unverified']} unverified, {g['totals'].get('misattributed', 0)} misattributed, "
            f"{g['totals'].get('contradictions', 0)} contradictions)"
        )
    t = report.get("llm_totals")
    if t:
        print(
            f"  LLM: {t['calls']} call(s), {t['input_tokens']} in / {t['output_tokens']} out tokens, "
            f"est. cost ${t['cost_usd']:.4f} ({cfg.llm.mode})"
        )
    if final.get("halted"):
        print(f"  HALTED: {final.get('halt_reason')}")
        return 2
    return 0


def _graph(args: argparse.Namespace) -> int:
    from acr.graph import mermaid

    text = mermaid()
    out = REPO_ROOT / "docs" / "graph.mmd"
    out.write_text(text)
    print(text if args.print else f"wrote {out.relative_to(REPO_ROOT)}")
    return 0


def _estimate(args: argparse.Namespace) -> int:
    """Pre-flight Gemini cost estimate from a dry-run's actual prompts. Makes no API calls."""
    import json

    from acr.config import resolve_path

    cfg = load_config()
    runs = resolve_path(cfg.base.paths.outputs_dir)
    if args.run:
        src = runs / args.run
    else:
        done = sorted(p for p in runs.glob("*_full_*") if (p / "run_summary.json").exists()
                      and json.loads((p / "run_summary.json").read_text()).get("llm_calls"))  # fmt: skip
        if not done:
            print("no completed full dry-run with LLM calls found; run `acr run` first")
            return 1
        src = done[-1]
    calls = json.loads((src / "run_summary.json").read_text())["llm_calls"]
    est = cfg.llm.estimate
    tin = sum(c["input_tokens"] for c in calls)
    tout = sum(
        max(c["output_tokens"], est.output_tokens_per_call.get(c["node"], 500)) for c in calls
    )
    print(
        f"Estimate from {src.name}: {len(calls)} LLM calls, ~{tin:,} input and ~{tout:,} output tokens"
    )
    for label, model in (
        ("test", cfg.llm.gemini.test_model),
        ("final", cfg.llm.gemini.final_model),
    ):
        r = cfg.llm.cost_per_million[model]
        cost = (tin * r.input + tout * r.output) / 1e6
        lo, hi = cost * (1 - est.uncertainty), cost * (1 + est.uncertainty)
        print(f"  {label:<5} {model:<24} ${cost:.4f}  (range ${lo:.4f} to ${hi:.4f})")
    print(
        "No API calls were made. Rates: config/llm.yaml (check Google's pricing page before use)."
    )
    return 0


def _tune(args: argparse.Namespace) -> int:
    from acr.config import resolve_path
    from acr.tuning import load_tuning, run

    cfg = load_config()
    tcfg = load_tuning()
    if args.folds:
        tcfg = tcfg.model_copy(update={"folds": args.folds})
    runs_dir = resolve_path(cfg.base.paths.outputs_dir)
    if args.run:
        src = runs_dir / args.run
    else:
        done = sorted(p for p in runs_dir.glob("*_full_*")
                      if (p / "artefacts" / "selection" / "selection.json").exists())  # fmt: skip
        if not done:
            print("no completed full run found; run `acr run` first or pass --run")
            return 1
        src = done[-1]
    print(f"tuning on train partition of {src.name}")
    out = run(src, cfg, tcfg, args.only, lambda *a: print(*a, flush=True))
    print(f"report: {out / 'tuning_report.md'}")
    return 0


def _nodes(_: argparse.Namespace) -> int:
    import acr.graph  # noqa: F401  (imports register every node)
    from acr.audit import NODE_REGISTRY

    print(json.dumps({k: v.__dict__ for k, v in NODE_REGISTRY.items()}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="acr", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument(
        "--sample",
        nargs="?",
        type=float,
        const=0.0,
        default=None,
        help="stratified customer sample; bare flag uses config default (10%%)",
    )
    r.add_argument(
        "--llm",
        choices=["dry-run", "gemini", "ollama"],
        default=None,
        help="LLM mode (default from config/llm.yaml: dry-run)",
    )
    r.add_argument(
        "--gemini-model",
        choices=["test", "final"],
        default=None,
        help="which configured Gemini model to use",
    )
    r.add_argument(
        "--exclude-features",
        default=None,
        help="comma-separated features to drop from both models (sensitivity runs)",
    )
    r.set_defaults(func=_run)

    gr = sub.add_parser("graph", help="export the compiled graph as Mermaid to docs/graph.mmd")
    gr.add_argument("--print", action="store_true")
    gr.set_defaults(func=_graph)

    t = sub.add_parser(
        "tune", help="offline train-only CV tuning; writes a report, changes nothing"
    )
    t.add_argument("--run", default=None, help="source run id (default: latest full run)")
    t.add_argument("--only", choices=["both", "scorecard", "lightgbm"], default="both")
    t.add_argument("--folds", type=int, default=None)
    t.set_defaults(func=_tune)

    e = sub.add_parser(
        "estimate-cost", help="estimate Gemini cost from a dry-run's prompts (no API calls)"
    )
    e.add_argument("--run", default=None, help="source run id (default: latest full run)")
    e.set_defaults(func=_estimate)

    n = sub.add_parser("nodes", help="print every node's declared contract as JSON")
    n.set_defaults(func=_nodes)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
