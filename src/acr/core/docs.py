"""Documentation engine: spec, template filling and deterministic tables.

Design rule: tables are rendered by code from the run's facts, so their numbers are exact.
Only the narrative prose comes from the LLM (or, in dry-run, from the spec's templates).
The grounding check then verifies every number in the finished document.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import yaml

from acr.llm.base import PROMPTS_DIR

PLACEHOLDER = re.compile(r"\{([A-Za-z0-9_.]+)(?::([^{}]+))?\}")


def load_spec(version: str = "v1") -> dict:
    return yaml.safe_load((PROMPTS_DIR / f"doc_sections_{version}.yaml").read_text())


def resolve(facts: Mapping, path: str) -> Any:
    cur: Any = facts
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit():
            cur = cur[int(part)]
        else:
            raise KeyError(f"fact not found: {path}")
    return cur


def money(x: float, nd: int = 1) -> str:
    return f"{x / 1e6:,.{nd}f} million"


def fmt(value: Any, spec: str | None) -> str:
    if spec and spec.startswith("m"):
        return money(float(value), int(spec[1:] or 1))
    return format(value, spec) if spec else str(value)


def fill(template: str, facts: Mapping) -> str:
    return PLACEHOLDER.sub(lambda m: fmt(resolve(facts, m.group(1)), m.group(2)), template)


# --- tables ------------------------------------------------------------------------------


def _md(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(x, nd=2):
    return f"{x:.{nd}%}"


def data_table(f, ctx):
    return _md(["Table", "Rows"], [[k, f"{v:,}"] for k, v in f["data"]["tables"].items()])


def split_table(f, ctx):
    s = f["split"]
    return _md(["Partition", "Applicants", "Share", "Default rate"],
               [[p, f"{s['counts'][p]:,}", _pct(s["proportions"][p]), _pct(s["default_rate"][p])]
                for p in ("train", "test", "oot")])  # fmt: skip


def selection_funnel_table(f, ctx):
    ex = f["features"]["exclusions_by_stage"]
    order = ["config", "role", "missing", "binning", "iv", "csi", "corr", "vif", "cap", "model"]
    return _md(["Stage", "Features excluded"], [[k, ex[k]] for k in order if k in ex])


def selected_features_table(f, ctx):
    return _md(["Feature", "Leg", "IV"],
               [[s["feature"], s["leg"], f"{s['iv']:.4f}"] for s in f["features"]["selected_list"]])  # fmt: skip


def performance_table(f, ctx):
    rows = []
    for m in ("champion", "challenger"):
        for p in ("train", "test", "oot"):
            x = f[m]["performance"][p]
            rows.append([m, p, f"{x['gini']:.4f}", f"{x['auc']:.4f}", f"{x['ks']:.4f}",
                         _pct(x["mean_pd"]), _pct(x["default_rate"])])  # fmt: skip
    return _md(["Model", "Partition", "Gini", "AUC", "KS", "Mean PD", "Default rate"], rows)


def calibration_table(f, ctx):
    rows = [[m, f"{c['ece']:.4f}", f"{c['slope']:.3f}", f"{c['hl_p_value']:.4f}", f"{c['pd_to_dr']:.3f}"]
            for m in ("champion", "challenger") for c in [f[m]["calibration_oot"]]]  # fmt: skip
    return _md(["Model (OOT)", "Decile ECE", "Slope", "HL p-value", "Mean PD / DR"], rows)


def grade_table(f, ctx):
    return _md(["Grade", "Accounts", "Share", "Mean PD", "Default rate", "Binomial p (under)", "Light"],
               [[g["grade"], f"{g['n']:,}", _pct(g["share"]), _pct(g["mean_pd"]), _pct(g["default_rate"]),
                 f"{g['p_under']:.4f}", g["traffic_light"]] for g in f["grades_oot"]])  # fmt: skip


def stability_table(f, ctx):
    s = f["stability"]
    rows = [["Score PSI, train vs test", f"{s['score_psi']['train_vs_test']:.4f}"],
            ["Score PSI, train vs OOT", f"{s['score_psi']['train_vs_oot']:.4f}"],
            ["Largest feature CSI", f"{s['csi_max']:.4f}"]]  # fmt: skip
    rows += [
        [f"Stress PSI, under-30s x{k.split('_x')[1]}", f"{v:.4f}"]
        for k, v in s["stress_psi"].items()
    ]
    if s.get("external"):
        rows.append(["Score PSI, application_test vs train", f"{s['external']['score_psi']:.4f}"])
    return _md(["Measure", "Value"], rows)


def comparison_table(f, ctx):
    c = f["comparison"]
    return _md(["Measure", "Value"], [["OOT Gini difference (challenger - champion)", f"{c['diff']:.4f}"],
                                      ["AUC difference", f"{c['auc_diff']:.4f}"],
                                      ["95% bootstrap interval", f"{c['ci_low']:.4f} to {c['ci_high']:.4f}"]])  # fmt: skip


def findings_table(f, ctx):
    return _md(["ID", "Severity", "Area", "Finding"],
               [[x["id"], x["severity"], x["area"], x["title"]] for x in f["findings"]["list"]])  # fmt: skip


def findings_detail(f, ctx):
    return "\n\n".join(f"**{x['id']} ({x['severity']}) {x['title']}.** {x['detail']}"
                       + (f" *Remediation:* {x['remediation']}" if x["remediation"] else "")
                       for x in f["findings"]["list"])  # fmt: skip


def critic_table(f, ctx):
    reviews, extras = ctx.get("critic_reviews", []), ctx.get("extra_analyses", [])
    if not reviews or all(r.get("skipped") for r in reviews):
        return "_The critic agent was skipped (no LLM provider available)._"
    rows = [[r["iteration"], c["id"], c["severity"], c["area"], c["title"], c["detail"]]
            for r in reviews for c in r.get("challenges", [])]  # fmt: skip
    out = (
        [_md(["Iteration", "ID", "Severity", "Area", "Challenge", "Detail"], rows)] if rows else []
    )
    if extras:
        out.append(_md(["Iteration", "Analysis requested", "Rationale", "Outcome"],
                       [[e["iteration"], e["kind"], e["rationale"], "error: " + e["error"] if "error" in e else "completed"]
                        for e in extras]))  # fmt: skip
    out += [f"*Review {r['iteration']} conclusion:* {r.get('conclusion', '')}" for r in reviews]
    return "\n\n".join(out)


def monitoring_table(f, ctx):
    t = f["thresholds"]
    return _md(["Metric", "Amber", "Red", "Frequency"], [
        ["Score PSI (vs development)", t["psi_amber"], t["psi_red"], "monthly"],
        ["Feature CSI (each selected feature)", t["csi_amber"], t["csi_red"], "monthly"],
        ["Gini (minimum)", f"{t['gini_min_oot']} + margin", t["gini_min_oot"], "quarterly"],
        ["Grade binomial test p-value (under-prediction)", t["binomial_alpha_amber"], t["binomial_alpha_red"], "quarterly"],
        ["Champion vs challenger Gini gap", t["challenger_gap_medium"], "n/a", "quarterly"],
    ])  # fmt: skip


def staging_table(f, ctx):
    e = f["ecl"]
    trig = e["stage_triggers"]
    rows = [[f"Stage {s}", f"{e['stage_counts'][s]:,}", _pct(e["by_stage"][f"stage_{s}"]["realised_default_rate"])]
            for s in ("1", "2", "3")]  # fmt: skip
    t2 = _md(["Trigger", "Accounts"], [[k, f"{v:,}"] for k, v in trig.items()])
    return _md(["Stage", "Accounts", "Realised default rate"], rows) + "\n\n" + t2


def lifetime_table(f, ctx):
    lm = f["ecl"]["lifetime_methods"]
    return _md(["Method", "Weighted ECL", "Stage 2 ECL", "Stage 2 mean lifetime PD"],
               [[m, money(v["ecl_weighted"]), money(v["stage2_ecl"]), _pct(v["stage2_mean_lifetime_pd"])]
                for m, v in lm.items()])  # fmt: skip


def lgd_table(f, ctx):
    return _md(["LGD", "Model ECL", "Final ECL", "Coverage"],
               [[_pct(r["lgd"], 0), money(r["ecl_model"]), money(r["ecl_final"]), _pct(r["coverage_final"])]
                for r in f["ecl"]["lgd_sensitivity"]])  # fmt: skip


def scenario_table(f, ctx):
    sc = f["ecl"]["by_scenario"]["total"]
    sp = f["ecl"]["assumptions"]["scenarios"]
    rows = [[k, f"{v['weight']:.2f}", f"{v['pd_scalar']:.2f}"] for k, v in sp.items()]
    t1 = _md(["Scenario", "Weight", "PD scalar"], rows)
    t2 = _md(["Scenario column", "Total ECL"], [[k, money(v)] for k, v in sc.items()])
    return t1 + "\n\n" + t2


def ecl_stage_table(f, ctx):
    rows = []
    for k, v in f["ecl"]["by_stage"].items():
        rows.append([k, f"{v['accounts']:,.0f}", money(v["ead"]), money(v["ecl_model"]), money(v["ecl_final"]),
                     _pct(v["coverage_final"]), _pct(v["floor_binding_share"], 1)])  # fmt: skip
    return _md(
        ["Stage", "Accounts", "Exposure", "Model ECL", "Final ECL", "Coverage", "Floor binds"], rows
    )


def ecl_assumptions_table(f, ctx):
    a = f["ecl"]["assumptions"]
    rows = [["LGD (unsecured backstop, para 97)", _pct(a["lgd_base"], 0)],
            ["12-month PD floor (para 96)", _pct(a["pd_floor_12m"])],
            ["Floor Stage 1 / 2 / 3 (para 82(3)(x))",
             f"{_pct(a['floors']['stage1'], 0)} / {_pct(a['floors']['stage2'], 0)} / {_pct(a['floors']['stage3_0_1y'], 0)}"],
            ["Median term (months)", f"{a['term_months']['median']:.0f}"],
            ["Lifetime method (primary)", a["lifetime_primary"]]]  # fmt: skip
    rows += [[f"Implied monthly rate, {k}", _pct(v)] for k, v in a["implied_monthly_rates"].items()]
    return _md(["Assumption", "Value"], rows)


def shap_table(f, ctx):
    e = f["explain"]
    t1 = _md(
        ["Feature", "Leg", "Share of mean abs SHAP", "Constraint"],
        [[x["feature"], x["leg"], _pct(x["share"], 1), x["constraint"]] for x in e["top_features"]],
    )
    t2 = _md(
        ["Leg", "Share of mean abs SHAP"], [[k, _pct(v, 1)] for k, v in e["leg_shares"].items()]
    )
    return t1 + "\n\n" + t2


def explain_checks_table(f, ctx):
    e = f["explain"]
    st, mono = e["stability"], e["monotone"]
    rows = [["Applicants explained (OOT)", f"{e['applicants_explained']:,}"],
            ["SHAP additivity, max error (log-odds)", f"{e['additivity_max_error']:.1e}"],
            ["Monotonic constraints: features tested / violating rows",
             f"{mono['features_tested']} / {mono['violating_rows']}"]]  # fmt: skip
    rows += [[f"Top-{x['top_n']} driver overlap / rank correlation, {k.replace('_', ' ')}",
              f"{x['top_overlap']:.0%} / {x['spearman']:.3f}"] for k, x in st.items()]  # fmt: skip
    rows += [
        [f"Challenger mean PD / default rate, gender {g}", f"{r:.3f}"]
        for g, r in e["gender_pd_to_dr"].items()
    ]
    return _md(["Check", "Result"], rows)


TABLES = {fn.__name__: fn for fn in (
    data_table, split_table, selection_funnel_table, selected_features_table, performance_table,
    calibration_table, grade_table, stability_table, comparison_table, findings_table,
    findings_detail, critic_table, monitoring_table, staging_table, lifetime_table, lgd_table,
    scenario_table, ecl_stage_table, ecl_assumptions_table, shap_table, explain_checks_table)}  # fmt: skip
