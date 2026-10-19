"""Facts: the single, curated summary of a run that LLM agents and documents may cite.

* Everything handed to an LLM is built from ``build_facts`` (plus critic extra analyses),
  never from tables of loans. ``assert_summary_only`` enforces that.
* Document tables are rendered from facts by code, so their numbers are exact.
* The grounding check verifies every number in a generated document against the numeric
  leaves of the facts (see ``acr.core.grounding``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

MAX_LIST = 40  # no list in an LLM payload may be longer than this
MAX_PAYLOAD_CHARS = 60_000
FORBIDDEN_KEYS = {"SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"}


DESC = {
    "lightgbm": lambda m: (f"a monotonically constrained LightGBM model on {m['lightgbm']['n_features']} "
                           f"features ({m['lightgbm']['n_trees']} trees)"),
    "scorecard": lambda m: (f"a {m['scorecard']['n_features']}-variable weight-of-evidence logistic "
                            "regression scorecard"),
}  # fmt: skip


def _perf(p: Mapping) -> dict:
    return {
        k: p[k] for k in ("n", "defaults", "default_rate", "mean_pd", "auc", "gini", "ks", "brier")
    }


def build_facts(state: Mapping[str, Any]) -> dict:
    """Curated facts from the run state and config. JSON-safe; summaries only."""
    cfg = state["config"]
    base, gates, e = cfg["base"], cfg["gates"], cfg["ecl"]
    gate = gates["sample"] if state.get("sample_frac") is not None else gates["full"]
    dq, sp, sel, mdl = state["dq"], state["split"], state["selection"], state["model"]
    v = state.get("validation", {})
    vm = v.get("metrics", {})
    oot_frac, test_frac = base["split"]["oot_frac"], base["split"]["test_frac_of_dev"]

    calib = {}
    cpath = mdl.get("paths", {}).get("calibration")
    if cpath:
        calib = json.loads(Path(cpath).read_text())

    f: dict[str, Any] = {
        "run": {
            "run_id": state["run_id"],
            "config_hash": state.get("config_hash"),
            "seed": base["seed"],
            "sample_frac": state.get("sample_frac"),
            "pseudo_oot": True,
        },  # fmt: skip
        "data": {
            "applicants": dq["tables"]["application"]["rows"],
            "default_rate": dq["target_rate"],
            "tables": {k: t["rows"] for k, t in dq["tables"].items() if t.get("rows")},
            "external_applicants": (sp.get("external") or {}).get("n"),
            "dq_failed_by_severity": dq["failed_by_severity"],
        },
        "split": {
            "design_pct": {
                "oot": oot_frac * 100,
                "dev": (1 - oot_frac) * 100,
                "test_of_dev": test_frac * 100,
                "train_of_dev": (1 - test_frac) * 100,
                "train": (1 - oot_frac) * (1 - test_frac) * 100,
                "test": (1 - oot_frac) * test_frac * 100,
            },  # fmt: skip
            "population": sp["population"],
            "counts": sp["counts"],
            "proportions": sp["proportions"],
            "default_rate": sp["target_rate"],
        },
        "features": {
            "total": sel["n_features_total"],
            "candidates": sel["n_candidates"],
            "screened": sel["n_screened"],
            "selected": sel["n_selected"],
            "selected_by_leg": sel["selected_by_leg"],
            "exclusions_by_stage": sel["exclusions_by_stage"],
            "selected_list": [
                {"feature": s["feature"], "leg": s["leg"], "iv": s["iv"]} for s in sel["selected"]
            ],  # fmt: skip
        },
        "roles": {
            "champion": mdl["roles"]["champion"],
            "challenger": mdl["roles"]["challenger"],
            "champion_desc": DESC[mdl["roles"]["champion"]](mdl),
            "challenger_desc": DESC[mdl["roles"]["challenger"]](mdl),
        },
        "champion": {
            "model": mdl["champion"]["model"],
            "type": mdl["champion"]["type"],
            "n_features": mdl["champion"]["n_features"],
            "calibration_decision": mdl["champion"]["calibration"],
        },
        "challenger": {
            "model": mdl["challenger"]["model"],
            "type": mdl["challenger"]["type"],
            "n_features": mdl["challenger"]["n_features"],
            "calibration_decision": mdl["challenger"]["calibration"],
        },
        "scorecard": {
            "n_features": mdl["scorecard"]["n_features"],
            "calibration_oof": {
                k: calib.get("scorecard", {}).get(k)
                for k in ("oof_ece", "oof_slope", "oof_intercept")
            },
            "scaling": mdl["scorecard"]["scaling"],
        },
        "lightgbm": {
            "n_features": mdl["lightgbm"]["n_features"],
            "n_trees": mdl["lightgbm"]["n_trees"],
            "monotone": mdl["lightgbm"]["monotone"],
            "calibration_oof": {
                k: calib.get("lightgbm", {}).get(k)
                for k in ("oof_ece", "oof_slope", "oof_intercept")
            },
            "params": {
                k: base["model"]["lightgbm"][k]
                for k in (
                    "max_depth",
                    "num_leaves",
                    "min_child_samples",
                    "reg_lambda",
                    "learning_rate",
                )
            },
        },
        "origination_proxy": {
            "type": mdl["origination_proxy"]["type"],
            "n_features": len(mdl["origination_proxy"]["features"]),
        },
        "thresholds": {
            "gini_min_test": gate["gini_min_test"],
            "gini_min_oot": gate["gini_min_oot"],
            "psi_amber": gate["psi_amber"],
            "psi_red": gate["psi_red"],
            "gates_enforced": gate["enforce"],
            **{
                k: gates["validation"][k]
                for k in (
                    "csi_amber",
                    "csi_red",
                    "binomial_alpha_red",
                    "binomial_alpha_amber",
                    "overfit_gap_medium",
                    "challenger_gap_medium",
                    "segment_gini_min",
                )
            },
            **{
                k: base["features"][k]
                for k in (
                    "iv_min",
                    "iv_suspicious",
                    "corr_max",
                    "vif_max",
                    "csi_max",
                    "max_features",
                    "p_value_max",
                    "missing_max",
                )
            },
            "max_tree_depth_limit": 10,
            "max_scorecard_features_limit": 15,
        },  # fmt: skip
    }
    if vm:
        for name in ("champion", "challenger"):
            f[name]["performance"] = {p: _perf(x) for p, x in vm[name]["performance"].items()}
            f[name]["calibration_oot"] = vm[name]["calibration"]["oot"]
        f["champion"]["rank_breaks_grade_oot"] = vm["champion"]["rank_breaks_grade_oot"]
        f["gaps"] = {f"{m}_train_minus_oot_gini": vm[m]["performance"]["train"]["gini"]
                     - vm[m]["performance"]["oot"]["gini"] for m in ("champion", "challenger")}  # fmt: skip
        seg_path = v.get("paths", {}).get("segments_oot")
        if seg_path:
            seg = pd.read_csv(seg_path)
            seg = seg[seg["n"] >= gates["validation"]["segment_min_n"]]
            f["segments_oot"] = seg.round(6).to_dict(orient="records")
        f["comparison"] = vm["comparison"] | {
            "auc_diff": vm["comparison"]["diff"] / 2,
            "champion_lead": -vm["comparison"]["diff"],  # champion minus challenger
            "champion_lead_auc": -vm["comparison"]["diff"] / 2,
        }
        st = vm["stability"]
        f["stability"] = {"score_psi": st["score_psi"], "stress_psi": st["stress_psi"],
                          "csi_max": st["csi_max"], "external": st.get("external")}  # fmt: skip
        f["external_score_share"] = vm["external_score_share"]
        g = v.get("grade_table_oot", {})
        f["grades_oot"] = (
            [dict(zip(g, row, strict=True)) for row in zip(*g.values(), strict=True)] if g else []
        )
        f["findings"] = {"by_severity": v["findings_by_severity"],
                         "list": [{k: x[k] for k in ("id", "severity", "area", "title", "detail", "remediation")}
                                  for x in v["findings"]],
                         "gate_decision": v["gate_decision"]}  # fmt: skip
    ex = state.get("explain")
    if ex:
        f["explain"] = {k: ex[k] for k in ("method", "applicants_explained", "additivity_max_error",
                                           "top_features", "leg_shares", "external_score_share",
                                           "stability", "monotone", "reason_codes_per_applicant",
                                           "gender_pd_to_dr", "family_shares",
                                           "primary_reason_top_decile", "families") if k in ex}  # fmt: skip
    ecl = state.get("ecl")
    if ecl:
        f["ecl"] = {
            "illustrative": True, "regulation": ecl["regulation"], "book": ecl["book"],
            "assumptions": ecl["assumptions"], "stage_counts": ecl["stage_counts"],
            "stage_triggers": ecl["stage_triggers"], "by_stage": ecl["by_stage"],
            "by_scenario": ecl["by_scenario"], "lifetime_methods": ecl["lifetime_methods"],
            "lgd_sensitivity": ecl["lgd_sensitivity"], "floor_uplift": ecl["floor_uplift"],
            "checks": ecl["checks"],
            "staging_rules": {k: e["staging"][k] for k in ("sicr_dpd_presumption", "credit_impaired_dpd",
                                                           "lookback_months", "sicr_grade_notches",
                                                           "sicr_pd_ratio", "sicr_abs_pd_min")},
            "revolving": e["revolving"],
        }  # fmt: skip
    return _round(f)


def _round(x, nd: int = 6):
    if isinstance(x, dict):
        return {str(k): _round(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [_round(v, nd) for v in x]
    if isinstance(x, float):
        return round(x, nd)
    return x


def assert_summary_only(payload: Any) -> None:
    """Raise if a payload could carry loan-level data: long lists, ID keys or excessive size."""

    def walk(x, path="payload"):
        if isinstance(x, Mapping):
            bad = FORBIDDEN_KEYS & set(map(str, x))
            if bad:
                raise ValueError(f"{path} contains identifier keys {sorted(bad)}")
            for k, val in x.items():
                walk(val, f"{path}.{k}")
        elif isinstance(x, (list, tuple)):
            if len(x) > MAX_LIST:
                raise ValueError(f"{path} has {len(x)} items (limit {MAX_LIST}); summaries only")
            for i, val in enumerate(x):
                walk(val, f"{path}[{i}]")

    walk(payload)
    size = len(json.dumps(payload, default=str))
    if size > MAX_PAYLOAD_CHARS:
        raise ValueError(f"payload is {size} characters (limit {MAX_PAYLOAD_CHARS})")


def numeric_leaves(x, path: str = "") -> list[tuple[str, float]]:
    """All numeric values in a nested structure, with their dotted paths (bools excluded)."""
    out: list[tuple[str, float]] = []
    if isinstance(x, Mapping):
        for k, v in x.items():
            out += numeric_leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out += numeric_leaves(v, f"{path}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        out.append((path, float(x)))
    return out
