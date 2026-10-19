"""Deterministic validation findings: rules over validation metrics -> structured findings.

Each rule states its threshold (from config/gates.yaml) in the finding text, so a reader can
see why a severity was assigned. The LLM critic (Phase 6) can add challenges on top of these,
but cannot change or remove them.
"""

from __future__ import annotations

import pandas as pd

from acr.config import GateSet, ValidationCfg
from acr.state import Finding


def build_findings(m: dict, grades: pd.DataFrame, csi: pd.DataFrame, seg: pd.DataFrame,
                   v: ValidationCfg, gate: GateSet, is_sample: bool) -> list[Finding]:  # fmt: skip
    out: list[Finding] = []

    def add(area, severity, title, detail, remediation=""):
        out.append(Finding(id=f"V{len(out) + 1:02d}", area=area, severity=severity, title=title,
                           detail=detail, remediation=remediation))  # fmt: skip

    if not m["integrity"]["ok"]:
        add(
            "Integrity",
            "red",
            "Model artefacts changed after modelling",
            f"Checksum mismatch: {m['integrity']['mismatches']}. Validation results cannot be relied on.",
            "Re-run modelling; investigate any process that writes to model artefacts.",
        )

    ch = m["champion"]["performance"]
    for part, floor in (("test", gate.gini_min_test), ("oot", gate.gini_min_oot)):
        g = ch[part]["gini"]
        if g < floor:
            add(
                "Performance",
                "red",
                f"Champion {part} Gini below gate",
                f"{part} Gini {g:.4f} (AUC {ch[part]['auc']:.4f}) is below the minimum {floor:.2f}.",
                "Revisit feature engineering and selection; do not deploy.",
            )
        elif g < floor + v.gini_amber_margin:
            add(
                "Performance",
                "medium",
                f"Champion {part} Gini close to gate",
                f"{part} Gini {g:.4f} (AUC {ch[part]['auc']:.4f}) is within {v.gini_amber_margin:.2f} "
                f"of the {floor:.2f} minimum.",
                "Monitor closely; set an early-warning trigger in the monitoring plan.",
            )

    roles = m.get("roles", {"champion": "champion", "challenger": "challenger"})
    names = {"lightgbm": "monotone LightGBM", "scorecard": "WoE scorecard"}
    ch_name = names.get(roles["champion"], "champion")
    cl_name = names.get(roles["challenger"], "challenger")
    cl = m["challenger"]["performance"]
    for role, name, p in (("Champion", ch_name, ch), ("Challenger", cl_name, cl)):
        gap = p["train"]["gini"] - p["oot"]["gini"]
        if gap > v.overfit_gap_medium:
            add(
                "Performance" if role == "Champion" else "Benchmarking",
                "medium",
                f"{role} ({name}) overfits the training data",
                f"Train Gini {p['train']['gini']:.4f} (AUC {p['train']['auc']:.4f}) vs OOT Gini "
                f"{p['oot']['gini']:.4f} (AUC {p['oot']['auc']:.4f}); gap {gap:.4f}, threshold "
                f"{v.overfit_gap_medium:.2f}.",
                "Simplify the model or strengthen regularisation; compare on holdout only.",
            )

    cmp_ = m["comparison"]  # diff = challenger Gini - champion Gini on OOT
    detail = (f"Champion ({ch_name}) OOT Gini {ch['oot']['gini']:.4f} (AUC {ch['oot']['auc']:.4f}) vs "
              f"challenger ({cl_name}) {cl['oot']['gini']:.4f} (AUC {cl['oot']['auc']:.4f}); "
              f"challenger minus champion {cmp_['diff']:.4f} (95% bootstrap CI {cmp_['ci_low']:.4f} "
              f"to {cmp_['ci_high']:.4f}; AUC difference {cmp_['diff'] / 2:.4f}).")  # fmt: skip
    if cmp_["diff"] > v.challenger_gap_medium:
        add("Benchmarking", "medium", "Challenger materially outperforms champion",
            detail + f" The gap exceeds {v.challenger_gap_medium:.2f}.",
            "Document why the champion is preferred, or review the champion choice.")  # fmt: skip
    elif cmp_["ci_low"] > 0:
        add("Benchmarking", "low", "Challenger outperforms champion, within tolerance",
            detail + f" Significant but below {v.challenger_gap_medium:.2f}.",
            "Keep under review.")  # fmt: skip
    elif cmp_["ci_high"] < 0:
        add("Benchmarking", "info", "Champion outperforms the benchmark",
            detail + " The champion's added complexity is supported by a significant gain in "
            "discrimination over the benchmark.",
            "Keep the benchmark as a documented fallback and monitor the gap.")  # fmt: skip

    sp = m["stability"]["score_psi"]
    worst = max(sp.values())
    if worst >= gate.psi_red:
        add(
            "Stability",
            "red",
            "Score distribution unstable",
            f"Score PSI {worst:.4f} at or above the red threshold {gate.psi_red:.2f}.",
            "Investigate population shift before use.",
        )
    elif worst >= gate.psi_amber:
        add(
            "Stability",
            "medium",
            "Score distribution shift",
            f"Score PSI {worst:.4f} at or above the amber threshold {gate.psi_amber:.2f}.",
            "Investigate drivers of the shift.",
        )

    def band(x):
        return "red" if x >= gate.psi_red else "amber" if x >= gate.psi_amber else "green"

    steps = ", ".join(
        f"{k.split('_x')[1]}x: {x:.4f} ({band(x)})" for k, x in m["stability"]["stress_psi"].items()
    )
    add(
        "Stability",
        "medium",
        "Stability testing is not temporal (pseudo-OOT)",
        f"Holdouts are random stratified draws with no application dates, so score PSI is "
        f"near zero by construction (max {worst:.4f}). Stress ladder, up-weighting applicants "
        f"under 30 in the OOT book: {steps}. Score PSI only flags large mix shifts, so it is a "
        "coarse drift detector. Real temporal stability is untested.",
        "Re-validate on a true out-of-time sample; monitor per-feature CSI alongside score PSI.",
    )

    ext = m["stability"].get("external")
    if ext:
        x = ext["score_psi"]
        sev, label = (
            ("red", "unstable")
            if x >= gate.psi_red
            else (("medium", "shifted") if x >= gate.psi_amber else ("info", "stable"))
        )
        add(
            "Stability",
            sev,
            f"External population (application_test) score distribution {label}",
            f"Score PSI of {ext['n']:,} unlabelled application_test applicants against train: "
            f"{x:.4f} (amber {gate.psi_amber:.2f}, red {gate.psi_red:.2f}); challenger PD PSI "
            f"{ext['challenger_pd_psi']:.4f}. Mean champion PD {ext['mean_pd_champion']:.4f} vs "
            f"{ext['mean_pd_champion_train']:.4f} on train. No outcomes exist for this "
            "population, so only distributional stability can be checked; it was drawn from the "
            "same period, so it is not a temporal test either.",
            "Investigate drivers if amber or red." if sev != "info" else "",
        )  # fmt: skip

    csi_cols = [c for c in csi.columns if c.startswith("csi_")]
    for _, r in csi.iterrows():
        w = max(r[c] for c in csi_cols)
        if w >= v.csi_red:
            add(
                "Stability",
                "high",
                f"Unstable characteristic {r['feature']}",
                f"CSI {w:.4f} at or above {v.csi_red:.2f}.",
                "Re-bin or replace the feature.",
            )
        elif w >= v.csi_amber:
            add(
                "Stability",
                "medium",
                f"Characteristic shift {r['feature']}",
                f"CSI {w:.4f} at or above {v.csi_amber:.2f}.",
                "Monitor this feature.",
            )

    reds = grades[grades["traffic_light"] == "red"]
    ambers = grades[grades["traffic_light"] == "amber"]
    if len(reds):
        sev = "high" if len(reds) >= 3 else "medium"
        lst = ", ".join(f"grade {int(r.grade)} (DR {r.default_rate:.4f} vs PD {r.mean_pd:.4f})"
                        for r in reds.itertuples())  # fmt: skip
        add(
            "Calibration",
            sev,
            "PD under-predicts default in some grades (OOT)",
            f"One-sided binomial test rejects at {v.binomial_alpha_red} in {len(reds)} grade(s): "
            f"{lst}. The test assumes independent defaults.",
            "Consider grade-level recalibration or a margin of conservatism.",
        )
    if len(ambers):
        add(
            "Calibration",
            "low",
            "Amber calibration in some grades (OOT)",
            f"{len(ambers)} grade(s) at {v.binomial_alpha_amber} significance: "
            f"{', '.join(str(int(g)) for g in ambers['grade'])}.",
            "Monitor.",
        )
    cal = m["champion"]["calibration"]["oot"]
    if cal["hl_p_value"] < 0.05:
        add(
            "Calibration",
            "low",
            "Hosmer-Lemeshow rejects on OOT",
            f"HL statistic {cal['hl_stat']:.2f}, p-value {cal['hl_p_value']:.4f}. With "
            f"{ch['oot']['n']:,} observations HL is highly powered; decile ECE is "
            f"{cal['ece']:.4f} and the calibration slope {cal['slope']:.3f}.",
            "Rely on ECE, slope and grade-level tests for materiality.",
        )

    if m["champion"]["rank_breaks_grade_oot"] > 0:
        add(
            "Discrimination",
            "high",
            "Rank-ordering breaks across grades (OOT)",
            f"{m['champion']['rank_breaks_grade_oot']} reversal(s) in default rate across grades.",
            "Review master scale boundaries.",
        )
    if m["champion"]["rank_breaks_decile_oot"] > 0:
        add(
            "Discrimination",
            "medium",
            "Rank-ordering breaks across PD deciles (OOT)",
            f"{m['champion']['rank_breaks_decile_oot']} reversal(s) across deciles.",
            "Investigate.",
        )

    thin = grades[grades["share"] < v.grade_share_min]
    if len(thin):
        add(
            "Master scale",
            "low",
            "Sparsely populated grades",
            f"Grade(s) {', '.join(str(int(g)) for g in thin['grade'])} hold under "
            f"{v.grade_share_min:.0%} of the OOT book ({', '.join(str(int(n)) for n in thin['n'])} accounts).",
            "Consider merging grades or recalibrating boundaries.",
        )
    fat = grades[grades["share"] > v.grade_share_max]
    if len(fat):
        add(
            "Master scale",
            "medium",
            "Concentrated grades",
            f"Grade(s) {', '.join(str(int(g)) for g in fat['grade'])} hold over {v.grade_share_max:.0%}.",
            "Split the grade to improve granularity.",
        )

    judged = seg[seg["n"] >= v.segment_min_n]
    weak = judged[judged["gini"] < v.segment_gini_min]
    for r in weak.itertuples():
        add(
            "Segments",
            "medium",
            f"Weak discrimination in {r.segment} = {r.value}",
            f"OOT Gini {r.gini:.4f} (AUC {r.auc:.4f}; threshold Gini {v.segment_gini_min:.2f}, "
            f"n = {r.n:,}).",
            "Consider a segment-specific model or overlay.",
        )
    miscal = judged[(judged["pd_to_dr"] - 1).abs() > v.calibration_ratio_tol]
    for r in miscal.itertuples():
        sev = "medium" if r.segment == "gender" else "low"
        add(
            "Segments" if r.segment != "gender" else "Fairness",
            sev,
            f"Segment calibration off in {r.segment} = {r.value}",
            f"Mean PD / default rate {r.pd_to_dr:.3f} (tolerance ±{v.calibration_ratio_tol:.0%}, "
            f"n = {r.n:,}).",
            "Protected attributes are excluded from the model; investigate proxies."
            if r.segment == "gender"
            else "Review segment calibration.",
        )

    share = m["external_score_share"]
    measure = (
        "of mean absolute SHAP"
        if roles["champion"] == "lightgbm"
        else "of the scorecard's points range"
    )
    if share > 0:
        add(
            "Conceptual soundness",
            "medium" if share > v.external_score_share_max else "low",
            "Champion uses undisclosed external scores",
            f"EXT_SOURCE features carry {share:.0%} {measure} in the champion (threshold "
            f"{v.external_score_share_max:.0%}); their provider and construction are not disclosed, "
            "so they are third-party model inputs.",
            "Obtain documentation of the external scores and monitor their availability. Without "
            "them performance falls materially (docs/sensitivity_ext_source.md).",
        )  # fmt: skip

    rc = m.get("recalibration")
    if rc and rc.get("applied"):
        b, a = rc["before"], rc["after"]
        add("Calibration", "info", "Grade-level recalibration applied to the champion",
            f"A monotone log-odds recalibration with {rc['anchors']} grade anchors, fitted on train "
            f"out-of-fold PDs, was applied. On OOT: decile ECE {b['ece']:.4f} before vs {a['ece']:.4f} "
            f"after; calibration slope {b['slope']:.3f} vs {a['slope']:.3f}; Hosmer-Lemeshow p-value "
            f"{b['hl_p_value']:.4f} vs {a['hl_p_value']:.4f}; red grades {b['red_grades']} vs "
            f"{a['red_grades']}. Gini is unchanged ({b['gini']:.4f} vs {a['gini']:.4f}) because the map "
            "preserves rank order.",
            "Refit the recalibration at each model refresh; monitor grade-level binomial tests.")  # fmt: skip

    ex = m.get("explain")
    if ex:
        mono = ex["monotone"]
        if mono["enabled"] and mono["violating_rows"] > 0:
            add(
                "Explainability",
                "high",
                "Monotonic constraints violated",
                f"{mono['violating_rows']} sweep rows moved PD against the allowed direction across "
                f"{mono['features_tested']} constrained features.",
                "Investigate the constraint set-up before use.",
            )
        elif mono["enabled"]:
            add(
                "Explainability",
                "info",
                "Monotonic constraints verified",
                f"Sweeping each of {mono['features_tested']} constrained features across its range for "
                f"{mono['rows_swept']} OOT applicants moved PD only in the allowed direction.",
            )
        if ex["additivity_max_error"] > 1e-6:
            add(
                "Explainability",
                "high",
                "SHAP values do not reconcile to model scores",
                f"Maximum gap {ex['additivity_max_error']:.2e} in log-odds.",
                "Recompute explanations.",
            )
        st = ex["stability"]
        lo_overlap = min(x["top_overlap"] for x in st.values())
        lo_rho = min(x["spearman"] for x in st.values())
        if any(x["spearman"] != x["spearman"] for x in st.values()):  # NaN: nothing to explain
            add(
                "Explainability",
                "medium",
                "LightGBM explanations could not be assessed",
                "Mean absolute SHAP is zero in at least one sample: the LightGBM model made no splits, "
                "usually because min_child_samples exceeds the data available.",
                "Review the LightGBM parameters for this data size.",
            )
        elif lo_overlap < v.shap_top_overlap_min or lo_rho < v.shap_spearman_min:
            add(
                "Explainability",
                "medium",
                "LightGBM explanations are unstable",
                f"Lowest top-{st['test_vs_oot']['top_n']} driver overlap {lo_overlap:.0%} (minimum "
                f"{v.shap_top_overlap_min:.0%}); lowest rank correlation {lo_rho:.3f} (minimum "
                f"{v.shap_spearman_min}).",
                "Do not rely on reason codes until stable.",
            )
        else:
            add(
                "Explainability",
                "info",
                "LightGBM explanations are stable",
                f"Top-{st['test_vs_oot']['top_n']} SHAP drivers overlap at least {lo_overlap:.0%} across OOT "
                f"halves, test vs OOT and train vs OOT; rank correlation at least {lo_rho:.3f}.",
            )
        if (
            roles["champion"] != "lightgbm"
            and ex["external_score_share"] > v.external_score_share_max
        ):
            add("Explainability", "medium", "LightGBM relies heavily on external scores",
                f"EXT_SOURCE features carry {ex['external_score_share']:.0%} of mean absolute SHAP "
                f"(threshold {v.external_score_share_max:.0%}).", "Document the scores.")  # fmt: skip
        for g, r in ex["gender_pd_to_dr"].items():
            if abs(r - 1) > v.calibration_ratio_tol:
                add(
                    "Fairness",
                    "medium",
                    f"LightGBM calibration off for gender = {g}",
                    f"Mean PD / default rate {r:.3f} (tolerance ±{v.calibration_ratio_tol:.0%}).",
                    "Investigate proxies; gender is not a model input.",
                )

    add(
        "Conceptual soundness",
        "medium",
        "Target is not the regulatory default definition",
        "TARGET marks payment difficulties on early instalments, not 90+ DPD within 12 months "
        "(the NPA definition). PDs are therefore 12-month-equivalent proxies for ECL use.",
        "Rebuild the target on a 90+ DPD, 12-month window when performance data allow.",
    )

    if is_sample:
        add(
            "Scope",
            "info",
            "Sample run",
            "Metrics come from a stratified sample; gates are warn-only.",
        )
    return out


def gate_decision(findings: list[Finding], gate: GateSet) -> tuple[str, str]:
    reds = [f for f in findings if f.severity == "red"]
    if not reds:
        return "continue", "no red findings"
    reason = "; ".join(f"{f.id} {f.title}" for f in reds)
    return ("halt", reason) if gate.enforce else ("warn", f"red findings (not enforced): {reason}")
