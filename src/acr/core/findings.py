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

    gap = ch["train"]["gini"] - ch["oot"]["gini"]
    if gap > v.overfit_gap_medium:
        add(
            "Performance",
            "medium",
            "Champion overfitting",
            f"Train Gini {ch['train']['gini']:.4f} (AUC {ch['train']['auc']:.4f}) exceeds OOT Gini "
            f"{ch['oot']['gini']:.4f} (AUC {ch['oot']['auc']:.4f}) by {gap:.4f} "
            f"(threshold {v.overfit_gap_medium:.2f}).",
            "Simplify the scorecard or strengthen selection.",
        )
    cl = m["challenger"]["performance"]
    cl_gap = cl["train"]["gini"] - cl["oot"]["gini"]
    if cl_gap > v.overfit_gap_medium:
        add(
            "Benchmarking",
            "medium",
            "Challenger overfits the training data",
            f"LightGBM train Gini {cl['train']['gini']:.4f} (AUC {cl['train']['auc']:.4f}) vs OOT "
            f"Gini {cl['oot']['gini']:.4f} (AUC {cl['oot']['auc']:.4f}); gap {cl_gap:.4f}, "
            f"threshold {v.overfit_gap_medium:.2f}.",
            "Treat challenger train metrics as optimistic; compare on holdout only.",
        )

    cmp_ = m["comparison"]
    if cmp_["diff"] > v.challenger_gap_medium:
        add(
            "Benchmarking",
            "medium",
            "Challenger materially outperforms champion",
            f"Challenger OOT Gini {cl['oot']['gini']:.4f} (AUC {cl['oot']['auc']:.4f}) vs champion "
            f"{ch['oot']['gini']:.4f} (AUC {ch['oot']['auc']:.4f}): Gini difference "
            f"{cmp_['diff']:.4f} (95% bootstrap CI {cmp_['ci_low']:.4f} to {cmp_['ci_high']:.4f}; "
            f"AUC difference {cmp_['diff'] / 2:.4f}) exceeds {v.challenger_gap_medium:.2f}.",
            "Document why the interpretable scorecard is preferred, or investigate the non-linear "
            "effects the challenger captures (interactions, finer bins).",
        )
    elif cmp_["ci_low"] > 0:
        add(
            "Benchmarking",
            "low",
            "Challenger outperforms champion, within tolerance",
            f"OOT Gini difference {cmp_['diff']:.4f} (95% CI {cmp_['ci_low']:.4f} to "
            f"{cmp_['ci_high']:.4f}; AUC difference {cmp_['diff'] / 2:.4f}) is significant but "
            f"below {v.challenger_gap_medium:.2f}.",
            "Acceptable cost of interpretability; keep under review.",
        )

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
    if share > v.external_score_share_max:
        add(
            "Conceptual soundness",
            "medium",
            "Heavy reliance on undisclosed external scores",
            f"EXT_SOURCE features contribute {share:.0%} of the scorecard's points range "
            f"(threshold {v.external_score_share_max:.0%}); their provider and construction are "
            "not disclosed.",
            "Obtain documentation of the external scores. Without them OOT Gini falls from about 0.50 to 0.40 and the gate fails (docs/sensitivity_ext_source.md).",
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
