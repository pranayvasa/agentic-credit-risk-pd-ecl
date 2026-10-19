"""Dry-run provider: deterministic responses, zero cost, no network.

It stands in for the LLM so the whole graph runs in development and tests. Responses are
built from the same structured payload the real model receives (``context``), so every
number a dry-run response contains comes from the run's facts. Token counts are estimated
from text length so cost reports show roughly what a real run would consume.
"""

from __future__ import annotations

import json

from acr.llm.base import LLMResponse
from acr.llm.cost import estimate_tokens


class DryRunProvider:
    name = "dry-run"
    model = "dry-run"

    def generate(self, prompt: str, *, task: str, context: dict | None = None) -> LLMResponse:
        if task.startswith("critic_iter"):
            text = critic_response(int(task.removeprefix("critic_iter")), context or {})
        elif task.startswith("doc_section") and context and "dry_run_text" in context:
            text = context["dry_run_text"]
        else:
            text = _CANNED.get(task, _default)(prompt)
        return LLMResponse(
            text, estimate_tokens(prompt), estimate_tokens(text), self.model, self.name
        )


def _f(x, nd=4):
    return f"{x:.{nd}f}"


def critic_response(iteration: int, payload: dict) -> str:
    """A skeptical-validator review derived from the facts. Iteration 0 raises challenges and
    requests analyses; iteration 1 reviews the results; later iterations add nothing."""
    f = payload.get("facts", {})
    if iteration >= 2 or not f:
        return json.dumps({"challenges": [], "requests": [],
                           "conclusion": "No further analysis requested."})  # fmt: skip
    if iteration == 1:
        return json.dumps(_review(payload.get("extra_analyses", [])))
    th, ch, cl = f["thresholds"], f["champion"], f["challenger"]
    out, req = [], []
    cmp_ = f.get("comparison", {})
    if cmp_ and cmp_["diff"] > th["challenger_gap_medium"]:
        out.append({"id": "C1", "severity": "medium", "area": "Benchmarking",
                    "title": "Interpretability premium is large",
                    "detail": f"The challenger's OOT Gini is {_f(cl['performance']['oot']['gini'])} against "
                              f"{_f(ch['performance']['oot']['gini'])} for the champion, a difference of "
                              f"{_f(cmp_['diff'])} (95% CI {_f(cmp_['ci_low'])} to {_f(cmp_['ci_high'])}). "
                              "The development document must justify accepting this loss of "
                              "discrimination for interpretability."})  # fmt: skip
    share = f.get("external_score_share", 0)
    if share > 0.5:
        out.append({"id": "C2", "severity": "medium", "area": "Conceptual soundness",
                    "title": "Dependence on undisclosed external scores",
                    "detail": f"External scores carry {share:.1%} of the points range. Their "
                              "construction is unknown, so a change at the provider would move "
                              "PDs without any change in this model."})  # fmt: skip
    ecl = f.get("ecl")
    if ecl:
        out.append({"id": "C3", "severity": "medium", "area": "ECL",
                    "title": "ECL is driven by a regulatory LGD backstop",
                    "detail": f"Base LGD is {ecl['assumptions']['lgd_base']:.0%} (para 97 backstop), "
                              f"giving total coverage of {ecl['by_stage']['total']['coverage_final']:.2%}. "
                              "This is an assumption, not an estimate; its range should be shown."})  # fmt: skip
        req.append({"kind": "lgd_sensitivity", "params": {"values": "0.40,0.60,0.75"},
                    "rationale": "Show coverage across a plausible LGD range."})  # fmt: skip
    out.append({"id": "C4", "severity": "medium", "area": "Stability",
                "title": "No temporal evidence of stability",
                "detail": f"Score PSI is {_f(f['stability']['score_psi']['train_vs_oot'])} because the "
                          "holdout is random. Discrimination by age is the most plausible drift "
                          "channel and should be checked directly."})  # fmt: skip
    req.append({"kind": "segment_cut", "params": {"segment": "age_band", "partition": "oot"},
                "rationale": "Check discrimination and calibration across age bands."})  # fmt: skip
    req.append({"kind": "grade_concentration", "params": {"partition": "oot"},
                "rationale": "Check how concentrated the master scale is."})  # fmt: skip
    return json.dumps({"challenges": out, "requests": req,
                       "conclusion": f"{len(out)} challenges raised; {len(req)} analyses requested."})  # fmt: skip


def _review(extras: list[dict]) -> dict:
    notes = []
    for e in extras:
        if "error" in e:
            notes.append(f"{e['kind']}: request could not be run ({e['error']}).")
            continue
        r = e["result"]
        if not r.get("table"):
            notes.append(f"{e['kind']}: no rows met the minimum size, so nothing to conclude.")
        elif e["kind"] == "lgd_sensitivity":
            cov = [row["coverage_final"] for row in r["table"]]
            notes.append(f"LGD sensitivity: final coverage ranges from {min(cov):.2%} to {max(cov):.2%} "
                         "across the requested LGDs, so the backstop choice is material.")  # fmt: skip
        elif e["kind"] == "segment_cut":
            worst = min(r["table"], key=lambda row: row["gini"])
            notes.append(f"Segment cut by {r['segment']}: the weakest band is {worst['value']} with Gini "
                         f"{_f(worst['gini'])} (AUC {_f(worst['auc'])}) and PD/DR {_f(worst['pd_to_dr'], 3)}.")  # fmt: skip
        elif e["kind"] == "grade_concentration":
            top = max(r["table"], key=lambda row: row["share"])
            notes.append(f"Grade concentration: HHI {_f(r['hhi'])}; the largest grade is {top['grade']} "
                         f"with {top['share']:.1%} of accounts.")  # fmt: skip
        else:
            notes.append(f"{e['kind']}: reviewed.")
    return {"challenges": [], "requests": [],
            "conclusion": " ".join(notes) or "No analyses to review."}  # fmt: skip


def _default(_: str) -> str:
    return "[dry-run] No LLM call made."


_CANNED: dict = {}
