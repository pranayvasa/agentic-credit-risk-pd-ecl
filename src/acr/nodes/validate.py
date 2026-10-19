"""Node 6: independent validation (read-only over model artefacts) and the performance gate.

Second-line controls:
* Integrity: every model artefact's SHA-256 is checked against the checksum recorded by the
  model node, before and after validation. A mismatch is a red finding.
* Independence: this module imports no fitting code. It reads scored outputs, the points
  table and the train-fitted binning (transform only).
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from acr.audit import node
from acr.core import binning as B
from acr.core import validation as V
from acr.core.findings import build_findings, gate_decision
from acr.nodes._util import artefact_dir, cfg, is_sample, sha256

EXTERNAL = ("APP_EXT_SOURCE_1", "APP_EXT_SOURCE_2", "APP_EXT_SOURCE_3", "APP_EXT_SOURCE_MEAN",
            "APP_EXT_SOURCE_MIN")  # fmt: skip


def _integrity(model: dict) -> dict:
    bad = [k for k, p in model["paths"].items() if sha256(p) != model["checksums"][k]]
    return {"ok": not bad, "mismatches": bad}


def _r(x):
    """Round floats for state and reports (6 dp) so numbers are stable and citable."""
    if isinstance(x, dict):
        return {k: _r(v) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return round(float(x), 6)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def _segment_frame(features: pd.DataFrame) -> pd.DataFrame:
    age = features["APP_AGE_YEARS"]
    return pd.DataFrame({
        "contract_type": features["APP_NAME_CONTRACT_TYPE"],
        "income_type": features["APP_NAME_INCOME_TYPE"],
        "gender": features["APP_CODE_GENDER"],
        "age_band": pd.cut(age, [0, 30, 45, 60, 200], right=False,
                           labels=["<30", "30-44", "45-59", "60+"]),
        "bureau_history": np.where(features["BUR_COUNT"] > 0, "yes", "no"),
    }, index=features.index)  # fmt: skip


@node(
    "validate",
    role="Independent second-line validation over scored outputs: Gini/AUC/KS by partition, "
    "score PSI and feature CSI, calibration (HL, ECE, slope, grade-level binomial tests), "
    "rank-ordering, segments, champion vs challenger with bootstrap CI; structured findings; "
    "performance and stability gates. Read-only: artefact checksums verified before and after.",
    reads=["config", "sample_frac", "split", "model", "selection"],
    writes=["validation", "gate_log"],
    on_failure="A red finding with gate enforcement on routes to the halt node with a halt report.",
)
def validate(view):
    c = cfg(view)
    v, gate, sample = c.gates.validation, c.gate_set(is_sample(view)), is_sample(view)
    model, sel = view["model"], view["selection"]
    integrity_before = _integrity(model)

    scored = pd.read_parquet(model["paths"]["scored"])
    points = pd.read_csv(model["paths"]["points"])
    features = pd.read_parquet(sel["paths"]["features_all"])
    meta = json.loads(Path(sel["paths"]["meta"]).read_text())
    with open(sel["paths"]["binning"], "rb") as f:
        bp = pickle.load(f)
    selected = meta["selected"]
    oot = scored[scored["partition"] == "oot"]

    m: dict = {"champion": {}, "challenger": {}}
    for name, col, gcol in (("champion", "pd_champion", "grade_champion"),
                            ("challenger", "pd_challenger", "grade_challenger")):  # fmt: skip
        m[name]["performance"] = V.by_partition(scored, col)
        m[name]["calibration"] = {p: V.calibration(g["TARGET"], g[col])
                                  for p, g in scored.groupby("partition")}  # fmt: skip
        m[name]["grade_psi"] = V.grade_psi(scored, gcol)
    grades = V.grade_table(oot, "pd_champion", "grade_champion", v.binomial_alpha_red,
                           v.binomial_alpha_amber)  # fmt: skip
    deciles = V.decile_table(oot, "pd_champion")
    m["champion"]["rank_breaks_grade_oot"] = V.rank_order_breaks(
        grades["default_rate"], grades["n"]
    )
    m["champion"]["rank_breaks_decile_oot"] = V.rank_order_breaks(
        deciles["default_rate"], deciles["n"]
    )

    cat = [f for f in bp.variable_names if f in meta["categorical"]]
    bins = B.bins(bp, features[list(bp.variable_names)], cat)
    ext = (pd.read_parquet(model["paths"]["scored_external"])
           if "scored_external" in model["paths"] else None)  # fmt: skip
    part = scored["partition"].reindex(bins.index)
    if ext is not None:
        part = part.where(~bins.index.isin(ext.index), "external")
    csi = V.csi_table(bins, part, selected)
    csi_cols = [c for c in csi.columns if c.startswith("csi_")]
    m["stability"] = {
        "score_psi": V.score_psi(scored),
        "stress_psi": {
            f"under30_x{w:g}": V.stress_psi(scored, features["APP_AGE_YEARS"], w)
            for w in v.stress_young_weights
        },
        "csi_max": float(csi[csi_cols].to_numpy().max()) if len(csi) else 0.0,
    }
    if ext is not None:
        trn = scored[scored["partition"] == "train"]
        m["stability"]["external"] = {
            "n": int(len(ext)),
            "score_psi": V.external_psi(trn["score"], ext["score"]),
            "challenger_pd_psi": V.external_psi(trn["pd_challenger"], ext["pd_challenger"]),
            "mean_pd_champion": float(ext["pd_champion"].mean()),
            "mean_pd_champion_train": float(trn["pd_champion"].mean()),
        }
    seg = V.segments(oot, _segment_frame(features), "pd_champion")
    m["comparison"] = V.bootstrap_gini_diff(oot["TARGET"], oot["pd_champion"], oot["pd_challenger"],
                                            v.bootstrap_n, c.base.seed)  # fmt: skip
    m["external_score_share"] = V.points_range_share(points, [f for f in EXTERNAL if f in selected])

    out = artefact_dir(view, "validation")
    grades.to_csv(out / "grade_table_oot.csv", index=False)
    deciles.to_csv(out / "decile_table_oot.csv", index=False)
    csi.to_csv(out / "csi_selected.csv", index=False)
    seg.to_csv(out / "segments_oot.csv", index=False)

    integrity_after = _integrity(model)
    m["integrity"] = {"ok": integrity_before["ok"] and integrity_after["ok"],
                      "mismatches": sorted(set(integrity_before["mismatches"] + integrity_after["mismatches"]))}  # fmt: skip

    findings = build_findings(m, grades, csi, seg, v, gate, sample)
    decision, reason = gate_decision(findings, gate)
    sev = pd.Series([f.severity for f in findings]).value_counts().to_dict()
    metrics = _r(m)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "findings.json").write_text(json.dumps([f.model_dump() for f in findings], indent=2))
    return {
        "validation": {
            "status": "halt" if decision == "halt" else "pass",
            "gate_decision": decision,
            "metrics": metrics,
            "grade_table_oot": _r(grades.to_dict(orient="list")),
            "findings": [f.model_dump() for f in findings],
            "findings_by_severity": {
                k: int(sev.get(k, 0)) for k in ("red", "high", "medium", "low", "info")
            },
            "pseudo_oot": True,
            "paths": {p.stem: str(p) for p in out.iterdir()},
        },
        "gate_log": [{"gate": "validation", "decision": decision, "reason": reason}],
    }
