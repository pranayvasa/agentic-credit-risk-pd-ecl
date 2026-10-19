"""Whitelisted deterministic analyses the LLM critic may request.

The critic can only name one of these kinds (``acr.state.AnalysisKind``) with small, validated
parameters. Each analysis reads saved outputs (scored PDs, features, the ECL account table)
and returns a small summary table. None of them fit, tune or modify a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from acr.core.metrics import gini, psi

SEGMENTS = ("contract_type", "income_type", "gender", "age_band", "bureau_history")


@dataclass
class Inputs:
    scored: pd.DataFrame  # labelled scored outputs (partition, TARGET, PDs, grades)
    features: pd.DataFrame  # leg features (selected columns are enough)
    ecl_accounts: pd.DataFrame | None
    selected: list[str]
    floors: dict
    pd_floor: float
    lgd_base: float


class BadRequest(ValueError):
    """Parameters outside what the analysis accepts."""


def _segment_series(features: pd.DataFrame, name: str) -> pd.Series:
    if name == "contract_type":
        return features["APP_NAME_CONTRACT_TYPE"].astype("string")
    if name == "income_type":
        return features["APP_NAME_INCOME_TYPE"].astype("string")
    if name == "gender":
        return features["APP_CODE_GENDER"].astype("string")
    if name == "age_band":
        return pd.cut(features["APP_AGE_YEARS"], [0, 25, 30, 40, 50, 60, 200], right=False,
                      labels=["<25", "25-29", "30-39", "40-49", "50-59", "60+"]).astype("string")  # fmt: skip
    if name == "bureau_history":
        return pd.Series(np.where(features["BUR_COUNT"] > 0, "yes", "no"), index=features.index)
    raise BadRequest(f"segment must be one of {SEGMENTS}")


def segment_cut(x: Inputs, segment: str = "age_band", partition: str = "oot") -> dict:
    if partition not in ("train", "test", "oot"):
        raise BadRequest("partition must be train, test or oot")
    df = x.scored[x.scored["partition"] == partition]
    seg = _segment_series(x.features, segment).reindex(df.index).fillna("missing")
    rows = []
    for val, g in df.groupby(seg):
        if len(g) < 200:
            continue
        y = g["TARGET"]
        gi = gini(y, g["pd_champion"]) if 0 < y.sum() < len(y) else float("nan")
        rows.append({"value": str(val), "n": int(len(g)), "default_rate": float(y.mean()),
                     "mean_pd": float(g["pd_champion"].mean()),
                     "pd_to_dr": float(g["pd_champion"].mean() / y.mean()) if y.mean() else float("nan"),
                     "gini": gi, "auc": (gi + 1) / 2})  # fmt: skip
    return {"segment": segment, "partition": partition, "table": rows}


def lgd_sensitivity(x: Inputs, values: str = "0.40,0.60,0.75") -> dict:
    if x.ecl_accounts is None:
        raise BadRequest("no ECL results in this run")
    try:
        lgds = [float(v) for v in str(values).split(",")]
    except ValueError as exc:
        raise BadRequest("values must be comma-separated numbers") from exc
    if not lgds or len(lgds) > 8 or not all(0 < v <= 1 for v in lgds):
        raise BadRequest("1 to 8 LGD values in (0, 1]")
    a = x.ecl_accounts
    unit = a["ecl_model"] / x.lgd_base  # ECL is linear in LGD: unit ECL at LGD = 1
    rows = []
    for v in lgds:
        model = unit * v
        final = np.maximum(model, a["ecl_floor"])
        rows.append({"lgd": v, "ecl_model": float(model.sum()), "ecl_final": float(final.sum()),
                     "coverage_final": float(final.sum() / a["ead0"].sum()),
                     "floor_binding_share": float((a["ecl_floor"] > model).mean())})  # fmt: skip
    return {"table": rows}


def binning_stability_alt(x: Inputs, n_bins: int = 10) -> dict:
    """CSI of selected numeric features on equal-frequency bins (independent of optbinning)."""
    n_bins = int(n_bins)
    if not 4 <= n_bins <= 20:
        raise BadRequest("n_bins must be 4..20")
    part = x.scored["partition"]
    rows = []
    for f in x.selected:
        col = x.features[f]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        col = col.reindex(part.index).astype("float64")
        tr = col[part == "train"]
        edges = np.unique(np.nanquantile(tr, np.linspace(0, 1, n_bins + 1)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        b = pd.cut(col, edges, labels=False).fillna(-1)
        rows.append({"feature": f, "csi_test": psi(b[part == "train"], b[part == "test"]),
                     "csi_oot": psi(b[part == "train"], b[part == "oot"])})  # fmt: skip
    return {"n_bins": n_bins, "table": rows}


def grade_concentration(x: Inputs, partition: str = "oot") -> dict:
    df = x.scored[x.scored["partition"] == partition]
    share = df["grade_champion"].value_counts(normalize=True).sort_index()
    rows = [{"grade": int(g), "share": float(s)} for g, s in share.items()]
    out = {"partition": partition, "hhi": float((share**2).sum()), "table": rows}
    if x.ecl_accounts is not None:
        ead = x.ecl_accounts.groupby("grade")["ead0"].sum()
        out["ead_hhi"] = float(((ead / ead.sum()) ** 2).sum())
    return out


def pd_floor_impact(x: Inputs, floor: float = 0.001) -> dict:
    """How many accounts an alternative (higher) 12-month PD floor would bind, by stage."""
    floor = float(floor)
    if not 0 < floor < 0.05:
        raise BadRequest("floor must be in (0, 0.05)")
    if x.ecl_accounts is None:
        raise BadRequest("no ECL results in this run")
    a = x.ecl_accounts
    rows = [{"stage": int(s), "accounts": int(len(g)),
             "share_below_regulatory_floor": float((g["pd12"] < x.pd_floor).mean()),
             "share_below_alternative_floor": float((g["pd12"] < floor).mean())}
            for s, g in a.groupby("stage")]  # fmt: skip
    return {"regulatory_floor": x.pd_floor, "alternative_floor": floor, "table": rows}


ANALYSES = {
    "segment_cut": segment_cut,
    "lgd_sensitivity": lgd_sensitivity,
    "binning_stability_alt": binning_stability_alt,
    "grade_concentration": grade_concentration,
    "pd_floor_impact": pd_floor_impact,
}


def run(kind: str, params: dict, x: Inputs) -> dict:
    """Run one whitelisted analysis. Unknown parameters are rejected."""
    import inspect

    fn = ANALYSES[kind]
    allowed = set(inspect.signature(fn).parameters) - {"x"}
    unknown = set(params) - allowed
    if unknown:
        raise BadRequest(
            f"unknown parameters for {kind}: {sorted(unknown)} (allowed: {sorted(allowed)})"
        )
    return fn(x, **params)
