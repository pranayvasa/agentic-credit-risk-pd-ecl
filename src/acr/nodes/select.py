"""Node 4: feature assembly and selection (train only)."""

from __future__ import annotations

import json
import pickle

import pandas as pd

from acr.audit import node
from acr.core import binning as B
from acr.core.selection import select_features
from acr.nodes._util import artefact_dir, cfg


@node(
    "select",
    role="Join the four legs; select scorecard features on train only by missingness, IV, "
    "train-vs-test CSI, correlation, VIF, a size cap and coefficient sign/significance; log "
    "every exclusion with a reason.",
    reads=["config", "split", "legs"],
    writes=["selection"],
)
def select(view):
    c = cfg(view)
    legs = view["legs"]
    order = ["application", "bureau", "behavioural", "previous"]
    features = pd.concat([pd.read_parquet(legs[k]["features_path"]) for k in order], axis=1)
    dictionary = pd.concat([pd.read_csv(legs[k]["dictionary_path"]) for k in order],
                           ignore_index=True)  # fmt: skip
    assignment = pd.read_parquet(view["split"]["assignment_path"])
    sel = select_features(features, dictionary, assignment, c.base.features)

    out = artefact_dir(view, "selection")
    features.to_parquet(out / "features_all.parquet")
    dictionary.to_csv(out / "feature_dictionary.csv", index=False)
    with open(out / "binning_process.pkl", "wb") as f:
        pickle.dump(sel.binning, f)
    sel.iv.sort_values("iv", ascending=False).to_csv(out / "iv_table.csv", index=False)
    sel.exclusions.to_csv(out / "exclusions.csv", index=False)
    B.binning_tables(sel.binning, list(sel.binning.variable_names)).to_csv(
        out / "binning_tables.csv", index=False)  # fmt: skip
    iv = sel.iv.set_index("feature")["iv"]
    meta = {
        "candidates": sel.candidates, "categorical": sel.categorical,
        "screened": sel.screened, "selected": sel.selected,
        "csi": {k: round(v, 6) for k, v in sel.csi.items()},
    }  # fmt: skip
    (out / "selection.json").write_text(json.dumps(meta, indent=2))
    stage_counts = sel.exclusions["stage"].value_counts().to_dict()
    leg_of = dictionary.set_index("name")["leg"]
    return {
        "selection": {
            "n_features_total": int(features.shape[1]),
            "n_candidates": len(sel.candidates),
            "n_screened": len(sel.screened),
            "n_selected": len(sel.selected),
            "selected": [
                {
                    "feature": f,
                    "leg": leg_of[f],
                    "iv": round(float(iv[f]), 4),
                    "csi": round(sel.csi[f], 4),
                    "vif": sel.vif.get(f),
                }
                for f in sel.selected
            ],  # fmt: skip
            "selected_by_leg": pd.Series([leg_of[f] for f in sel.selected])
            .value_counts()
            .to_dict(),
            "exclusions_by_stage": {k: int(v) for k, v in stage_counts.items()},
            "paths": {
                "features_all": str(out / "features_all.parquet"),
                "dictionary": str(out / "feature_dictionary.csv"),
                "binning": str(out / "binning_process.pkl"),
                "iv_table": str(out / "iv_table.csv"),
                "exclusions": str(out / "exclusions.csv"),
                "binning_tables": str(out / "binning_tables.csv"),
                "meta": str(out / "selection.json"),
            },
        }
    }
