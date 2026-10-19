"""Shared implementation for the four leg nodes."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from acr.core.io import read
from acr.core.legs.common import LegResult, leg_quality
from acr.core.schema import ID
from acr.nodes._util import artefact_dir


def run_leg(view: Mapping, leg: str, build: Callable) -> dict:
    t0 = time.perf_counter()
    paths = {k: v["path"] for k, v in view["dq"]["tables"].items()}
    ids = read(view["split"]["assignment_path"], [ID])[ID].to_numpy()
    ext = view["split"].get("external")
    if ext:  # unlabelled applicants get features too, so the model can score them
        ids = np.concatenate([ids, read(ext["path"], [ID])[ID].to_numpy()])
    res: LegResult = build(paths, ids)
    q = leg_quality(res)
    if q["status"] == "fail":
        raise ValueError(f"leg {leg} failed quality checks: {q['issues']}")
    out = artefact_dir(view, "legs", leg)
    res.features.to_parquet(out / "features.parquet")
    res.dictionary.to_csv(out / "dictionary.csv", index=False)
    roles = res.dictionary["role"].value_counts().to_dict()
    return {
        leg: {
            "features_path": str(out / "features.parquet"),
            "dictionary_path": str(out / "dictionary.csv"),
            "n_features": int(res.features.shape[1]),
            "roles": {k: int(v) for k, v in roles.items()},
            "dq": q,
            "seconds": round(time.perf_counter() - t0, 2),
        }
    }


def load_dictionary(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
