"""Deterministic executor for critic requests. Runs only whitelisted analyses."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from acr.audit import node, write_event
from acr.core import extras as X
from acr.nodes._util import cfg
from acr.nodes.critic import round_floats

SEGMENT_COLS = ["APP_NAME_CONTRACT_TYPE", "APP_NAME_INCOME_TYPE", "APP_CODE_GENDER",
                "APP_AGE_YEARS", "BUR_COUNT"]  # fmt: skip


@node(
    "run_extra_analyses",
    role="Run the critic's whitelisted, deterministic analysis requests over saved outputs; "
    "never touches model artefacts.",
    reads=["config", "critic_requests", "critic_iteration", "model", "ecl", "selection"],
    writes=["extra_analyses", "critic_requests"],
)
def run_extra_analyses(view):
    c = cfg(view)
    meta = json.loads(Path(view["selection"]["paths"]["meta"]).read_text())
    cols = list(dict.fromkeys(SEGMENT_COLS + meta["selected"]))
    feats = pd.read_parquet(view["selection"]["paths"]["features_all"], columns=cols)
    ecl_paths = (view.get("ecl") or {}).get("paths", {})
    inputs = X.Inputs(
        scored=pd.read_parquet(view["model"]["paths"]["scored"]),
        features=feats,
        ecl_accounts=pd.read_parquet(ecl_paths["accounts"]) if "accounts" in ecl_paths else None,
        selected=meta["selected"],
        floors=c.ecl.floors,
        pd_floor=c.ecl.pd["floor_12m"],
        lgd_base=float(c.ecl.lgd["base"]),
    )
    it = view["critic_iteration"] - 1
    done = []
    for r in view.get("critic_requests", []):
        rec = {
            "iteration": it,
            "kind": r["kind"],
            "params": r["params"],
            "rationale": r["rationale"],
        }
        try:
            rec["result"] = round_floats(X.run(r["kind"], r["params"], inputs))
        except X.BadRequest as exc:
            rec["error"] = str(exc)
        write_event(view["run_dir"], {"event": "extra_analysis", "iteration": it, "kind": r["kind"],
                                      "ok": "error" not in rec})  # fmt: skip
        done.append(rec)
    return {"extra_analyses": done, "critic_requests": []}
