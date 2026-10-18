"""Node 0: orchestrator / run setup."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from acr.audit import node, write_event
from acr.config import AppConfig, resolve_path


@node(
    "setup",
    role="Validate config, create run_id and run folder, snapshot config, start the audit trail.",
    reads=["config", "sample_frac", "llm_mode"],
    writes=["run_id", "run_dir", "config_hash", "critic_iteration", "critic_requests", "halted"],
    on_failure="Invalid config raises a pydantic ValidationError before any data is touched.",
)
def setup(view):
    cfg = AppConfig.model_validate(view["config"])
    h = cfg.hash()
    frac = view.get("sample_frac")
    scope = "full" if frac is None else f"s{round(frac * 100):02d}"
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{scope}_{h}"
    run_dir = resolve_path(cfg.base.paths.outputs_dir) / run_id
    (run_dir / "artefacts").mkdir(parents=True, exist_ok=False)
    (Path(run_dir) / "config_snapshot.json").write_text(
        json.dumps(cfg.model_dump(mode="json"), indent=2, sort_keys=True)
    )
    write_event(
        run_dir,
        {
            "event": "run_start",
            "run_id": run_id,
            "config_hash": h,
            "config_version": cfg.base.config_version,
            "seed": cfg.base.seed,
            "sample_frac": frac,
            "llm_mode": view.get("llm_mode"),
        },
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "config_hash": h,
        "critic_iteration": 0,
        "critic_requests": [],
        "halted": False,
    }
