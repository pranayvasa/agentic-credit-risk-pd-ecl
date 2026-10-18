"""Helpers shared by nodes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from acr.config import AppConfig


def cfg(view: Mapping[str, Any]) -> AppConfig:
    return AppConfig.model_validate(view["config"])


def artefact_dir(view: Mapping[str, Any], *parts: str) -> Path:
    p = Path(view["run_dir"], "artefacts", *parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_sample(view: Mapping[str, Any]) -> bool:
    return view.get("sample_frac") is not None


STUB = "stub: implemented in a later phase"
