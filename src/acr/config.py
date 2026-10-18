"""Typed, versioned configuration.

All YAML files under ``config/`` are validated into pydantic models. The canonical JSON
form of the merged config is hashed so every run records exactly which settings produced
its numbers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw_dir: str
    outputs_dir: str


class DataCfg(_Strict):
    id_col: str
    target_col: str
    tables: dict[str, str]


class SplitCfg(_Strict):
    oot_frac: float = Field(gt=0, lt=1)
    test_frac_of_dev: float = Field(gt=0, lt=1)


class SampleCfg(_Strict):
    default_frac: float = Field(gt=0, le=1)


class FeatureCfg(_Strict):
    iv_min: float
    iv_suspicious: float
    corr_max: float
    vif_max: float
    csi_max: float


class ScorecardCfg(_Strict):
    pdo: float
    base_score: float
    base_odds: float


class MasterScaleCfg(_Strict):
    pd_upper_bounds: list[float]

    @model_validator(mode="after")
    def _monotone(self) -> MasterScaleCfg:
        b = self.pd_upper_bounds
        if any(x >= y for x, y in zip(b, b[1:], strict=False)) or b[-1] != 1.0:
            raise ValueError("master scale bounds must be strictly increasing and end at 1.0")
        return self


class ModelCfg(_Strict):
    scorecard: ScorecardCfg
    lightgbm: dict[str, float | int]
    master_scale: MasterScaleCfg


class BaseCfg(_Strict):
    config_version: str
    seed: int
    paths: Paths
    data: DataCfg
    split: SplitCfg
    sample: SampleCfg
    features: FeatureCfg
    model: ModelCfg


class GateSet(_Strict):
    enforce: bool
    gini_min_test: float
    gini_min_oot: float
    psi_red: float
    psi_amber: float
    dq_halt_on: list[str]


class GatesCfg(_Strict):
    full: GateSet
    sample: GateSet


class GeminiCfg(_Strict):
    test_model: str
    final_model: str
    use: Literal["test", "final"]
    temperature: float

    @property
    def model(self) -> str:
        return self.test_model if self.use == "test" else self.final_model


class OllamaCfg(_Strict):
    host: str
    model: str


class TokenRate(_Strict):
    input: float
    output: float


class LLMCfg(_Strict):
    mode: Literal["dry-run", "gemini", "ollama"]
    max_critic_iterations: int = Field(ge=0, le=2)
    gemini: GeminiCfg
    ollama: OllamaCfg
    cost_per_million: dict[str, TokenRate]


class Scenario(_Strict):
    weight: float = Field(ge=0, le=1)
    pd_scalar: float = Field(gt=0)


class ECLCfg(_Strict):
    regulation: dict[str, str]
    product_category: str
    pd: dict[str, float]
    lgd: dict[str, float | list[float]]
    staging: dict[str, float | int]
    floors: dict[str, float]
    scenarios: dict[str, Scenario]

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> ECLCfg:
        total = sum(s.weight for s in self.scenarios.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"scenario weights must sum to 1, got {total}")
        return self


class AppConfig(_Strict):
    base: BaseCfg
    gates: GatesCfg
    llm: LLMCfg
    ecl: ECLCfg

    def hash(self) -> str:
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def gate_set(self, is_sample: bool) -> GateSet:
        return self.gates.sample if is_sample else self.gates.full


def _read(name: str, config_dir: Path) -> dict:
    with open(config_dir / name) as f:
        return yaml.safe_load(f)


def load_config(config_dir: Path = CONFIG_DIR, llm_mode: str | None = None) -> AppConfig:
    llm = _read("llm.yaml", config_dir)
    if llm_mode:
        llm["mode"] = llm_mode
    return AppConfig(
        base=_read("base.yaml", config_dir),
        gates=_read("gates.yaml", config_dir),
        llm=llm,
        ecl=_read("ecl_rbi.yaml", config_dir),
    )
