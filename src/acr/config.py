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


# Author's hard limits for a lean model. Neither pipeline runs nor tuning may exceed them.
MAX_SCORECARD_FEATURES = 15
MAX_TREE_DEPTH = 10


def check_depth(depth: int) -> int:
    if not 1 <= int(depth) <= MAX_TREE_DEPTH:
        raise ValueError(f"LightGBM max_depth must be 1..{MAX_TREE_DEPTH} (no unlimited depth), "
                         f"got {depth}")  # fmt: skip
    return int(depth)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Paths(_Strict):
    raw_dir: str
    interim_dir: str
    outputs_dir: str


class RuntimeCfg(_Strict):
    max_concurrency: int = Field(ge=1)


class DQCfg(_Strict):
    min_rows: int
    target_rate_band: tuple[float, float]
    high_missing_share: float
    duplicate_share_max: float


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
    min_bin_size: float
    max_n_bins: int
    missing_max: float
    max_features: int = Field(ge=1, le=MAX_SCORECARD_FEATURES)
    p_value_max: float
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


class LGBMFitCfg(_Strict):
    validation_frac: float
    early_stopping_rounds: int


class CalibrationCfg(_Strict):
    cv_folds: int
    ece_max: float
    slope_tol: float


class ModelCfg(_Strict):
    scorecard: ScorecardCfg
    lightgbm_fit: LGBMFitCfg
    calibration: CalibrationCfg
    lightgbm: dict[str, float | int]

    @model_validator(mode="after")
    def _lean_trees(self) -> ModelCfg:
        if "max_depth" not in self.lightgbm:
            raise ValueError("LightGBM max_depth must be set explicitly (limit 10)")
        check_depth(self.lightgbm["max_depth"])
        return self

    master_scale: MasterScaleCfg


class BaseCfg(_Strict):
    config_version: str
    seed: int
    paths: Paths
    runtime: RuntimeCfg
    data: DataCfg
    dq: DQCfg
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


class ValidationCfg(_Strict):
    gini_amber_margin: float
    csi_amber: float
    csi_red: float
    binomial_alpha_red: float
    binomial_alpha_amber: float
    calibration_ratio_tol: float
    overfit_gap_medium: float
    challenger_gap_medium: float
    segment_gini_min: float
    segment_min_n: int
    external_score_share_max: float
    grade_share_min: float
    grade_share_max: float
    bootstrap_n: int
    stress_young_weights: list[float]


class GatesCfg(_Strict):
    full: GateSet
    sample: GateSet
    validation: ValidationCfg


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


class BookCfg(_Strict):
    partition: Literal["train", "test", "oot"]
    seasoning_months: int = Field(ge=0)


class TermCfg(_Strict):
    rate_source: Literal["previous_application"]
    min_months: int = Field(ge=1)
    max_months: int = Field(ge=1)
    fallback_monthly_rate: float = Field(gt=0, lt=1)


class RevolvingCfg(_Strict):
    life_months: int = Field(ge=1)
    ccf: float = Field(gt=0, le=1)


class LifetimeCfg(_Strict):
    primary: Literal["constant_hazard", "markov"]
    markov_default_bucket: int = Field(ge=1, le=5)


class ECLCfg(_Strict):
    regulation: dict[str, str]
    product_category: str
    book: BookCfg
    term: TermCfg
    revolving: RevolvingCfg
    lifetime: LifetimeCfg
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


def resolve_path(p: str | Path) -> Path:
    """Config paths are relative to the repo root unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


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
