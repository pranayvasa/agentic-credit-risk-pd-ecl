"""Typed pipeline state.

Design rule: state carries artefact *paths*, metric dictionaries, findings and audit metadata.
It never carries DataFrames. Tables live as parquet under ``outputs/<run_id>/artefacts/``.
That keeps state serialisable and auditable, and it means anything handed to an LLM is, by
construction, a summary rather than loan-level data.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict

Severity = Literal["info", "low", "medium", "high", "red"]


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer for keys written by parallel branches (the leg fan-out)."""
    out = dict(left or {})
    for k, v in (right or {}).items():
        if k in out:
            raise ValueError(f"parallel branches both wrote '{k}'")
        out[k] = v
    return out


class Finding(BaseModel):
    """A structured validation or critic finding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    area: str
    severity: Severity
    title: str
    detail: str
    remediation: str = ""


# Whitelisted deterministic analyses the critic may request. The critic can only pick from
# this menu; there is no request type that touches model parameters.
AnalysisKind = Literal[
    "segment_cut",
    "lgd_sensitivity",
    "binning_stability_alt",
    "grade_concentration",
    "pd_floor_impact",
]


class CriticRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: AnalysisKind
    params: dict[str, str | int | float] = {}
    rationale: str


class LLMCall(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    node: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class PipelineState(TypedDict, total=False):
    # --- run setup (orchestrator) ---
    run_id: str
    run_dir: str
    config: dict[str, Any]  # validated AppConfig dumped to JSON-safe dict
    config_hash: str
    sample_frac: float | None  # None = full population
    llm_mode: str

    # --- ingestion and data quality ---
    dq: dict[str, Any]  # checks, severities, target rate, table shapes

    # --- split ---
    split: dict[str, Any]  # id-list paths and proportions per partition

    # --- legs (parallel fan-out; merged by reducer) ---
    legs: Annotated[dict[str, Any], merge_dicts]

    # --- feature assembly and selection ---
    selection: dict[str, Any]

    # --- modelling (champion + challenger) ---
    model: dict[str, Any]

    # --- independent validation (read-only over model artefacts) ---
    validation: dict[str, Any]

    # --- ECL ---
    ecl: dict[str, Any]

    # --- critic loop ---
    critic_iteration: int
    critic_requests: list[dict[str, Any]]  # pending, for this iteration
    critic_reviews: Annotated[list[dict[str, Any]], operator.add]
    extra_analyses: Annotated[list[dict[str, Any]], operator.add]

    # --- documentation and grounding ---
    documents: dict[str, str]  # doc name -> markdown path
    grounding: dict[str, Any]

    # --- cross-cutting ---
    llm_calls: Annotated[list[dict[str, Any]], operator.add]
    gate_log: Annotated[list[dict[str, Any]], operator.add]
    halted: bool
    halt_reason: str
    report: dict[str, Any]
