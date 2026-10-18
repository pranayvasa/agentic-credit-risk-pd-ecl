"""Append-only JSONL audit trail and the node contract.

Every graph node is wrapped by :func:`node`, which:

* gives the node a read-only view containing only the state keys it declared in ``reads``
  (reading anything else raises ``KeyError``);
* rejects any returned key not declared in ``writes``;
* logs start, end (or error), input and output summaries, and timing to ``audit.jsonl``.

The declarations are also kept in a registry, which the README generator uses to build the
per-node table, so the documentation cannot drift from the code.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from acr.state import PipelineState

AUDIT_FILE = "audit.jsonl"


class ContractViolation(RuntimeError):
    """A node read or wrote a state key outside its declared contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _summarise(value: Any, depth: int = 0) -> Any:
    """Compact, JSON-safe summary of a state value for the audit trail."""
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value if len(value) <= 120 else value[:117] + "..."
    if isinstance(value, Mapping):
        if depth >= 1:
            return {"type": "dict", "keys": sorted(map(str, value))[:12]}
        return {str(k): _summarise(v, depth + 1) for k, v in list(value.items())[:20]}
    if isinstance(value, (list, tuple)):
        return {"type": "list", "len": len(value)}
    return {"type": type(value).__name__}


def write_event(run_dir: str | Path, event: dict[str, Any]) -> None:
    path = Path(run_dir) / AUDIT_FILE
    record = {"ts": _now(), **event}
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def read_events(run_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / AUDIT_FILE
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass(frozen=True)
class NodeSpec:
    name: str
    role: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    uses_llm: bool
    on_failure: str


NODE_REGISTRY: dict[str, NodeSpec] = {}


def node(
    name: str,
    *,
    role: str,
    reads: list[str],
    writes: list[str],
    uses_llm: bool = False,
    on_failure: str = "Exception propagates; run stops and the audit trail records the error.",
) -> Callable[[Callable[[Mapping[str, Any]], dict[str, Any]]], Callable[[PipelineState], dict]]:
    """Decorator enforcing a node's declared read/write contract and auditing it."""
    always_readable = ("run_id", "run_dir")
    allowed_reads = tuple(dict.fromkeys([*always_readable, *reads]))
    allowed_writes = tuple(writes)

    def deco(fn: Callable[[Mapping[str, Any]], dict[str, Any]]) -> Callable[[PipelineState], dict]:
        spec = NodeSpec(name, role, tuple(reads), allowed_writes, uses_llm, on_failure)
        NODE_REGISTRY[name] = spec

        def wrapped(state: PipelineState) -> dict[str, Any]:
            run_dir = state.get("run_dir")
            view = MappingProxyType({k: state[k] for k in allowed_reads if k in state})
            if run_dir:
                write_event(
                    run_dir,
                    {
                        "event": "node_start",
                        "node": name,
                        "inputs": {k: _summarise(view[k]) for k in reads if k in view},
                    },
                )
            t0 = time.perf_counter()
            try:
                out = fn(view) or {}
            except Exception as exc:
                if run_dir:
                    write_event(
                        run_dir,
                        {
                            "event": "node_error",
                            "node": name,
                            "error": f"{type(exc).__name__}: {exc}",
                            "seconds": round(time.perf_counter() - t0, 3),
                        },
                    )
                raise
            undeclared = set(out) - set(allowed_writes)
            if undeclared:
                raise ContractViolation(
                    f"node '{name}' wrote undeclared keys: {sorted(undeclared)}"
                )
            # run_dir is created by the setup node, so setup's start is logged after the fact.
            if not run_dir and out.get("run_dir"):
                run_dir = out["run_dir"]
                write_event(run_dir, {"event": "node_start", "node": name, "inputs": {}})
            if run_dir:
                write_event(
                    run_dir,
                    {
                        "event": "node_end",
                        "node": name,
                        "outputs": {k: _summarise(v) for k, v in out.items()},
                        "seconds": round(time.perf_counter() - t0, 3),
                    },
                )
            return out

        wrapped.__name__ = fn.__name__
        wrapped.spec = spec  # type: ignore[attr-defined]
        return wrapped

    return deco
