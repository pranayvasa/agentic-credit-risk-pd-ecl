"""Token cost accounting using per-model rates from config/llm.yaml."""

from __future__ import annotations

from acr.config import LLMCfg


def call_cost(
    cfg: LLMCfg, model: str, provider: str, input_tokens: int, output_tokens: int
) -> float:
    rate = cfg.cost_per_million.get(model) or cfg.cost_per_million.get(provider)
    if rate is None:
        return 0.0
    return (input_tokens * rate.input + output_tokens * rate.output) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough estimate (about 4 characters per token) for dry-run and pre-flight costing."""
    return max(1, len(text) // 4)
