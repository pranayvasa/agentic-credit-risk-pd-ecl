"""Provider abstraction: dry-run (default), Gemini, Ollama.

LLM nodes call ``get_provider``. If the selected provider is unavailable (no key, no local
server), it returns ``None`` with a reason and the node skips itself; the deterministic part
of the pipeline still completes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from acr.config import REPO_ROOT, LLMCfg

PROMPTS_DIR = REPO_ROOT / "prompts"


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, prompt: str, *, task: str) -> LLMResponse: ...


def load_prompt(name: str, version: str = "v1") -> str:
    """Prompts are versioned files under prompts/, never inline strings."""
    return (PROMPTS_DIR / f"{name}_{version}.md").read_text()


def get_provider(cfg: LLMCfg) -> tuple[LLMProvider | None, str]:
    if cfg.mode == "dry-run":
        from acr.llm.dryrun import DryRunProvider

        return DryRunProvider(), "dry-run: canned responses, no API calls"
    if cfg.mode == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            return None, "GEMINI_API_KEY not set; LLM nodes skipped"
        from acr.llm.gemini import GeminiProvider

        return GeminiProvider(
            cfg.gemini.model, cfg.gemini.temperature
        ), f"gemini:{cfg.gemini.model}"
    if cfg.mode == "ollama":
        from acr.llm.ollama import OllamaProvider, ollama_available

        ok, why = ollama_available(cfg.ollama.host, cfg.ollama.model)
        if not ok:
            return None, f"Ollama unavailable ({why}); LLM nodes skipped"
        return OllamaProvider(cfg.ollama.host, cfg.ollama.model), f"ollama:{cfg.ollama.model}"
    return None, f"unknown LLM mode {cfg.mode!r}"
