"""Local Ollama model: the no-cost fallback. Uses the HTTP API directly (no extra dependency)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from acr.llm.base import LLMResponse


def ollama_available(host: str, model: str) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as r:
            tags = json.load(r)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return False, f"cannot reach {host}: {exc}"
    names = {m.get("name") for m in tags.get("models", [])}
    if model not in names:
        return False, f"model {model} not pulled"
    return True, "ok"


class OllamaProvider:
    name = "ollama"

    def __init__(self, host: str, model: str) -> None:
        self.host = host
        self.model = model

    def generate(self, prompt: str, *, task: str, context: dict | None = None) -> LLMResponse:
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        return LLMResponse(
            text=out.get("response", ""),
            input_tokens=int(out.get("prompt_eval_count", 0)),
            output_tokens=int(out.get("eval_count", 0)),
            model=self.model,
            provider=self.name,
        )
