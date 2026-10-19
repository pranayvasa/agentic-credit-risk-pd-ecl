"""Google Gemini via the official google-genai SDK. Key read from GEMINI_API_KEY.

Transient errors (rate limits, temporary unavailability) are retried with exponential
backoff. Critic calls request JSON output so the structured review parses reliably.
"""

from __future__ import annotations

import time

from acr.llm.base import LLMResponse

TRANSIENT = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL", "DEADLINE")


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, temperature: float, max_retries: int = 6) -> None:
        from google import genai

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self._client = genai.Client()  # reads GEMINI_API_KEY from the environment

    def generate(self, prompt: str, *, task: str, context: dict | None = None) -> LLMResponse:
        from google.genai import types

        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            response_mime_type="application/json" if task.startswith("critic") else None,
        )
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=cfg
                )
                break
            except (
                Exception
            ) as exc:  # the SDK raises several error types; retry only transient ones
                if attempt == self.max_retries or not any(t in str(exc) for t in TRANSIENT):
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))
        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            model=self.model,
            provider=self.name,
        )
