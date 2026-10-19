"""Google Gemini via the official google-genai SDK. Key read from GEMINI_API_KEY."""

from __future__ import annotations

from acr.llm.base import LLMResponse


class GeminiProvider:
    name = "gemini"

    def __init__(self, model: str, temperature: float) -> None:
        from google import genai

        self.model = model
        self.temperature = temperature
        self._client = genai.Client()  # reads GEMINI_API_KEY from the environment

    def generate(self, prompt: str, *, task: str, context: dict | None = None) -> LLMResponse:
        from google.genai import types

        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=self.temperature),
        )
        usage = resp.usage_metadata
        return LLMResponse(
            text=resp.text or "",
            input_tokens=int(getattr(usage, "prompt_token_count", 0) or 0),
            output_tokens=int(getattr(usage, "candidates_token_count", 0) or 0),
            model=self.model,
            provider=self.name,
        )
