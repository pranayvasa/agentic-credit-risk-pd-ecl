"""Dry-run provider: deterministic canned responses, zero cost.

Token counts are estimated from text length so the cost report can show what a real run
would roughly consume.
"""

from __future__ import annotations

import json

from acr.llm.base import LLMResponse
from acr.llm.cost import estimate_tokens


class DryRunProvider:
    name = "dry-run"
    model = "dry-run"

    def generate(self, prompt: str, *, task: str) -> LLMResponse:
        text = _CANNED.get(task, _default)(prompt)
        return LLMResponse(
            text, estimate_tokens(prompt), estimate_tokens(text), self.model, self.name
        )


def _critic_first(_: str) -> str:
    return json.dumps(
        {
            "challenges": [
                {
                    "id": "C1",
                    "severity": "medium",
                    "area": "ECL",
                    "title": "LGD assumption drives ECL",
                    "detail": "LGD is a regulatory backstop, not an estimate. Show ECL sensitivity to LGD.",
                }
            ],
            "requests": [
                {"kind": "lgd_sensitivity", "params": {}, "rationale": "Quantify LGD impact."}
            ],
            "conclusion": "Further analysis requested.",
        }
    )


def _critic_second(_: str) -> str:
    return json.dumps(
        {
            "challenges": [],
            "requests": [],
            "conclusion": "Requested analyses reviewed; no further requests.",
        }
    )


def _default(_: str) -> str:
    return "[dry-run] No LLM call made."


_CANNED = {
    "critic_iter0": _critic_first,
    "critic_iter1": _critic_second,
}
