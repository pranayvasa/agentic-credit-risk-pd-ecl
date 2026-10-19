"""Node 8: LLM critic, a skeptical second-line validator.

The critic sees summarised metrics only. Its output is parsed into a fixed schema: free-text
challenges plus requests drawn from a whitelisted menu of deterministic analyses
(``acr.state.AnalysisKind``). Nothing in that schema can reference or change model parameters.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from acr.audit import node, write_event
from acr.core.facts import assert_summary_only, build_facts
from acr.llm import get_provider, load_prompt
from acr.llm.cost import call_cost
from acr.nodes._util import cfg
from acr.state import CriticRequest, LLMCall


def round_floats(x, nd: int = 6):
    if isinstance(x, dict):
        return {k: round_floats(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [round_floats(v, nd) for v in x]
    if isinstance(x, float):
        return round(x, nd)
    return x


def payload_for(view) -> dict:
    """The only thing the critic sees: curated facts plus results of its own requests."""
    payload = {"facts": build_facts(view), "extra_analyses": view.get("extra_analyses", [])}
    assert_summary_only(payload)
    return payload


def _parse(text: str) -> tuple[dict, list[dict], list[str]]:
    errors: list[str] = []
    try:
        raw = json.loads(text.strip().removeprefix("```json").removesuffix("```"))
    except json.JSONDecodeError as exc:
        return {"challenges": [], "conclusion": text[:500]}, [], [f"unparseable JSON: {exc}"]
    requests: list[dict] = []
    for r in raw.get("requests", []):
        try:
            requests.append(CriticRequest.model_validate(r).model_dump())
        except ValidationError as exc:
            errors.append(
                f"rejected request {r!r}: not on the whitelist ({exc.error_count()} errors)"
            )
    review = {"challenges": raw.get("challenges", []), "conclusion": raw.get("conclusion", "")}
    return review, requests, errors


@node(
    "critic",
    role="LLM second-line challenger: reviews validation and ECL summaries, may request whitelisted "
    "deterministic analyses. Cannot change the model.",
    reads=[
        "config",
        "config_hash",
        "sample_frac",
        "dq",
        "split",
        "selection",
        "model",
        "explain",
        "validation",
        "ecl",
        "extra_analyses",
        "critic_iteration",
    ],
    writes=["critic_reviews", "critic_requests", "critic_iteration", "llm_calls"],
    uses_llm=True,
    on_failure="If no LLM is available the node records a skip and the pipeline continues.",
)
def critic(view):
    c = cfg(view)
    it = view.get("critic_iteration", 0)
    provider, why = get_provider(c.llm)
    if provider is None:
        return {
            "critic_reviews": [{"iteration": it, "skipped": True, "reason": why}],
            "critic_requests": [],
            "critic_iteration": it + 1,
            "llm_calls": [],
        }
    payload = payload_for(view)
    prompt = (
        load_prompt("critic")
        .replace("{{ITERATION}}", str(it + 1))
        .replace("{{MAX}}", str(c.llm.max_critic_iterations))
        .replace("{{PAYLOAD}}", json.dumps(payload, default=str, indent=1))
    )
    resp = provider.generate(prompt, task=f"critic_iter{min(it, 2)}", context=payload)
    review, requests, errors = _parse(resp.text)
    for e in errors:
        write_event(view["run_dir"], {"event": "critic_request_rejected", "detail": e})
    for r in requests:
        write_event(view["run_dir"], {"event": "critic_request", "iteration": it, **r})
    call = LLMCall(
        node="critic",
        provider=resp.provider,
        model=resp.model,
        input_tokens=resp.input_tokens,
        output_tokens=resp.output_tokens,
        cost_usd=call_cost(c.llm, resp.model, resp.provider, resp.input_tokens, resp.output_tokens),
    )
    return {
        "critic_reviews": [{"iteration": it, "provider": why, **review, "rejected": errors}],
        "critic_requests": requests,
        "critic_iteration": it + 1,
        "llm_calls": [call.model_dump()],
    }
