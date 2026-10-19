"""Contradiction check: prose must not contradict the validation findings or the run's facts.

Numbers can be right while the words are wrong ("rank-ordering is strong" when a finding says
it breaks). Each rule fires only when its condition holds in this run (a finding is present, or
a fact is true), and flags a prose sentence that makes the contradicting claim without a
qualifying word. Rules are deliberately narrow: a missed contradiction is possible, a false
alarm should be rare. Tables and code-rendered finding details are not checked.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass

NEG = r"\b(not|no|never|without|untested|unverified|lack\w*|rather than|limited|cannot|can't|remains?|except|but|although|however|despite|break\w*|revers\w*|only|may|could|would)\b"


@dataclass(frozen=True)
class Rule:
    name: str
    applies: Callable[[Mapping], bool]
    claim: str
    unless: str = NEG


def _has_finding(prefix: str) -> Callable[[Mapping], bool]:
    return lambda f: any(
        x["title"].startswith(prefix) for x in f.get("findings", {}).get("list", [])
    )


RULES = [
    Rule("rank-ordering described as clean while a finding reports breaks",
         _has_finding("Rank-ordering breaks"),
         r"rank[- ]order\w*[^.]*\b(strong|consistent|robust|monoton\w*|preserved|intact|clean)\b"),
    Rule("temporal stability claimed while testing is pseudo-OOT",
         _has_finding("Stability testing is not temporal"),
         r"\b(temporally (stable|validated|robust|tested)|stable over time|proven over time|validated over time|"
         r"(demonstrat\w*|confirm\w*|establish\w*|prove[sn]?|shows?|evidenc\w*|pass\w*)\b[^.]*\btemporal (stability|validation|robustness))\b"),
    Rule("external scores described as documented or validated",
         _has_finding("Champion uses undisclosed external scores"),
         r"external (credit )?scores?[^.]*\b(are|is|were|was|have been|has been) (fully )?(documented|disclosed|validated|transparent)\b"),
    Rule("target described as the regulatory default definition",
         _has_finding("Target is not the regulatory default definition"),
         r"\b(target|default definition)\b[^.]*\b(aligns?|aligned|consistent|matches|meets|equals)\b[^.]*\bregulatory\b",
         r"\b(not|does not|rather than|differ\w*|deviat\w*)\b"),
    Rule("challenger described as better while the champion outperforms it",
         _has_finding("Champion outperforms the benchmark"),
         r"\b(challenger|benchmark|scorecard)\b[^.]*\b(outperforms?|beats?|is (better|stronger|superior))\b[^.]*\bchampion\b"),
    Rule("critic said to have prompted analyses when it requested none",
         lambda f: f.get("_extra_analyses_run", 0) == 0 and f.get("_critic_ran", False),
         r"\b(prompt\w*|request\w*|commission\w*|ran|triggered)\b[^.]*\b(analys\w*|reviews?)\b",
         r"\b(no|not|without|did not|none|may|can|could|only|whitelisted)\b"),
]  # fmt: skip


def _sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith(("|", "#", ">", "**V", "*Review")):
            continue
        out += [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]
    return out


def check(text: str, facts: Mapping, extra_analyses: list, critic_reviews: list) -> list[dict]:
    ctx = {**facts, "_extra_analyses_run": len(extra_analyses),
           "_critic_ran": any(not r.get("skipped") for r in critic_reviews)}  # fmt: skip
    active = [r for r in RULES if r.applies(ctx)]
    hits = []
    for sent in _sentences(text):
        low = sent.lower()
        for r in active:
            if re.search(r.claim, low) and not re.search(r.unless, low):
                hits.append({"rule": r.name, "sentence": sent[:300]})
    return hits
