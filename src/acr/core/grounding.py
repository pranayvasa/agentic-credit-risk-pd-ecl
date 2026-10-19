"""Grounding check: every number in a generated document must exist in the run's facts.

How a number in the text is matched (with d = decimals shown in the text):
* plain "0.5008"   -> some fact rounds to 0.5008 at d decimals (or, as a percentage-point
                      value, some fact x 100 rounds to it);
* percent "4.48%"  -> some fact rounds to 0.0448 at d + 2 decimals, or a fact already in
                      percentage units rounds to 4.48;
* scaled "2,059.4 million" (also bn, billion, crore, lakh) -> some fact is within half a unit
                      of the last shown digit after scaling.
Structural numbers are allowlisted rather than verified: years, regulatory references
(SR 11-7, RBI/DOR/2026-27/398, paragraph numbers), dates, "95% CI", list markers and
small integers (20 or less: stages, grades, counts of items).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

SCALES = {"million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "crore": 1e7, "lakh": 1e5}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"

# Removed before extraction: references and structural text whose digits are not data.
STRUCTURAL = [
    r"SR\s?11-7",
    r"RBI/DOR/\d{4}-\d{2}/\d+",
    r"DOR\.[A-Z.]+No\.\d+/[\d.]+/\d{4}-\d{2}",
    r"\b(?:para(?:graph)?s?|paras)\s+\d+(?:\(\d+\))?(?:\([a-z]+\))?(?:\s*(?:-|to|and)\s*\d+)?",
    r"\b\d{1,2}\s+(?:" + MONTHS + r")\s+\d{4}\b",
    r"\b(?:" + MONTHS + r")\s+\d{4}\b",
    r"\b95%\s*(?:CI|confidence|bootstrap)",
    r"^\s*\d+\.\s",  # numbered list markers
    r"^#+\s*\d+(?:\.\d+)*\.?",  # numbered headings
    r"\b\d{4}-\d{2}-\d{2}\b",  # ISO dates
]
NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?![A-Za-z0-9_])"
    r"(\s?%|\s?(?:million|mn|billion|bn|crore|lakh)\b)?"
)


@dataclass
class Token:
    text: str
    value: float
    decimals: int
    unit: str  # "", "%", or a scale word
    context: str
    status: str = ""  # verified | allowlisted | unverified


@dataclass
class Report:
    tokens: list[Token] = field(default_factory=list)

    @property
    def counts(self) -> dict:
        c = {"verified": 0, "allowlisted": 0, "unverified": 0}
        for t in self.tokens:
            c[t.status] += 1
        return c

    @property
    def unverified(self) -> list[Token]:
        return [t for t in self.tokens if t.status == "unverified"]


def _strip_structural(text: str) -> str:
    for pat in STRUCTURAL:
        text = re.sub(
            pat, lambda m: " " * len(m.group(0)), text, flags=re.MULTILINE | re.IGNORECASE
        )
    return text


def extract(text: str) -> list[Token]:
    clean = _strip_structural(text)
    out = []
    for m in NUMBER.finditer(clean):
        whole, frac, unit = m.group(1), m.group(2) or "", (m.group(3) or "").strip().lower()
        value = float(whole.replace(",", "") + frac)
        ctx = text[max(0, m.start() - 45) : m.end() + 25].replace("\n", " ")
        out.append(Token(m.group(0).strip(), value, max(len(frac) - 1, 0), unit, ctx))
    return out


def _allowlisted(t: Token) -> bool:
    if t.unit:
        return False
    return t.decimals == 0 and (t.value <= 20 or 1990 <= t.value <= 2100)


def _matches(t: Token, refs: np.ndarray) -> bool:
    d, v = t.decimals, t.value
    half = 0.5 * 10.0**-d + 1e-12
    if t.unit == "%":
        return bool(np.any(np.abs(refs * 100 - v) <= half) or np.any(np.abs(refs - v) <= half))
    if t.unit in SCALES:
        s = SCALES[t.unit]
        return bool(np.any(np.abs(refs / s - v) <= half))
    return bool(np.any(np.abs(refs - v) <= half) or np.any(np.abs(refs * 100 - v) <= half))


def check(text: str, reference: list[float]) -> Report:
    refs = np.abs(np.asarray(reference, dtype="float64"))
    report = Report(extract(text))
    for t in report.tokens:
        if _matches(t, refs):
            t.status = "verified"
        elif _allowlisted(t):
            t.status = "allowlisted"
        else:
            t.status = "unverified"
    return report
