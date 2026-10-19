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
    status: str = ""  # verified | allowlisted | unverified | misattributed


@dataclass
class Report:
    tokens: list[Token] = field(default_factory=list)

    @property
    def counts(self) -> dict:
        c = {"verified": 0, "allowlisted": 0, "unverified": 0, "misattributed": 0}
        for t in self.tokens:
            c[t.status] += 1
        return c

    @property
    def unverified(self) -> list[Token]:
        return [t for t in self.tokens if t.status == "unverified"]

    @property
    def misattributed(self) -> list[Token]:
        return [t for t in self.tokens if t.status == "misattributed"]


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


# --- label-aware check -------------------------------------------------------------------
# A number can exist in the facts and still be attached to the wrong label ("challenger PD
# PSI of 0.015" when 0.015 is really a stress-test PSI). For decimals and percentages in prose:
#   1. take the words between the previous number (or clause start) and this number;
#   2. find the measure named closest to the number (Gini, AUC, PSI, slope, p-value...);
#   3. if qualifiers are named too (champion, challenger, train, OOT, stress...), narrow to the
#      facts carrying the most of them;
#   4. the number must equal one of those facts. If it only equals some other fact, it is
#      "misattributed". With no measure named, only the existence check applies.
# Tables are skipped: code renders them straight from facts.

PHRASES = [
    (r"out[- ]of[- ]time", "oot"), (r"probability of default", "pd"),
    (r"mean pd\s*/\s*default rate|pd\s*/\s*dr|pd-to-dr|pd to dr", "pdtodr"),
    (r"default rate", "defaultrate"), (r"p[- ]value", "pvalue"),
    (r"expected calibration error|calibration error", "ece"),
    (r"population stability index", "psi"), (r"characteristic stability index", "csi"),
    (r"hosmer[- ]lemeshow", "hl"), (r"information value", "iv"),
    (r"loss given default", "lgd"), (r"expected credit loss", "ecl"),
    (r"exposure at default", "ead"), (r"kolmogorov[- ]smirnov", "ks"),
    (r"\bbenchmark\b", "challenger"), (r"application_test|application test", "external"),
    (r"\bleads? by\b", "lead"), (r"\bdifference\b", "lead"),
    (r"constant[- ]hazard", "hazard"),
]  # fmt: skip
PATH_PHRASES = [("pd_to_dr", "pdtodr"), ("default_rate", "defaultrate"), ("p_value", "pvalue"),
                ("constant_hazard", "hazard"), ("champion_lead", "lead")]  # fmt: skip
MEASURES = {"gini", "auc", "ks", "psi", "csi", "ece", "slope", "intercept", "hl", "pvalue",
            "brier", "iv", "lgd", "ecl", "ead", "coverage", "pdtodr", "defaultrate", "lead",
            "scalar", "weight", "spearman", "overlap", "hhi"}  # fmt: skip
LABEL_WINDOW = 8  # a label is the few words just before the number
QUALIFIERS = {"champion", "challenger", "scorecard", "lightgbm", "external", "stress", "train",
              "test", "oot", "before", "after", "upside", "downside", "base", "markov", "hazard"}  # fmt: skip


def _canon(text: str) -> str:
    text = text.lower()
    for pat, rep in PHRASES:
        text = re.sub(pat, rep, text)
    return text


def path_tokens(path: str) -> set[str]:
    p = path.lower()
    for a, b in PATH_PHRASES:
        p = p.replace(a, b)
    return {t for t in re.split(r"[._\[\]]+", p) if t and not t.isdigit()}


def leaf_tokens(path: str) -> set[str]:
    """Tokens of the last two path segments: what the fact measures, not its namespace."""
    segs = re.sub(r"\[\d+\]", "", path).split(".")
    return path_tokens(".".join(segs[-2:]))


def _label_words(line: str, prev_end: int, start: int) -> list[str]:
    seg = re.sub(r"\([^()]*\)", " ", line[prev_end:start])  # closed asides are not labels
    seg = re.split(r"[.;!?:,](?:\s|$)|[()]", seg)[-1]
    return re.findall(r"[a-z0-9]+", _canon(seg))[-LABEL_WINDOW:]


def _close(x: np.ndarray, v: float, d: int) -> np.ndarray:
    """Equal at the shown precision: rounded, or truncated (a formatting slip, not an invention)."""
    unit = 10.0**-d
    return (np.abs(x - v) <= 0.5 * unit + 1e-12) | ((x - v >= -1e-12) & (x - v < unit))


def _value_matches(t: Token, values: np.ndarray) -> np.ndarray:
    d, v = t.decimals, t.value
    vals = np.abs(values)
    if t.unit == "%":
        return _close(vals * 100, v, d) | _close(vals, v, d)
    if t.unit in SCALES:
        return _close(vals / SCALES[t.unit], v, d)
    return _close(vals, v, d) | _close(vals * 100, v, d)


def check_labelled(text: str, labelled: list[tuple[str, float]]) -> Report:
    """Existence check for every number, plus a label check for decimals and percentages in
    prose. Statuses: verified, allowlisted, unverified, misattributed."""
    paths = [p for p, _ in labelled]
    values = np.asarray([v for _, v in labelled], dtype="float64")
    ptoks = [path_tokens(p) for p in paths]
    ltoks = [leaf_tokens(p) for p in paths]
    report = Report()
    for line in text.splitlines():
        prose = not line.lstrip().startswith("|")
        clean = _strip_structural(line)
        prev_end = 0
        for m in NUMBER.finditer(clean):
            whole, frac, unit = m.group(1), m.group(2) or "", (m.group(3) or "").strip().lower()
            t = Token(m.group(0).strip(), float(whole.replace(",", "") + frac),
                      max(len(frac) - 1, 0), unit, line[max(0, m.start() - 45): m.end() + 25])  # fmt: skip
            hits = _value_matches(t, values)
            t.status = "verified"
            if not hits.any():
                t.status = "allowlisted" if _allowlisted(t) else "unverified"
            elif prose and (t.decimals > 0 or t.unit == "%"):
                words = _label_words(line, prev_end, m.start())
                measures = {w for w in words if w in MEASURES}
                if "lead" in measures:  # a difference names the gap, not the metric it is a gap in
                    measures = {"lead"}
                if measures:
                    cand = np.array([bool(measures & lt) for lt in ltoks])
                    quals = set(words) & QUALIFIERS
                    if quals and cand.any():
                        qscore = np.array([len(pt & quals) for pt in ptoks]) * cand
                        if qscore.max() > 0:
                            cand = qscore == qscore.max()
                    if cand.any() and not hits[cand].any():
                        t.status = "misattributed"
                        j = int(np.argmax(cand))
                        t.context += f"  [reads as {paths[j]} = {values[j]:.6g}]"
            prev_end = m.end()
            report.tokens.append(t)
    return report
