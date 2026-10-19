"""Annuity maths: implied rates, contractual term and amortising exposure.

Annuity identity for a loan of principal P, monthly instalment A, monthly rate r, n months:
    A = P * r / (1 - (1 + r) ** -n)
Given (P, A, n) the rate solves by bisection (A is increasing in r). Given (P, A, r):
    n = -ln(1 - r * P / A) / ln(1 + r)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

R_MAX = 0.5  # monthly rate search ceiling (600% a year); far above anything in the data


def annuity(P, r, n):
    P, r, n = (np.asarray(x, dtype="float64") for x in (P, r, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(r > 0, P * r / (1 - (1 + r) ** -n), P / n)


def solve_rate(P, A, n, iters: int = 80) -> np.ndarray:
    """Monthly rate implied by principal, instalment and term (vectorised bisection).
    Loans with A * n <= P (no interest) get rate 0."""
    P, A, n = (np.asarray(x, dtype="float64") for x in (P, A, n))
    lo, hi = np.zeros_like(P), np.full_like(P, R_MAX)
    for _ in range(iters):
        mid = (lo + hi) / 2
        too_low = annuity(P, mid, n) < A
        lo, hi = np.where(too_low, mid, lo), np.where(too_low, hi, mid)
    r = (lo + hi) / 2
    return np.where(A * n <= P, 0.0, r)


def implied_rates(prev: pd.DataFrame) -> dict[str, float]:
    """Median implied monthly rate per contract type, from approved prior loans with a known
    term. ``prev`` needs NAME_CONTRACT_TYPE, NAME_CONTRACT_STATUS, AMT_CREDIT, AMT_ANNUITY,
    CNT_PAYMENT."""
    ok = ((prev["NAME_CONTRACT_STATUS"].astype("string") == "Approved")
          & (prev["CNT_PAYMENT"] > 0) & (prev["AMT_ANNUITY"] > 0) & (prev["AMT_CREDIT"] > 0))  # fmt: skip
    p = prev[ok]
    r = solve_rate(p["AMT_CREDIT"], p["AMT_ANNUITY"], p["CNT_PAYMENT"])
    s = pd.Series(r, index=p.index).groupby(p["NAME_CONTRACT_TYPE"].astype("string")).median()
    return {str(k): float(v) for k, v in s.items()}


def term_months(P, A, r, min_months: int, max_months: int) -> np.ndarray:
    """Contractual term solved from the annuity identity at rate r, rounded up and clipped.
    If the instalment does not cover interest (r * P >= A) the term is set to the maximum."""
    P, A, r = (np.asarray(x, dtype="float64") for x in (P, A, r))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = 1 - r * P / A
        n = np.where(r > 0, -np.log(x) / np.log1p(r), P / A)
    n = np.where((x <= 0) | ~np.isfinite(n), max_months, np.ceil(n - 1e-9))
    return np.clip(n, min_months, max_months).astype(int)


def balance_schedule(P, A, r, n, horizon: int) -> np.ndarray:
    """Outstanding balance at the *start* of months 1..horizon (exposure if default occurs in
    that month). Shape (accounts, horizon). Zero after the final instalment."""
    P, A, r = (np.asarray(x, dtype="float64")[:, None] for x in (P, A, r))
    n = np.asarray(n)[:, None]
    t = np.arange(horizon)[None, :]  # months already paid before month t+1
    growth = (1 + r) ** t
    with np.errstate(divide="ignore", invalid="ignore"):
        paid = np.where(r > 0, A * (growth - 1) / r, A * t)
    bal = np.clip(P * growth - paid, 0, None)
    return np.where(t < n, bal, 0.0)
