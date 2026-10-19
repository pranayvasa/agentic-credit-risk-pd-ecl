"""Lifetime PD term structures.

Both methods turn an account's 12-month PD into monthly hazards h_t (probability of default in
month t given survival to t-1); marginal default probability in month t is S_{t-1} * h_t.

1. Constant hazard (baseline): h_t = 1 - (1 - PD12) ** (1/12) for every month.
2. Markov: monthly transitions between DPD buckets estimated from bureau_balance (train
   customers only; decision D4). The chain gives the *shape* of default timing for a loan
   starting current. Each account's hazards are that shape scaled by a factor k chosen so the
   12-month cumulative PD equals the account's PD12. So both methods agree on year one and
   differ only in how risk is spread over the remaining life.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def constant_hazards(pd12: np.ndarray, horizon: int) -> np.ndarray:
    h = 1 - (1 - np.asarray(pd12, dtype="float64")) ** (1 / 12)
    return np.repeat(h[:, None], horizon, axis=1)


def marginal(hazards: np.ndarray) -> np.ndarray:
    surv = np.cumprod(1 - hazards, axis=1)
    prev = np.hstack([np.ones((hazards.shape[0], 1)), surv[:, :-1]])
    return prev * hazards


def transition_matrix(bb: pd.DataFrame, default_bucket: int) -> pd.DataFrame:
    """Monthly transition matrix over states 0..default_bucket-1 plus "D" (absorbing).

    ``bb`` has SK_ID_BUREAU, MONTHS_BALANCE and bucket (0-5, NaN for closed/unknown).
    Only consecutive observed months of the same credit count as a transition."""
    d = bb.dropna(subset=["bucket"]).sort_values(["SK_ID_BUREAU", "MONTHS_BALANCE"])
    state = np.minimum(d["bucket"].to_numpy(), default_bucket).astype(int)
    same = (d["SK_ID_BUREAU"].to_numpy()[1:] == d["SK_ID_BUREAU"].to_numpy()[:-1]) & (
        np.diff(d["MONTHS_BALANCE"].to_numpy()) == 1)  # fmt: skip
    k = default_bucket + 1
    counts = np.zeros((k, k))
    np.add.at(counts, (state[:-1][same], state[1:][same]), 1)
    counts[default_bucket, :] = 0
    counts[default_bucket, default_bucket] = 1  # default is absorbing
    unseen = counts.sum(axis=1) == 0
    counts[unseen, unseen] = 1  # a state never observed stays put rather than producing NaN
    P = counts / counts.sum(axis=1, keepdims=True)
    labels = [str(i) for i in range(default_bucket)] + ["D"]
    return pd.DataFrame(P, index=labels, columns=labels)


def markov_profile(P: pd.DataFrame, horizon: int) -> np.ndarray:
    """Monthly hazards of reaching default for a credit that starts current (state 0)."""
    M = P.to_numpy()
    v = np.zeros(M.shape[0])
    v[0] = 1.0
    cum = []
    for _ in range(horizon):
        v = v @ M
        cum.append(v[-1])
    cum = np.array(cum)
    prev = np.r_[0.0, cum[:-1]]
    return np.where(prev < 1, (cum - prev) / (1 - prev), 1.0)


def scaled_hazards(profile: np.ndarray, pd12: np.ndarray, iters: int = 60) -> np.ndarray:
    """Scale the Markov shape per account so that 1 - prod_{t<=12}(1 - k * lambda_t) = PD12."""
    lam12 = profile[:12]
    pd12 = np.asarray(pd12, dtype="float64")
    lo = np.zeros_like(pd12)
    hi = np.full_like(pd12, 1 / profile.max())
    for _ in range(iters):
        k = (lo + hi) / 2
        cum = 1 - np.prod(1 - k[:, None] * lam12[None, :], axis=1)
        too_low = cum < pd12
        lo, hi = np.where(too_low, k, lo), np.where(too_low, hi, k)
    k = (lo + hi) / 2
    return np.clip(k[:, None] * profile[None, :], 0, 1)
