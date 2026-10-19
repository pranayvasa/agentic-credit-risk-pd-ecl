"""ECL computation per account, per scenario and lifetime method.

For performing loans (Stages 1 and 2):
    ECL = LGD * sum_{t=1..H} marginal_PD_t * EAD_t * DF_t,    DF_t = (1 + EIR) ** -t
with horizon H = min(12, term) for Stage 1 (12-month ECL) and H = term for Stage 2 (lifetime).
For Stage 3 (credit-impaired): PD = 1, ECL = LGD * EAD_0.

The engine works at LGD = 1 ("unit ECL"); ECL at any LGD is unit ECL * LGD, which makes the
LGD sensitivity exact. Forward-looking scenarios scale the 12-month PD before the term
structure is built; staging is held fixed across scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from acr.core.ecl.lifetime import constant_hazards, marginal, scaled_hazards


@dataclass
class Book:
    stage: np.ndarray
    pd12: np.ndarray
    ead: np.ndarray  # (accounts, horizon) exposure at start of each month
    rate: np.ndarray  # monthly EIR
    term: np.ndarray  # months
    pd_floor: float

    @property
    def ead0(self) -> np.ndarray:
        return self.ead[:, 0]


def unit_ecl(book: Book, pd_scalar: float, method: str, profile: np.ndarray | None) -> dict:
    """ECL at LGD = 1 and lifetime PD, per account, for one scenario and method."""
    H = book.ead.shape[1]
    pd12 = np.clip(book.pd12 * pd_scalar, book.pd_floor, 0.9999)
    hz = (
        constant_hazards(pd12, H)
        if method == "constant_hazard"
        else scaled_hazards(profile[:H], pd12)
    )
    m = marginal(hz)
    t = np.arange(1, H + 1)[None, :]
    horizon = np.where(book.stage == 1, np.minimum(12, book.term), book.term)[:, None]
    m = np.where(t <= horizon, m, 0.0)
    df = (1 + book.rate[:, None]) ** -t
    unit = (m * book.ead * df).sum(axis=1)
    pd_h = m.sum(axis=1)
    s3 = book.stage == 3
    unit = np.where(s3, book.ead0, unit)
    pd_h = np.where(s3, 1.0, pd_h)
    return {"unit_ecl": unit, "pd_horizon": pd_h, "pd12_used": np.where(s3, 1.0, pd12)}


def floors_for(stage: np.ndarray, floors: dict) -> np.ndarray:
    """Stage-wise minimum provision rates (para 82(3)(x), unsecured retail). Every Stage 3
    account in this simulated book is in its first year of Stage 3."""
    return np.select([stage == 1, stage == 2, stage == 3],
                     [floors["stage1"], floors["stage2"], floors["stage3_0_1y"]])  # fmt: skip
