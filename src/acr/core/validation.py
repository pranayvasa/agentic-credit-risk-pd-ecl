"""Independent validation metrics. Pure functions over *scored outputs*; nothing here fits,
tunes or alters a model.

Pseudo-OOT note: train, test and OOT are random stratified draws from the same population,
so PSI and CSI are close to zero by construction. They test the plumbing, not real drift.
``stress_psi`` shows what the same metric reports under a deliberate population shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

from acr.core.metrics import calibration_slope_intercept, ece, gini, hosmer_lemeshow, ks, psi

PARTS = ("train", "test", "oot")


def perf(y: pd.Series, p: pd.Series) -> dict:
    auc = roc_auc_score(y, p)
    return {"n": int(len(y)), "defaults": int(y.sum()), "default_rate": float(y.mean()),
            "mean_pd": float(np.mean(p)), "auc": float(auc), "gini": float(2 * auc - 1),
            "ks": ks(y, p), "brier": float(brier_score_loss(y, p))}  # fmt: skip


def by_partition(scored: pd.DataFrame, pd_col: str) -> dict:
    return {
        part: perf(g["TARGET"], g[pd_col])
        for part, g in scored.groupby("partition")
        if part in PARTS
    }


def calibration(y: pd.Series, p: pd.Series) -> dict:
    hl, hl_p = hosmer_lemeshow(y, p)
    slope, intercept = calibration_slope_intercept(y, p)
    return {"hl_stat": hl, "hl_p_value": hl_p, "ece": ece(y, p), "slope": slope,
            "intercept": intercept, "pd_to_dr": float(np.mean(p) / max(y.mean(), 1e-12))}  # fmt: skip


def _decile_edges(train_score: pd.Series) -> np.ndarray:
    edges = np.unique(np.quantile(train_score, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def score_psi(scored: pd.DataFrame, col: str = "score") -> dict:
    """PSI of the score distribution, with deciles defined on train."""
    edges = _decile_edges(scored.loc[scored["partition"] == "train", col])
    b = pd.cut(scored[col], edges, labels=False)
    base = b[scored["partition"] == "train"]
    return {f"train_vs_{p}": psi(base, b[scored["partition"] == p]) for p in ("test", "oot")}


def grade_psi(scored: pd.DataFrame, grade_col: str) -> dict:
    base = scored.loc[scored["partition"] == "train", grade_col]
    return {f"train_vs_{p}": psi(base, scored.loc[scored["partition"] == p, grade_col])
            for p in ("test", "oot")}  # fmt: skip


def stress_psi(scored: pd.DataFrame, age_years: pd.Series, young_weight: float,
               col: str = "score") -> float:  # fmt: skip
    """Illustrative drift: weight OOT applicants under 30 by ``young_weight`` and recompute
    the score PSI against train. Shows the PSI gate firing on a shifted book."""
    edges = _decile_edges(scored.loc[scored["partition"] == "train", col])
    b = pd.cut(scored[col], edges, labels=False)
    base = b[scored["partition"] == "train"].value_counts()
    oot = scored["partition"] == "oot"
    w = np.where(age_years.reindex(scored.index)[oot] < 30, young_weight, 1.0)
    stressed = pd.Series(w, index=b[oot].index).groupby(b[oot]).sum()
    from acr.core.metrics import psi_from_counts

    return psi_from_counts(base, stressed)


def grade_table(df: pd.DataFrame, pd_col: str, grade_col: str, alpha_red: float,
                alpha_amber: float) -> pd.DataFrame:  # fmt: skip
    """Per-grade outcomes with one-sided binomial tests.

    ``p_under``: probability of at least the observed defaults if the true PD were the
    grade's mean PD. Small values mean the model under-predicts risk, which is the
    prudentially relevant direction. Traffic light: red at ``alpha_red``, amber at
    ``alpha_amber``. The test assumes independent defaults, so it is conservative in a
    real (correlated) portfolio.
    """
    g = df.groupby(grade_col).agg(n=("TARGET", "size"), defaults=("TARGET", "sum"),
                                  mean_pd=(pd_col, "mean")).reset_index()  # fmt: skip
    g = g.rename(columns={grade_col: "grade"})
    g["share"] = g["n"] / g["n"].sum()
    g["default_rate"] = g["defaults"] / g["n"]
    g["p_under"] = [
        float(stats.binom.sf(d - 1, n, p))
        for d, n, p in zip(g["defaults"], g["n"], g["mean_pd"], strict=True)
    ]
    g["p_over"] = [
        float(stats.binom.cdf(d, n, p))
        for d, n, p in zip(g["defaults"], g["n"], g["mean_pd"], strict=True)
    ]
    g["traffic_light"] = np.select([g["p_under"] <= alpha_red, g["p_under"] <= alpha_amber],
                                   ["red", "amber"], "green")  # fmt: skip
    return g


def rank_order_breaks(default_rates: pd.Series, n: pd.Series, min_n: int = 100) -> int:
    """Number of adjacent pairs (worse grade/decile, enough observations) where the default
    rate falls instead of rising."""
    dr = default_rates[n >= min_n].to_numpy()
    return int(np.sum(np.diff(dr) < 0))


def decile_table(df: pd.DataFrame, pd_col: str) -> pd.DataFrame:
    t = df.assign(decile=pd.qcut(df[pd_col].rank(method="first"), 10, labels=False) + 1)
    return t.groupby("decile").agg(n=("TARGET", "size"), defaults=("TARGET", "sum"),
                                   mean_pd=(pd_col, "mean"), default_rate=("TARGET", "mean")).reset_index()  # fmt: skip


def csi_table(bins: pd.DataFrame, partition: pd.Series, features: list[str]) -> pd.DataFrame:
    """CSI per feature against train, for test, OOT and (when present) the unlabelled
    external population (rows whose partition is ``"external"``)."""
    base = bins.loc[partition == "train"]
    targets = [p for p in ("test", "oot", "external") if (partition == p).any()]
    rows = [{"feature": f, **{f"csi_{p}": psi(base[f], bins.loc[partition == p, f]) for p in targets}}
            for f in features]  # fmt: skip
    return pd.DataFrame(rows)


def external_psi(train: pd.Series, external: pd.Series) -> float:
    """PSI of an unlabelled population against train, over train-defined deciles."""
    edges = _decile_edges(train)
    return psi(pd.cut(train, edges, labels=False), pd.cut(external, edges, labels=False))


def segments(df: pd.DataFrame, seg: pd.DataFrame, pd_col: str) -> pd.DataFrame:
    rows = []
    for var in seg.columns:
        for val, g in df.groupby(seg[var].reindex(df.index).astype("string").fillna("missing")):
            y, p = g["TARGET"], g[pd_col]
            rows.append({
                "segment": var, "value": str(val), "n": int(len(g)), "default_rate": float(y.mean()),
                "mean_pd": float(p.mean()), "pd_to_dr": float(p.mean() / y.mean()) if y.mean() > 0 else np.nan,
                "gini": gini(y, p) if 0 < y.sum() < len(y) else np.nan,
                "auc": (gini(y, p) + 1) / 2 if 0 < y.sum() < len(y) else np.nan,
            })  # fmt: skip
    return pd.DataFrame(rows)


def bootstrap_gini_diff(y: pd.Series, p_a: pd.Series, p_b: pd.Series, n: int, seed: int) -> dict:
    """Gini(b) - Gini(a) with a percentile bootstrap 95% interval (deterministic seed)."""
    rng = np.random.default_rng(seed)
    y, a, b = y.to_numpy(), p_a.to_numpy(), p_b.to_numpy()
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(gini(y[i], b[i]) - gini(y[i], a[i]))
    return {"diff": gini(y, b) - gini(y, a), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "n_boot": n}  # fmt: skip


def points_range_share(points: pd.DataFrame, features: list[str]) -> float:
    """Share of the scorecard's total points range contributed by ``features``."""
    rng = points.groupby("feature")["points"].agg(lambda s: s.max() - s.min())
    return float(rng.reindex(features).fillna(0).sum() / rng.sum()) if rng.sum() else 0.0
