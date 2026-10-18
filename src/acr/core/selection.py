"""Feature selection for the champion scorecard. Every learned step uses train only.

Stages, in order. Each exclusion is logged with the stage, reason and metric value:

1. role       protected characteristics are segment-only (never candidates)
2. missing    missing share on train above ``missing_max``
3. binning    optimal binning yields a single bin (no discrimination)
4. iv         IV below ``iv_min``, or above ``iv_suspicious`` (possible leakage)
5. csi        train-vs-test characteristic stability above ``csi_max`` (uses test *features* only)
6. corr       |Pearson correlation of WoE| above ``corr_max`` with a higher-IV kept feature
7. vif        iterative removal of the highest VIF above ``vif_max``
8. cap        beyond the top ``max_features`` by IV
9. model      backward elimination: wrong coefficient sign, then p-value above ``p_value_max``
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from acr.config import FeatureCfg
from acr.core import binning as B
from acr.core.guards import require_partition


@dataclass
class SelectionResult:
    candidates: list[str]
    categorical: list[str]
    selected: list[str]
    exclusions: pd.DataFrame
    iv: pd.DataFrame
    csi: dict[str, float]
    binning: object  # fitted optbinning BinningProcess (train only)
    screened: list[str] = field(default_factory=list)  # survivors of stages 1-5
    vif: dict[str, float] = field(default_factory=dict)


def _vif(W: pd.DataFrame) -> pd.Series:
    X = sm.add_constant(W.to_numpy(dtype="float64"), has_constant="add")
    return pd.Series(
        [variance_inflation_factor(X, i + 1) for i in range(W.shape[1])], index=W.columns
    )


def logit_fit(W: pd.DataFrame, y: pd.Series):
    return sm.Logit(y.to_numpy(dtype="float64"), sm.add_constant(W.to_numpy(dtype="float64"),
                    has_constant="add")).fit(disp=0, maxiter=200)  # fmt: skip


def backward_eliminate(W: pd.DataFrame, y: pd.Series, p_max: float,
                       log: list[dict], stage: str = "model") -> list[str]:  # fmt: skip
    """Drop wrong-sign coefficients first (WoE coefficients must be negative), then the least
    significant feature, one at a time, refitting after every drop."""
    feats = list(W.columns)
    while feats:
        res = logit_fit(W[feats], y)
        coefs = pd.Series(res.params[1:], index=feats)
        pvals = pd.Series(res.pvalues[1:], index=feats)
        wrong = coefs[coefs >= 0]
        if len(wrong):
            f = wrong.idxmax()
            log.append({"feature": f, "stage": stage, "reason": "wrong coefficient sign "
                        "(positive on WoE)", "value": round(float(coefs[f]), 6)})  # fmt: skip
        elif pvals.max() > p_max:
            f = pvals.idxmax()
            log.append({"feature": f, "stage": stage, "reason": f"p-value above {p_max}",
                        "value": round(float(pvals[f]), 6)})  # fmt: skip
        else:
            return feats
        feats.remove(f)
    return feats


def select_features(features: pd.DataFrame, dictionary: pd.DataFrame, assignment: pd.DataFrame,
                    cfg: FeatureCfg, binning=None) -> SelectionResult:  # fmt: skip
    """``features`` is indexed by SK_ID_CURR and covers all partitions; ``assignment`` holds
    SK_ID_CURR, TARGET and partition. Targets are read for train only.

    ``binning`` lets offline tuning reuse a BinningProcess already fitted on the same train
    rows with the same bin settings; pipeline runs always fit it here."""
    a = assignment.set_index("SK_ID_CURR")
    train_ids = a.index[a["partition"] == "train"]
    test_ids = a.index[a["partition"] == "test"]
    Xtr = require_partition(features.loc[train_ids], assignment, "train")
    ytr = a.loc[train_ids, "TARGET"]
    log: list[dict] = []

    d = dictionary.set_index("name")
    for f in d.index[d["role"] != "candidate"]:
        log.append({"feature": f, "stage": "role", "reason": f"role is {d.loc[f, 'role']} "
                    "(protected characteristic)", "value": None})  # fmt: skip
    cands = [f for f in d.index[d["role"] == "candidate"] if f in features.columns]
    categorical = [f for f in cands if d.loc[f, "kind"] == "categorical"]

    miss = Xtr[cands].isna().mean()
    for f in miss.index[miss > cfg.missing_max]:
        log.append({"feature": f, "stage": "missing", "reason": f"missing share above "
                    f"{cfg.missing_max}", "value": round(float(miss[f]), 4)})  # fmt: skip
    live = [f for f in cands if miss[f] <= cfg.missing_max]
    if binning is not None:
        live = list(binning.variable_names)
    cat_live = [f for f in live if f in categorical]

    bp = binning or B.fit_binning(Xtr[live], ytr, cat_live, cfg.min_bin_size, cfg.max_n_bins)
    iv = B.iv_table(bp, live).set_index("feature")
    keep = []
    for f in live:
        if iv.loc[f, "n_bins"] < 2:
            log.append({"feature": f, "stage": "binning", "reason": "single bin after optimal "
                        "binning", "value": int(iv.loc[f, "n_bins"])})  # fmt: skip
        elif iv.loc[f, "iv"] < cfg.iv_min:
            log.append({"feature": f, "stage": "iv", "reason": f"IV below {cfg.iv_min}",
                        "value": round(float(iv.loc[f, "iv"]), 6)})  # fmt: skip
        elif iv.loc[f, "iv"] > cfg.iv_suspicious:
            log.append({"feature": f, "stage": "iv", "reason": f"IV above {cfg.iv_suspicious}: "
                        "possible leakage", "value": round(float(iv.loc[f, "iv"]), 6)})  # fmt: skip
        else:
            keep.append(f)

    cat = [f for f in live if f in categorical]
    bins_tr = B.bins(bp, Xtr[live], cat)
    bins_te = B.bins(bp, features.loc[test_ids, live], cat)
    csi = B.csi(bins_tr, bins_te)
    for f in list(keep):
        if csi[f] > cfg.csi_max:
            log.append({"feature": f, "stage": "csi", "reason": f"train-vs-test CSI above "
                        f"{cfg.csi_max}", "value": round(csi[f], 6)})  # fmt: skip
            keep.remove(f)

    Wtr = B.woe(bp, Xtr[live], cat)
    screened = list(keep)
    selected = prune(Wtr, ytr, screened, iv["iv"], cfg, log)
    final_vif = _vif(Wtr[selected]).round(4).to_dict() if len(selected) > 1 else {}

    exclusions = pd.DataFrame(log, columns=["feature", "stage", "reason", "value"])
    exclusions["leg"] = exclusions["feature"].map(d["leg"])
    return SelectionResult(cands, categorical, selected, exclusions, iv.reset_index(), csi, bp,
                           screened, final_vif)  # fmt: skip


def prune(Wtr: pd.DataFrame, ytr: pd.Series, features: list[str], iv: pd.Series, cfg: FeatureCfg,
          log: list[dict]) -> list[str]:  # fmt: skip
    """Stages 6-9 (correlation, VIF, cap, backward elimination) on WoE-transformed train."""
    keep = sorted(features, key=lambda f: -iv[f])
    corr = Wtr[keep].corr().abs()
    kept: list[str] = []
    for f in keep:
        clash = [(g, corr.loc[f, g]) for g in kept if corr.loc[f, g] > cfg.corr_max]
        if clash:
            g, r = max(clash, key=lambda t: t[1])
            log.append({"feature": f, "stage": "corr", "reason": f"|corr| with {g} above "
                        f"{cfg.corr_max}", "value": round(float(r), 4)})  # fmt: skip
        else:
            kept.append(f)

    vif = _vif(Wtr[kept]) if len(kept) > 1 else pd.Series(dtype=float)
    while len(kept) > 1 and vif.max() > cfg.vif_max:
        f = vif.idxmax()
        log.append({"feature": f, "stage": "vif", "reason": f"VIF above {cfg.vif_max}",
                    "value": round(float(vif[f]), 4)})  # fmt: skip
        kept.remove(f)
        vif = _vif(Wtr[kept])

    for f in kept[cfg.max_features :]:
        log.append({"feature": f, "stage": "cap", "reason": f"beyond top {cfg.max_features} by IV",
                    "value": round(float(iv[f]), 6)})  # fmt: skip
    kept = kept[: cfg.max_features]
    return backward_eliminate(Wtr[kept], ytr, cfg.p_value_max, log)


def woe_frame(
    sel: SelectionResult, features: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    """WoE-transform any rows with the train-fitted binning."""
    live = list(sel.binning.variable_names)
    W = B.woe(sel.binning, features[live], [f for f in live if f in sel.categorical])
    return W[cols] if cols is not None else W
