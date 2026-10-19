"""Offline, deterministic hyperparameter tuning (``acr tune``). Not part of the graph.

Everything here is cross-validation *inside the train partition* of an existing run. The
test and OOT partitions are never read, so tuning cannot leak holdout information. Results
go to a report; a human reviews them and freezes the winners into ``config/base.yaml``.
The pipeline then runs on fixed config, as before.

Winner selection uses the one-standard-error rule: among configurations whose mean CV Gini
is within one standard error of the best, pick the simplest.
"""

from __future__ import annotations

import itertools
import json
import pickle
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from sklearn.model_selection import StratifiedKFold

from acr.config import (
    CONFIG_DIR,
    MAX_SCORECARD_FEATURES,
    AppConfig,
    check_depth,
    resolve_path,
)
from acr.core import binning as B
from acr.core import challenger as C
from acr.core.guards import require_partition
from acr.core.metrics import gini
from acr.core.scorecard import fit_scorecard
from acr.core.selection import prune, select_features


class ScorecardGrid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_n_bins: list[int]
    min_bin_size: list[float]
    max_features: list[int]
    iv_min: list[float]

    @field_validator("max_features")
    @classmethod
    def _cap(cls, v: list[int]) -> list[int]:
        if max(v) > MAX_SCORECARD_FEATURES:
            raise ValueError(
                f"max_features grid exceeds the {MAX_SCORECARD_FEATURES}-variable limit"
            )
        return v


class LGBMGrid(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_depth: list[int]

    @field_validator("max_depth")
    @classmethod
    def _depth(cls, v: list[int]) -> list[int]:
        return [check_depth(d) for d in v]

    num_leaves: list[int]
    min_child_samples: list[int]
    colsample_bytree: list[float]
    reg_lambda: list[float]


class TuningCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    folds: int
    selection_rule: str
    scorecard: ScorecardGrid
    lightgbm: LGBMGrid


def load_tuning(config_dir: Path = CONFIG_DIR) -> TuningCfg:
    return TuningCfg.model_validate(yaml.safe_load((config_dir / "tuning.yaml").read_text()))


@dataclass
class TrainData:
    features: pd.DataFrame  # train rows only
    y: pd.Series
    dictionary: pd.DataFrame
    candidates: list[str]  # candidates surviving the missingness screen
    categorical: list[str]
    source_run: str


def load_train(run_dir: Path) -> TrainData:
    sel = run_dir / "artefacts" / "selection"
    meta = json.loads((sel / "selection.json").read_text())
    with open(sel / "binning_process.pkl", "rb") as f:
        live = list(pickle.load(f).variable_names)
    a = pd.read_parquet(run_dir / "artefacts" / "split" / "assignment.parquet")
    feats = pd.read_parquet(sel / "features_all.parquet")
    train_ids = a.loc[a["partition"] == "train", "SK_ID_CURR"]
    Xtr = require_partition(feats.loc[train_ids], a, "train")
    y = a.set_index("SK_ID_CURR").loc[train_ids, "TARGET"]
    return TrainData(Xtr, y, pd.read_csv(sel / "feature_dictionary.csv"), live,
                     [c for c in live if c in meta["categorical"]], run_dir.name)  # fmt: skip


def one_se(results: pd.DataFrame, simplicity: list[str], ascending: list[bool]) -> pd.Series:
    best = results.loc[results["cv_gini_mean"].idxmax()]
    eligible = results[results["cv_gini_mean"] >= best["cv_gini_mean"] - best["cv_gini_se"]]
    return eligible.sort_values(simplicity, ascending=ascending).iloc[0]


def _folds(y: pd.Series, k: int, seed: int):
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def _summarise(rows: list[dict], keys: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    g = df.groupby(keys, sort=False)
    out = g.agg(cv_gini_mean=("val_gini", "mean"), cv_gini_std=("val_gini", "std"),
                fit_gini_mean=("fit_gini", "mean"),
                **{c: (c, "mean") for c in df.columns if c.startswith("n_")}).reset_index()  # fmt: skip
    out = out.rename(columns={"n_trees": "n_trees_mean"}) if "n_trees" in out else out
    out["cv_gini_se"] = out["cv_gini_std"] / np.sqrt(df["fold"].nunique())
    out["overfit_gap"] = out["fit_gini_mean"] - out["cv_gini_mean"]
    out.insert(
        out.columns.get_loc("cv_gini_mean") + 1, "cv_auc_mean", (out["cv_gini_mean"] + 1) / 2
    )
    return out


def tune_scorecard(td: TrainData, grid: ScorecardGrid, base: AppConfig, folds: int, seed: int,
                   log=print) -> pd.DataFrame:  # fmt: skip
    rows = []
    fcfg = base.base.features
    for k, (tr, va) in enumerate(_folds(td.y, folds, seed)):
        ids_tr, ids_va = td.features.index[tr], td.features.index[va]
        # Inside the fold: fold-train plays "train", fold-validation plays "test" (CSI screen).
        assign = pd.DataFrame({"SK_ID_CURR": np.r_[ids_tr, ids_va],
                               "TARGET": np.r_[td.y.iloc[tr], td.y.iloc[va]],
                               "partition": ["train"] * len(tr) + ["test"] * len(va)})  # fmt: skip
        Xtr = td.features.loc[ids_tr, td.candidates]
        for nb, mbs in itertools.product(grid.max_n_bins, grid.min_bin_size):
            t0 = time.perf_counter()
            bp = B.fit_binning(Xtr, td.y.iloc[tr], td.categorical, mbs, nb)
            W = B.woe(bp, td.features[td.candidates], td.categorical)
            for ivm in grid.iv_min:
                cfg = fcfg.model_copy(update={"max_n_bins": nb, "min_bin_size": mbs, "iv_min": ivm})
                # Screens (IV, CSI) depend only on the binning and iv_min; run them once.
                sel = select_features(td.features, td.dictionary, assign, cfg, binning=bp)
                iv = sel.iv.set_index("feature")["iv"]
                for mf in grid.max_features:
                    feats = prune(W.loc[ids_tr], td.y.iloc[tr], sel.screened, iv,
                                  cfg.model_copy(update={"max_features": mf}), [])  # fmt: skip
                    if not feats:
                        continue
                    sc = fit_scorecard(
                        W.loc[ids_tr], td.y.iloc[tr], feats, base.base.model.scorecard
                    )
                    rows.append({"fold": k, "max_n_bins": nb, "min_bin_size": mbs,
                                 "max_features": mf, "iv_min": ivm, "n_selected": len(feats),
                                 "features": "|".join(sorted(feats)),
                                 "fit_gini": gini(td.y.iloc[tr], sc.predict_pd(W.loc[ids_tr])),
                                 "val_gini": gini(td.y.iloc[va], sc.predict_pd(W.loc[ids_va]))})  # fmt: skip
            log(f"  scorecard fold {k} bins={nb} min_bin={mbs}: {time.perf_counter() - t0:.0f}s")
    keys = ["max_n_bins", "min_bin_size", "max_features", "iv_min"]
    res = _summarise(rows, keys)
    sets = (
        pd.DataFrame(rows)
        .groupby(keys, sort=False)["features"]
        .apply(lambda s: _mean_jaccard([set(x.split("|")) for x in s]))
        .rename("feature_jaccard")
    )
    return res.merge(sets.reset_index(), on=keys)


def _mean_jaccard(sets: list[set]) -> float:
    """Mean pairwise Jaccard similarity of the feature sets chosen in each fold (1 = identical)."""
    pairs = list(itertools.combinations(sets, 2))
    return float(np.mean([len(a & b) / len(a | b) for a, b in pairs])) if pairs else 1.0


def tune_lightgbm(td: TrainData, grid: LGBMGrid, base: AppConfig, folds: int, seed: int,
                  log=print) -> pd.DataFrame:  # fmt: skip
    rows = []
    params0 = dict(base.base.model.lightgbm)
    es = base.base.model.lightgbm_fit.early_stopping_rounds
    X = C.prepare(td.features[td.candidates], td.categorical)
    splits = _folds(td.y, folds, seed)
    combos, seen = [], set()
    for md, nl, mcs, col, lam in itertools.product(grid.max_depth, grid.num_leaves,
                                                   grid.min_child_samples, grid.colsample_bytree,
                                                   grid.reg_lambda):  # fmt: skip
        # A depth-d tree holds at most 2^d leaves; num_leaves above that builds the same model.
        key = (md, min(nl, 2**md), mcs, col, lam)
        if key not in seen:
            seen.add(key)
            combos.append((md, nl, mcs, col, lam))
    for i, (md, nl, mcs, col, lam) in enumerate(combos):
        t0 = time.perf_counter()
        p = {**params0, "max_depth": md, "num_leaves": nl, "min_child_samples": mcs,
             "colsample_bytree": col, "reg_lambda": lam}  # fmt: skip
        for k, (tr, va) in enumerate(splits):
            m = lgb.LGBMClassifier(**C._params(p, seed))
            m.fit(X.iloc[tr], td.y.iloc[tr], eval_X=X.iloc[va], eval_y=td.y.iloc[va],
                  eval_metric="auc", callbacks=[lgb.early_stopping(es, verbose=False)])  # fmt: skip
            rows.append({"fold": k, "max_depth": md, "num_leaves": nl, "min_child_samples": mcs,
                         "colsample_bytree": col, "reg_lambda": lam,
                         "n_trees": int(m.best_iteration_ or p["n_estimators"]),
                         "fit_gini": gini(td.y.iloc[tr], m.predict_proba(X.iloc[tr])[:, 1]),
                         "val_gini": gini(td.y.iloc[va], m.predict_proba(X.iloc[va])[:, 1])})  # fmt: skip
        log(f"  lightgbm {i + 1}/{len(combos)} depth={md} leaves={nl} min_child={mcs} col={col} "
            f"lambda={lam}: {time.perf_counter() - t0:.0f}s")  # fmt: skip
    res = _summarise(rows, ["max_depth", "num_leaves", "min_child_samples", "colsample_bytree",
                            "reg_lambda"])  # fmt: skip
    # Effective complexity for the 1-SE rule: a depth-d tree holds at most 2^d leaves.
    res["leaves_eff"] = np.minimum(res["num_leaves"], 2 ** res["max_depth"])
    return res


SC_SIMPLE = (["n_selected", "max_n_bins", "min_bin_size", "iv_min"], [True, True, False, False])
GB_SIMPLE = (["max_depth", "leaves_eff", "min_child_samples", "reg_lambda"],
             [True, True, False, False])  # fmt: skip


def _md(df: pd.DataFrame, n: int = 15) -> str:
    cols = [c for c in df.columns if c != "cv_gini_std"]
    t = df.sort_values("cv_gini_mean", ascending=False).head(n)[cols].round(4)
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join(
        "| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)
    )


def _by(res: pd.DataFrame, key: str) -> str:
    """Best configuration for each value of ``key`` (feature cap, or tree depth)."""
    best = res.loc[res.groupby(key)["cv_gini_mean"].idxmax()].sort_values(key)
    label = {
        "max_features": "Best configuration per feature cap",
        "max_depth": "Best configuration per max_depth",
    }[key]
    return f"{label}:\n\n{_md(best, n=len(best))}\n"


def run(source_run: Path, base: AppConfig, tcfg: TuningCfg, which: str = "both",
        log=print) -> Path:  # fmt: skip
    td = load_train(source_run)
    out = (
        resolve_path(base.base.paths.outputs_dir) / "tuning" / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    out.mkdir(parents=True)
    seed = base.base.seed
    current = {
        "scorecard": {k: getattr(base.base.features, k) for k in ScorecardGrid.model_fields},
        "lightgbm": {k: base.base.model.lightgbm[k] for k in LGBMGrid.model_fields},
    }
    best: dict = {}
    sections = [
        f"# Tuning report\n\nSource run `{td.source_run}`, config hash `{base.hash()}`, seed {seed}, "
        f"{tcfg.folds}-fold stratified CV on the **train partition only** ({len(td.y):,} rows). "
        "Test and OOT were not read.\n\nWinner rule: simplest configuration whose mean CV Gini "
        "is within one standard error of the best.\n\nLightGBM folds early-stop on their own "
        "validation slice, so LightGBM CV Gini is slightly optimistic in absolute terms but "
        "comparable across configurations. `feature_jaccard` is the mean pairwise Jaccard "
        "similarity of the scorecard feature sets chosen in each fold (1 = same features in "
        "every fold).\n"
    ]
    for name, fn, grid, simple in (
        ("scorecard", tune_scorecard, tcfg.scorecard, SC_SIMPLE),
        ("lightgbm", tune_lightgbm, tcfg.lightgbm, GB_SIMPLE),
    ):
        if which not in ("both", name):
            continue
        t0 = time.perf_counter()
        log(f"tuning {name} ...")
        res = fn(td, grid, base, tcfg.folds, seed, log)
        res.to_csv(out / f"{name}_results.csv", index=False)
        win = one_se(res, *simple)
        top = res.loc[res["cv_gini_mean"].idxmax()]
        keys = list(type(grid).model_fields)
        cur = res
        for k in keys:
            cur = cur[cur[k] == current[name][k]]
        best[name] = {k: (v.item() if hasattr(v, "item") else v) for k, v in win[keys].items()}
        sections.append(
            f"## {name}\n\n"
            f"- Configurations: {len(res)}; time {time.perf_counter() - t0:.0f}s\n"
            f"- Best mean CV Gini: {top['cv_gini_mean']:.4f} (SE {top['cv_gini_se']:.4f})\n"
            f"- **1-SE winner**: {best[name]} with CV Gini {win['cv_gini_mean']:.4f}, "
            f"overfit gap {win['overfit_gap']:.4f}\n"
            + (f"- Current config: CV Gini {cur['cv_gini_mean'].iloc[0]:.4f}, overfit gap "
               f"{cur['overfit_gap'].iloc[0]:.4f}\n" if len(cur) else "- Current config not in grid\n")
            + f"\n{_by(res, 'max_features' if name == 'scorecard' else 'max_depth')}"
            + f"\nTop 15 by mean CV Gini:\n\n{_md(res)}\n"
        )  # fmt: skip
    (out / "best_params.yaml").write_text(yaml.safe_dump({"current": current, "winner": best}))
    (out / "tuning_report.md").write_text("\n".join(sections))
    return out
