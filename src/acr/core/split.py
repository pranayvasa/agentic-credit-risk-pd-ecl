"""Customer-level stratified split: (dev 75 : OOT 25), then dev -> (train 75 : test 25).

Final shares: train 56.25%, test 18.75%, OOT 25%. The OOT is a *pseudo*-OOT: Home Credit has
no application dates, so it is a random stratified holdout, not a temporal one.

The split runs on SK_ID_CURR before any step that learns from data. Input order does not
matter: IDs are sorted first, so the same population and seed always give the same split.
"""

from __future__ import annotations

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from acr.core.schema import ID, TARGET

PARTITIONS = ("train", "test", "oot")


def stratified_sample(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if frac >= 1.0:
        return df
    keep, _ = train_test_split(df, train_size=frac, stratify=df[TARGET], random_state=seed)
    return keep


def split_population(
    ids: pd.DataFrame,
    oot_frac: float,
    test_frac_of_dev: float,
    seed: int,
    sample_frac: float | None = None,
) -> pd.DataFrame:
    """Return ``[SK_ID_CURR, TARGET, partition]``, sorted by ID."""
    df = ids[[ID, TARGET]].sort_values(ID).reset_index(drop=True)
    if df[ID].duplicated().any():
        raise ValueError(f"duplicate {ID} in split input")
    if sample_frac is not None:
        df = stratified_sample(df, sample_frac, seed).sort_values(ID).reset_index(drop=True)
    dev, oot = train_test_split(df, test_size=oot_frac, stratify=df[TARGET], random_state=seed)
    train, test = train_test_split(
        dev, test_size=test_frac_of_dev, stratify=dev[TARGET], random_state=seed
    )
    out = pd.concat([train.assign(partition="train"), test.assign(partition="test"),
                     oot.assign(partition="oot")])  # fmt: skip
    out["partition"] = pd.Categorical(out["partition"], categories=PARTITIONS)
    return out.sort_values(ID).reset_index(drop=True)


def summary(assign: pd.DataFrame) -> dict:
    n = len(assign)
    g = assign.groupby("partition", observed=False)[TARGET]
    counts = g.size().to_dict()
    return {
        "population": n,
        "counts": {k: int(v) for k, v in counts.items()},
        "proportions": {k: round(v / n, 6) for k, v in counts.items()},
        "target_rate": {k: round(float(v), 6) for k, v in g.mean().to_dict().items()},
        "overall_target_rate": round(float(assign[TARGET].mean()), 6),
        "assignment_hash": hashlib.sha256(
            pd.util.hash_pandas_object(assign[[ID, "partition"]], index=False).values.tobytes()
        ).hexdigest()[:16],
    }
