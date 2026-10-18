"""Leg agents: shape, dictionaries, hand-checked calculations and leakage controls."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from acr.config import load_config
from acr.core.io import ensure_parquet
from acr.core.legs import application, behavioural, bureau, previous
from acr.core.legs.common import leg_quality
from synthetic import make_tables, write_raw

LEGS = {"application": application, "bureau": bureau, "behavioural": behavioural,
        "previous": previous}  # fmt: skip


def _paths(tmp_path, tables):
    raw = write_raw(tmp_path / "raw", tables)
    loaded = ensure_parquet(raw, tmp_path / "interim", load_config().base.data.tables)
    return {k: v["path"] for k, v in loaded.items()}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("legs")
    tables = make_tables(n=1200)
    paths = _paths(tmp, tables)
    ids = tables["application"]["SK_ID_CURR"].to_numpy()
    return tables, paths, ids, {n: m.build(paths, ids) for n, m in LEGS.items()}


@pytest.mark.parametrize("leg", list(LEGS))
def test_one_row_per_id_and_dictionary_matches(built, leg):
    _, _, ids, results = built
    res = results[leg]
    assert list(res.features.index) == list(ids)
    assert res.features.index.is_unique
    assert list(res.dictionary["name"]) == list(res.features.columns)
    assert set(res.dictionary["leg"]) == {leg}
    assert res.dictionary["definition"].str.len().min() > 5
    assert leg_quality(res)["status"] == "pass"


@pytest.mark.parametrize("leg", list(LEGS))
def test_feature_prefixes(built, leg):
    prefix = {"application": "APP_", "bureau": "BUR_", "behavioural": "BEH_", "previous": "PRV_"}
    assert all(c.startswith(prefix[leg]) for c in built[3][leg].features.columns)


@pytest.mark.parametrize("leg", list(LEGS))
def test_leg_code_never_touches_target(leg):
    # Leakage control: leg builders are not given TARGET and must not read it.
    assert "TARGET" not in inspect.getsource(LEGS[leg])


def test_features_unchanged_when_target_permuted(tmp_path, built):
    tables, _, ids, results = built
    shuffled = {k: v.copy() for k, v in tables.items()}
    rng = np.random.default_rng(99)
    shuffled["application"]["TARGET"] = rng.permutation(
        shuffled["application"]["TARGET"].to_numpy()
    )
    paths = _paths(tmp_path, shuffled)
    for name, mod in LEGS.items():
        pd.testing.assert_frame_equal(mod.build(paths, ids).features, results[name].features)


def test_builds_are_deterministic(built):
    _, paths, ids, results = built
    for name, mod in LEGS.items():
        pd.testing.assert_frame_equal(mod.build(paths, ids).features, results[name].features)


def test_subset_of_ids_gives_same_rows(built):
    # Per-customer transforms: a customer's features do not depend on who else is in the run.
    _, paths, ids, results = built
    sub = ids[::7]
    for name, mod in LEGS.items():
        pd.testing.assert_frame_equal(
            mod.build(paths, sub).features, results[name].features.loc[sub]
        )


def test_protected_attributes_are_segment_only(built):
    d = built[3]["application"].dictionary.set_index("name")
    assert d.loc["APP_CODE_GENDER", "role"] == "segment_only"
    assert d.loc["APP_NAME_FAMILY_STATUS", "role"] == "segment_only"
    assert (d.drop(["APP_CODE_GENDER", "APP_NAME_FAMILY_STATUS"])["role"] == "candidate").all()


def test_days_employed_sentinel_becomes_missing_with_flag(built):
    tables, _, _, results = built
    f = results["application"].features
    sentinel = tables["application"].set_index("SK_ID_CURR")["DAYS_EMPLOYED"] == 365243
    assert f.loc[sentinel, "APP_EMPLOYED_YEARS"].isna().all()
    assert (f.loc[sentinel, "APP_EMPLOYED_SENTINEL_FLAG"] == 1).all()
    assert (f.loc[~sentinel, "APP_EMPLOYED_SENTINEL_FLAG"] == 0).all()


def test_instalment_dpd_uses_last_part_payment():
    ins = pd.DataFrame({
        "SK_ID_PREV": [1, 1, 1], "SK_ID_CURR": [7, 7, 7],
        "NUM_INSTALMENT_VERSION": [1, 1, 1], "NUM_INSTALMENT_NUMBER": [1, 1, 2],
        "DAYS_INSTALMENT": [-100.0, -100.0, -70.0],
        "DAYS_ENTRY_PAYMENT": [-105.0, -60.0, -75.0],   # instalment 1 finished 40 days late
        "AMT_INSTALMENT": [500.0, 500.0, 500.0], "AMT_PAYMENT": [300.0, 200.0, 400.0],
    })  # fmt: skip
    il = behavioural.instalment_level(ins).sort_values("due")
    assert il["dpd"].tolist() == [40.0, 0.0]
    assert il["paid_ratio"].tolist() == [1.0, 0.8]
    assert il["shortfall"].tolist() == [0.0, 100.0]
    assert il["early"].tolist() == [0.0, 5.0]


def test_bureau_status_buckets():
    s = pd.Series(["0", "1", "2", "5", "C", "X"])
    assert bureau.dpd_buckets(s).tolist()[:4] == [0, 1, 2, 5]
    assert bureau.dpd_buckets(s).iloc[4:].isna().all()


def test_customers_without_history_get_zero_counts_and_missing_ratios(built):
    tables, _, ids, results = built
    no_bureau = sorted(set(ids) - set(tables["bureau"]["SK_ID_CURR"]))
    assert no_bureau, "synthetic data should include customers with no bureau records"
    f = results["bureau"].features.loc[no_bureau]
    assert (f["BUR_COUNT"] == 0).all()
    assert f["BUR_ACTIVE_SHARE"].isna().all()
    assert not results["bureau"].has_record.loc[no_bureau].any()
