"""Split: proportions, stratification, no ID overlap, determinism, sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acr.core.split import split_population, summary


@pytest.fixture(scope="module")
def population():
    rng = np.random.default_rng(1)
    n = 100_000
    return pd.DataFrame({"SK_ID_CURR": rng.permutation(np.arange(1, n + 1)),
                         "TARGET": (rng.uniform(size=n) < 0.081).astype(int)})  # fmt: skip


def _split(pop, seed=42, sample=None):
    return split_population(
        pop, oot_frac=0.25, test_frac_of_dev=0.25, seed=seed, sample_frac=sample
    )


def test_proportions_match_specification(population):
    s = summary(_split(population))
    assert s["proportions"]["train"] == pytest.approx(0.5625, abs=1e-4)
    assert s["proportions"]["test"] == pytest.approx(0.1875, abs=1e-4)
    assert s["proportions"]["oot"] == pytest.approx(0.25, abs=1e-4)


def test_stratified_on_target(population):
    s = summary(_split(population))
    for p in ("train", "test", "oot"):
        assert s["target_rate"][p] == pytest.approx(s["overall_target_rate"], abs=5e-4)


def test_no_id_overlap_and_full_coverage(population):
    a = _split(population)
    sets = {p: set(a.loc[a.partition == p, "SK_ID_CURR"]) for p in ("train", "test", "oot")}
    assert not sets["train"] & sets["test"]
    assert not sets["train"] & sets["oot"]
    assert not sets["test"] & sets["oot"]
    assert set().union(*sets.values()) == set(population["SK_ID_CURR"])


def test_deterministic_and_order_invariant(population):
    a = summary(_split(population))["assignment_hash"]
    b = summary(_split(population.sample(frac=1, random_state=7)))["assignment_hash"]
    assert a == b


def test_seed_changes_assignment(population):
    assert summary(_split(population, 42))["assignment_hash"] != summary(
        _split(population, 43))["assignment_hash"]  # fmt: skip


def test_sample_is_stratified_and_sized(population):
    a = _split(population, sample=0.1)
    assert len(a) == pytest.approx(10_000, abs=1)
    assert a["TARGET"].mean() == pytest.approx(population["TARGET"].mean(), abs=5e-4)
    assert set(a["SK_ID_CURR"]) <= set(population["SK_ID_CURR"])


def test_duplicate_ids_rejected(population):
    dup = pd.concat([population, population.head(1)])
    with pytest.raises(ValueError, match="duplicate"):
        _split(dup)
