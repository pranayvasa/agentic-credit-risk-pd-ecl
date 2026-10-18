"""Config validation and hashing."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from acr.config import AppConfig, load_config


def test_config_loads_and_hash_is_stable():
    assert load_config().hash() == load_config().hash()


def test_default_llm_mode_is_dry_run():
    assert load_config().llm.mode == "dry-run"


def test_rbi_parameters_match_verified_values():
    ecl = load_config().ecl
    assert ecl.regulation["reference"] == "RBI/DOR/2026-27/398"
    assert ecl.pd["floor_12m"] == 0.0003  # para 96
    assert ecl.lgd["base"] == 0.70  # para 97, unsecured
    assert ecl.floors["stage1"] == 0.01  # para 82(3)(x)
    assert ecl.floors["stage2"] == 0.05


def test_scenario_weights_must_sum_to_one():
    d = load_config().model_dump(mode="json")
    d["ecl"]["scenarios"]["base"]["weight"] = 0.9
    with pytest.raises(ValidationError, match="sum to 1"):
        AppConfig.model_validate(d)


def test_unknown_config_keys_are_rejected():
    d = load_config().model_dump(mode="json")
    d["base"]["surprise"] = 1
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)


def test_critic_bound_cannot_exceed_two():
    d = load_config().model_dump(mode="json")
    d["llm"]["max_critic_iterations"] = 5
    with pytest.raises(ValidationError):
        AppConfig.model_validate(d)
