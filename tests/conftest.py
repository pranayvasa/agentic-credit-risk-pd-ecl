from __future__ import annotations

import pytest

from acr.config import load_config


@pytest.fixture
def init_state(tmp_path):
    """Initial graph input with outputs redirected to a temp folder."""

    def make(llm_mode: str = "dry-run", sample_frac: float | None = 0.1) -> dict:
        cfg = load_config(llm_mode=llm_mode).model_dump(mode="json")
        cfg["base"]["paths"]["outputs_dir"] = str(tmp_path / "outputs")
        return {"config": cfg, "sample_frac": sample_frac, "llm_mode": llm_mode}

    return make
