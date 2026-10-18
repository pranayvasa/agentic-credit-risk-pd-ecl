from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from acr.config import load_config  # noqa: E402
from synthetic import make_tables, write_raw  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_raw(tmp_path_factory) -> Path:
    """Synthetic raw CSVs shared by tests that do not modify them."""
    return write_raw(tmp_path_factory.mktemp("raw"), make_tables())


@pytest.fixture
def init_state(tmp_path, synthetic_raw):
    """Initial graph input pointing at synthetic data, with interim and outputs in temp folders."""

    def make(llm_mode: str = "dry-run", sample_frac: float | None = 0.5,
             raw_dir: Path | None = None) -> dict:  # fmt: skip
        cfg = load_config(llm_mode=llm_mode).model_dump(mode="json")
        cfg["base"]["paths"]["raw_dir"] = str(raw_dir or synthetic_raw)
        cfg["base"]["paths"]["interim_dir"] = str(tmp_path / "interim")
        cfg["base"]["paths"]["outputs_dir"] = str(tmp_path / "outputs")
        return {"config": cfg, "sample_frac": sample_frac, "llm_mode": llm_mode}

    return make
