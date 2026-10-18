"""Parquet cache: downcasting preserves values and the cache is reused or rebuilt correctly."""

from __future__ import annotations

import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from acr.core.io import downcast, ensure_parquet, read
from synthetic import make_tables, write_raw


def test_downcast_types_and_values():
    tbl = pa.table({
        "SK_ID_CURR": pa.array([100001, 100002], pa.int64()),
        "flag": pa.array([0.0, 1.0]),
        "amt": pa.array([1.5, None]),
        "kept": pa.array([123456.75, 2.5]),
        "cat": pa.array(["a", "b"]),
    })  # fmt: skip
    out = downcast(tbl, keep_float64={"kept"})
    assert out.schema.field("SK_ID_CURR").type == pa.int32()
    assert out.schema.field("flag").type == pa.int8()
    assert out.schema.field("amt").type == pa.float32()
    assert out.schema.field("kept").type == pa.float64()
    assert pa.types.is_dictionary(out.schema.field("cat").type)
    assert out.column("kept").to_pylist() == [123456.75, 2.5]


def test_cache_reused_then_rebuilt_on_change(tmp_path):
    raw = write_raw(tmp_path / "raw", make_tables(n=300))
    tables = {"application": "application_train.csv"}
    first = ensure_parquet(raw, tmp_path / "interim", tables)
    second = ensure_parquet(raw, tmp_path / "interim", tables)
    assert first["application"]["rebuilt"] and not second["application"]["rebuilt"]
    src = raw / "application_train.csv"
    os.utime(src, ns=(src.stat().st_atime_ns, src.stat().st_mtime_ns + 10**9))
    assert ensure_parquet(raw, tmp_path / "interim", tables)["application"]["rebuilt"]


def test_read_filters_ids_and_columns(tmp_path):
    raw = write_raw(tmp_path / "raw", make_tables(n=300))
    info = ensure_parquet(raw, tmp_path / "interim", {"application": "application_train.csv"})
    df = read(info["application"]["path"], ["SK_ID_CURR", "TARGET"], ids=[100001, 100005])
    assert list(df.columns) == ["SK_ID_CURR", "TARGET"]
    assert sorted(df["SK_ID_CURR"]) == [100001, 100005]
    assert pq.ParquetFile(info["application"]["path"]).metadata.num_rows == 300
    assert isinstance(df, pd.DataFrame)
