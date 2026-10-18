"""Memory-conscious data access.

Raw CSVs are converted once into a downcast parquet cache (``data/interim``):

* floats go to float32, except amounts listed in ``schema.KEEP_FLOAT64``;
* integer-valued columns with no nulls go to the smallest integer type that fits;
* strings are dictionary-encoded (they read back as pandas categoricals).

The cache is rebuilt when the source file's size or modification time changes. Downstream
code reads only the columns it needs and filters rows by ID at read time.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from acr.core.schema import KEEP_FLOAT64

CACHE_VERSION = 1
MANIFEST = "manifest.json"


def _infer_types(path: Path, n: int = 200_000) -> dict[str, pa.DataType]:
    """Fix column types from a large head sample so later CSV blocks cannot disagree."""
    head = pd.read_csv(path, nrows=n)
    types: dict[str, pa.DataType] = {}
    for c in head.columns:
        if c.startswith("SK_ID") or c == "TARGET":
            types[c] = pa.int64()
        elif pd.api.types.is_numeric_dtype(head[c]):
            types[c] = pa.float64()
        else:
            types[c] = pa.string()
    return types


def _int_type(lo: int, hi: int) -> pa.DataType:
    for t, info in ((pa.int8(), np.iinfo(np.int8)), (pa.int16(), np.iinfo(np.int16)),
                    (pa.int32(), np.iinfo(np.int32))):  # fmt: skip
        if lo >= info.min and hi <= info.max:
            return t
    return pa.int64()


def downcast(tbl: pa.Table, keep_float64: Iterable[str] = ()) -> pa.Table:
    keep = set(keep_float64)
    cols = []
    for name, col in zip(tbl.column_names, tbl.columns, strict=True):
        t = col.type
        if pa.types.is_integer(t) or (pa.types.is_floating(t) and name not in keep):
            mm = pc.min_max(col).as_py()
            integral = (
                col.null_count == 0
                and mm["min"] is not None
                and (pa.types.is_integer(t) or pc.all(pc.equal(col, pc.floor(col))).as_py())
            )
            if integral:
                col = col.cast(_int_type(int(mm["min"]), int(mm["max"])))
            elif pa.types.is_floating(t):
                col = col.cast(pa.float32())
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            col = pc.dictionary_encode(col)
        cols.append(col)
    return pa.table(cols, names=tbl.column_names)


def _fingerprint(path: Path) -> dict:
    st = path.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "cache_version": CACHE_VERSION}


def ensure_parquet(raw_dir: Path, interim_dir: Path, tables: dict[str, str]) -> dict[str, dict]:
    """Build or reuse the parquet cache. Returns {table: {path, rows, cols, rebuilt}}."""
    interim_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = interim_dir / MANIFEST
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    out: dict[str, dict] = {}
    for name, fname in tables.items():
        src = raw_dir / fname
        if not src.exists():
            out[name] = {"path": None, "missing": str(src)}
            continue
        dst = interim_dir / f"{name}.parquet"
        fp = _fingerprint(src)
        rebuilt = False
        if not dst.exists() or manifest.get(name) != fp:
            tbl = pacsv.read_csv(
                src, convert_options=pacsv.ConvertOptions(column_types=_infer_types(src))
            )
            tbl = downcast(tbl, KEEP_FLOAT64.get(name, ()))
            pq.write_table(tbl, dst, compression="zstd")
            del tbl
            manifest[name] = fp
            manifest_path.write_text(json.dumps(manifest, indent=2))
            rebuilt = True
        meta = pq.ParquetFile(dst).metadata
        out[name] = {"path": str(dst), "rows": meta.num_rows, "cols": meta.num_columns,
                     "rebuilt": rebuilt}  # fmt: skip
    return out


def read(
    path: str | Path,
    columns: list[str] | None = None,
    ids: Iterable[int] | None = None,
    id_col: str = "SK_ID_CURR",
) -> pd.DataFrame:
    """Read selected columns, optionally keeping only rows whose ``id_col`` is in ``ids``."""
    filters = None
    if ids is not None:
        filters = pc.is_in(pc.field(id_col), pa.array(np.asarray(list(ids), dtype=np.int64)))
    tbl = pq.read_table(path, columns=columns, filters=filters)
    return tbl.to_pandas()


def column_null_counts(path: str | Path) -> dict[str, int]:
    """Null counts per column from parquet metadata, without reading the data."""
    meta = pq.ParquetFile(path).metadata
    counts: dict[str, int] = {}
    for rg in range(meta.num_row_groups):
        g = meta.row_group(rg)
        for i in range(g.num_columns):
            c = g.column(i)
            st = c.statistics
            counts[c.path_in_schema] = counts.get(c.path_in_schema, 0) + (
                st.null_count if st is not None and st.has_null_count else 0
            )
    return counts
