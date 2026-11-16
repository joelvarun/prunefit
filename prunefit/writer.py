"""Write a Parquet file whose row groups are exactly the chosen groups."""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from prunefit.encode import encode_table
from prunefit.keys import key_columns, order_index
from prunefit.layout import Layout, uniform_boundaries


def _write(ordered: pa.Table, boundaries: list[int], out_path: str,
           compression: str) -> dict:
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with pq.ParquetWriter(out_path, ordered.schema, compression=compression,
                          write_statistics=True) as writer:
        for start, end in zip(boundaries, boundaries[1:]):
            chunk = ordered.slice(start, end - start)
            writer.write_table(chunk, row_group_size=end - start)
    meta = pq.ParquetFile(out_path).metadata
    return {"path": out_path, "bytes": os.path.getsize(out_path),
            "row_groups": meta.num_row_groups, "rows": meta.num_rows}


def write_layout(table: pa.Table, layout: Layout, out_path: str,
                 compression: str = "snappy") -> dict:
    """Materialize `layout`: sort by its keys, then cut at its boundaries."""
    codes, _ = encode_table(table, key_columns(layout.order))
    idx = order_index(codes, layout.order)
    ordered = table.take(pa.array(idx))
    return _write(ordered, layout.boundaries, out_path, compression)


def write_baseline(table: pa.Table, out_path: str, rows_per_group: int = 8192,
                   sort_keys: list[str] | None = None,
                   compression: str = "snappy") -> dict:
    """What a conventional pipeline produces: arrival order, fixed size groups."""
    sort_keys = sort_keys or (["ts"] if "ts" in table.column_names else [])
    if sort_keys:
        codes, _ = encode_table(table, key_columns(sort_keys))
        idx = order_index(codes, sort_keys)
        table = table.take(pa.array(idx))
    boundaries = uniform_boundaries(table.num_rows, rows_per_group)
    return _write(table, boundaries, out_path, compression)
