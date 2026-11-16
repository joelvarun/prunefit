"""Map every predicate column onto int64 codes.

Parquet prunes a row group when a predicate interval misses the group's
[min, max] statistics. That test is identical for integers, timestamps and
strings once each column is replaced by order preserving codes, so the rest of
the package only ever deals with integer intervals.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

NEG_INF = np.iinfo(np.int64).min
POS_INF = np.iinfo(np.int64).max


@dataclass
class ColumnCodec:
    """Order preserving encoder for one column."""

    name: str
    kind: str
    dictionary: list | None = None

    def code_bounds(self, value) -> tuple[int, int]:
        """Return the inclusive code interval that `value` occupies."""
        if self.kind == "string":
            assert self.dictionary is not None
            lo = bisect.bisect_left(self.dictionary, value)
            hi = bisect.bisect_right(self.dictionary, value) - 1
            if lo > hi:
                # Absent value: it still sits between two present codes, and a
                # group can only hold it if that gap is inside the group range.
                return lo, lo - 1
            return lo, hi
        return _numeric_code(value), _numeric_code(value)

    def lower(self, value) -> int:
        return self.code_bounds(value)[0]

    def upper(self, value) -> int:
        lo, hi = self.code_bounds(value)
        return hi if hi >= lo else lo - 1


def _numeric_code(value) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if hasattr(value, "timestamp"):
        micros = int(value.timestamp() * 1_000_000)
        return micros
    if isinstance(value, str):
        import datetime as dt

        stamp = dt.datetime.fromisoformat(value)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=dt.timezone.utc)
        return int(stamp.timestamp() * 1_000_000)
    raise TypeError(f"cannot encode value of type {type(value)!r}")


def encode_table(table: pa.Table, columns: list[str]) -> tuple[dict[str, np.ndarray], dict[str, ColumnCodec]]:
    """Encode the requested columns of `table` into int64 code arrays."""
    codes: dict[str, np.ndarray] = {}
    codecs: dict[str, ColumnCodec] = {}
    for name in columns:
        col = table.column(name)
        kind = _kind_of(col.type)
        if kind == "string":
            values = col.to_pylist()
            dictionary = sorted({v for v in values if v is not None})
            index = {v: i for i, v in enumerate(dictionary)}
            codes[name] = np.fromiter(
                (index.get(v, -1) for v in values), dtype=np.int64, count=len(values))
            codecs[name] = ColumnCodec(name, "string", dictionary)
        else:
            arr = col.combine_chunks()
            if pa.types.is_timestamp(col.type):
                arr = arr.cast(pa.int64())
            codes[name] = np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.int64)
            codecs[name] = ColumnCodec(name, kind)
    return codes, codecs


def _kind_of(dtype: pa.DataType) -> str:
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return "string"
    if pa.types.is_timestamp(dtype):
        return "timestamp"
    if pa.types.is_integer(dtype):
        return "integer"
    if pa.types.is_floating(dtype):
        return "float"
    raise TypeError(f"unsupported predicate column type {dtype}")
