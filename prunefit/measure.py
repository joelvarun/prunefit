"""Measure the real thing: bytes pulled off disk and row groups touched."""

from __future__ import annotations

import datetime as dt
import io
import time

import pyarrow as pa
import pyarrow.parquet as pq

from prunefit.workload import Query, Workload


class CountingFile(io.RawIOBase):
    """Read-only file wrapper that counts every byte pyarrow actually pulls."""

    def __init__(self, path: str):
        self._fh = open(path, "rb")
        self.bytes_read = 0
        self.read_calls = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        data = self._fh.read(size)
        self.bytes_read += len(data)
        self.read_calls += 1
        return data

    def readinto(self, buffer) -> int:
        data = self._fh.read(len(buffer))
        self.bytes_read += len(data)
        self.read_calls += 1
        buffer[: len(data)] = data
        return len(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._fh.seek(offset, whence)

    def tell(self) -> int:
        return self._fh.tell()

    def close(self) -> None:
        self._fh.close()


def _norm(value):
    if isinstance(value, str):
        try:
            return dt.datetime.fromisoformat(value).replace(tzinfo=None)
        except ValueError:
            return value
    if isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    return value


def _overlaps(predicate, lo, hi) -> bool:
    lo, hi = _norm(lo), _norm(hi)
    if predicate.op == "eq":
        v = _norm(predicate.value)
        return lo <= v <= hi
    if predicate.op == "in":
        return any(lo <= _norm(v) <= hi for v in predicate.value)
    if predicate.op == "range":
        return _norm(predicate.lo) <= hi and _norm(predicate.hi) >= lo
    if predicate.op == "le":
        return _norm(predicate.hi if predicate.hi is not None else predicate.value) >= lo
    return _norm(predicate.lo if predicate.lo is not None else predicate.value) <= hi


def row_groups_scanned(path: str, query: Query) -> tuple[int, int, int]:
    """Groups and rows a statistics-pruning reader must still touch."""
    meta = pq.ParquetFile(path).metadata
    names = [meta.schema.column(i).name for i in range(meta.num_columns)]
    groups = rows = 0
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        keep = True
        for predicate in query.predicates:
            stats = rg.column(names.index(predicate.column)).statistics
            if stats is None or not stats.has_min_max:
                continue
            if not _overlaps(predicate, stats.min, stats.max):
                keep = False
                break
        if keep:
            groups += 1
            rows += rg.num_rows
    return groups, rows, meta.num_row_groups


def read_with_count(path: str, query: Query, schema: pa.Schema) -> tuple[int, int]:
    """Run the query through pyarrow and report bytes read and rows returned."""
    counter = CountingFile(path)
    try:
        table = pq.read_table(pa.PythonFile(counter, mode="r"),
                              filters=query.expression(schema),
                              columns=query.projection)
        return counter.bytes_read, table.num_rows
    finally:
        counter.close()


def timed_read(path: str, query: Query, schema: pa.Schema) -> float:
    start = time.perf_counter()
    pq.read_table(path, filters=query.expression(schema), columns=query.projection)
    return time.perf_counter() - start


def benchmark(paths: dict[str, str], workload: Workload, schema: pa.Schema,
              repeats: int = 5) -> dict[str, list[dict]]:
    """Measure every layout on every query.

    Timings interleave the layouts inside each repeat and keep the minimum, so
    a machine that drifts over the run cannot hand one layout a free win.
    """
    out: dict[str, list[dict]] = {label: [] for label in paths}
    for label, path in paths.items():
        for q in workload.queries:
            groups, rows_scanned, total_groups = row_groups_scanned(path, q)
            read_bytes, rows = read_with_count(path, q, schema)
            out[label].append({
                "query": q.name, "template": q.template, "weight": q.weight,
                "bytes_read": read_bytes, "rows_returned": rows,
                "row_groups_scanned": groups, "row_groups_total": total_groups,
                "rows_scanned": rows_scanned, "seconds": float("inf")})
    for _ in range(repeats):
        for i, q in enumerate(workload.queries):
            for label, path in paths.items():
                elapsed = timed_read(path, q, schema)
                if elapsed < out[label][i]["seconds"]:
                    out[label][i]["seconds"] = elapsed
    return out


def column_compressed_bytes(path: str) -> dict[str, int]:
    """On-disk compressed bytes per column, summed over row groups."""
    meta = pq.ParquetFile(path).metadata
    out: dict[str, int] = {}
    for g in range(meta.num_row_groups):
        rg = meta.row_group(g)
        for c in range(meta.num_columns):
            col = rg.column(c)
            out[col.path_in_schema] = out.get(col.path_in_schema, 0) + col.total_compressed_size
    return out
