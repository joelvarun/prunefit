"""Search over row orders, with the exact boundary program inside each one."""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from prunefit.encode import encode_table
from prunefit.keys import order_index
from prunefit.layout import (Layout, cost_matrix, dp_boundaries, evaluate_boundaries,
                             uniform_boundaries)
from prunefit.stats import block_stats
from prunefit.workload import Workload


@dataclass
class FitConfig:
    block_rows: int = 1024
    rows_per_group: int = 8192       # baseline group size, and the group budget
    min_rows: int = 1024
    max_rows: int | None = None
    max_key_candidates: int = 4
    max_key_depth: int = 3
    time_buckets: tuple[str, ...] = ("1d", "7d", "30d")
    time_column: str = "ts"
    column_bytes: dict[str, int] | None = None


@dataclass
class FitResult:
    layout: Layout
    baseline_modeled: float
    arrival_dp_modeled: float
    candidates: list[dict] = field(default_factory=list)
    max_groups: int = 0
    block_rows: int = 0
    seconds: float = 0.0

    @property
    def modeled_reduction(self) -> float:
        if self.baseline_modeled <= 0:
            return 0.0
        return 1.0 - self.layout.modeled_bytes / self.baseline_modeled


def candidate_orders(workload: Workload, cfg: FitConfig) -> list[tuple[str, list[str]]]:
    """Arrival order plus every short sort-key permutation over hot predicate columns.

    Arrival order is always a candidate, so the fitted layout can never model
    worse than what a conventional writer produces.
    """
    hot = workload.predicate_columns[: cfg.max_key_candidates]
    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, list[str]]] = []

    def add(keys: tuple[str, ...]) -> None:
        if keys in seen:
            return
        seen.add(keys)
        out.append((",".join(keys), list(keys)))

    for depth in range(1, cfg.max_key_depth + 1):
        for keys in itertools.permutations(hot, depth):
            add(keys)

    # Hierarchical orders: coarse time bucket, then value clustering, then time.
    ts = cfg.time_column
    if ts in workload.predicate_columns:
        inner = [c for c in hot if c != ts][: max(1, cfg.max_key_candidates - 1)]
        for bucket in cfg.time_buckets:
            for depth in range(1, min(2, cfg.max_key_depth) + 1):
                for keys in itertools.permutations(inner, depth):
                    add((f"{ts}~{bucket}", *keys, ts))
    return out


def query_weights(workload: Workload, table: pa.Table, cfg: FitConfig) -> np.ndarray:
    """Bytes-per-row charge for each query: frequency times projected byte share."""
    col_bytes = cfg.column_bytes or {
        name: table.column(name).nbytes for name in table.column_names}
    n_rows = max(1, table.num_rows)
    weights = np.zeros(len(workload), dtype=np.float64)
    for i, q in enumerate(workload.queries):
        cols = q.projection or table.column_names
        projected = sum(col_bytes.get(c, 0) for c in cols)
        weights[i] = q.weight * projected / n_rows
    return weights


def fit(table: pa.Table, workload: Workload, cfg: FitConfig | None = None) -> FitResult:
    """Pick the row order and the row-group boundaries for this query log."""
    cfg = cfg or FitConfig()
    if cfg.rows_per_group % cfg.block_rows:
        raise ValueError("rows_per_group must be a multiple of block_rows so the "
                         "baseline layout lands on block edges")
    started = time.perf_counter()
    pred_cols = workload.predicate_columns
    codes, codecs = encode_table(table, pred_cols)
    intervals = [q.code_intervals(codecs) for q in workload.queries]
    weights = query_weights(workload, table, cfg)
    max_groups = math.ceil(table.num_rows / cfg.rows_per_group)

    best: Layout | None = None
    arrival_dp = math.inf
    baseline = math.inf
    rows: list[dict] = []

    for name, keys in [("arrival", ["ts"] if "ts" in codes else pred_cols[:1])] + \
            candidate_orders(workload, cfg):
        idx = order_index(codes, keys)
        stats = block_stats({c: v[idx] for c, v in codes.items()}, cfg.block_rows)
        cost = cost_matrix(stats, intervals, weights)
        boundaries, modeled = dp_boundaries(cost, stats, max_groups,
                                            cfg.min_rows, cfg.max_rows)
        if name == "arrival":
            arrival_dp = modeled
            baseline = evaluate_boundaries(
                uniform_boundaries(table.num_rows, cfg.rows_per_group), stats, cost)
        rows.append({"order": name, "keys": keys, "modeled_bytes": modeled,
                     "groups": len(boundaries) - 1})
        if best is None or modeled < best.modeled_bytes:
            best = Layout(order=keys, boundaries=boundaries, modeled_bytes=modeled,
                          order_name=name)

    assert best is not None
    return FitResult(layout=best, baseline_modeled=baseline, arrival_dp_modeled=arrival_dp,
                     candidates=sorted(rows, key=lambda r: r["modeled_bytes"]),
                     max_groups=max_groups, block_rows=cfg.block_rows,
                     seconds=time.perf_counter() - started)
