"""Exact row-group boundary selection.

A row group is scanned unless some predicate interval misses the group's
statistics. Growing a group can only widen those statistics, so the set of
queries that scan a group is monotone in the group interval: once a query
matches the run starting at block i, it matches every longer run from i. That
makes the full cost matrix computable in one accumulate pass per start block,
and the boundary choice an exact interval-partition dynamic program.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from prunefit.stats import BlockStats


@dataclass
class Layout:
    """A concrete row-group layout."""

    order: list[tuple[str, str]]
    boundaries: list[int]              # row offsets, length groups + 1
    modeled_bytes: float = 0.0
    order_name: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.boundaries) - 1

    @property
    def group_rows(self) -> list[int]:
        return [self.boundaries[i + 1] - self.boundaries[i] for i in range(self.n_groups)]


def cost_matrix(stats: BlockStats, intervals: list[list[tuple[str, int, int]]],
                weights: np.ndarray) -> np.ndarray:
    """cost[i, j] = modeled bytes scanned by the group covering blocks [i, j).

    `weights[q]` is the bytes-per-row charge for query q: its frequency weight
    times the share of row bytes its projection reads.
    """
    n = stats.n_blocks
    cost = np.full((n + 1, n + 1), np.inf, dtype=np.float64)
    rows = stats.rows.astype(np.float64)
    for i in range(n):
        runmin = {c: np.minimum.accumulate(v[i:]) for c, v in stats.mins.items()}
        runmax = {c: np.maximum.accumulate(v[i:]) for c, v in stats.maxs.items()}
        matched = np.zeros(n - i, dtype=np.float64)
        for q, preds in enumerate(intervals):
            hit = np.ones(n - i, dtype=bool)
            for col, lo, hi in preds:
                if lo > hi:                      # predicate matches no stored value
                    hit[:] = False
                    break
                np.logical_and(hit, (lo <= runmax[col]) & (hi >= runmin[col]), out=hit)
            if hit.any():
                matched += hit * weights[q]
        cost[i, i + 1:] = np.cumsum(rows[i:]) * matched
    return cost


def dp_boundaries(cost: np.ndarray, stats: BlockStats, max_groups: int,
                  min_rows: int = 1, max_rows: int | None = None) -> tuple[list[int], float]:
    """Optimal block boundaries under a group-count budget.

    Returns row offsets and the modeled bytes scanned. Exact for the cost
    model: no heuristic, no local search.
    """
    n = stats.n_blocks
    block_rows = stats.block_rows
    min_blocks = max(1, -(-min_rows // block_rows))
    max_blocks = n if max_rows is None else max(min_blocks, max_rows // block_rows)
    if max_groups * max_blocks < n:
        raise ValueError("group budget cannot cover the table; raise max_groups or max_rows")

    inf = np.inf
    best = np.full((max_groups + 1, n + 1), inf, dtype=np.float64)
    back = np.zeros((max_groups + 1, n + 1), dtype=np.int32)
    best[:, 0] = 0.0
    for k in range(1, max_groups + 1):
        prev = best[k - 1]
        for j in range(1, n + 1):
            lo = max(0, j - max_blocks)
            hi = j - min_blocks
            if hi < lo:
                continue
            candidates = prev[lo:hi + 1] + cost[lo:hi + 1, j]
            pos = int(np.argmin(candidates))
            if candidates[pos] < best[k, j]:
                best[k, j] = candidates[pos]
                back[k, j] = lo + pos
    k_best = int(np.argmin(best[:, n]))
    total = float(best[k_best, n])
    if not np.isfinite(total):
        raise ValueError("no feasible layout under the given constraints")

    cuts = [n]
    k, j = k_best, n
    while j > 0:
        j = int(back[k, j])
        k -= 1
        cuts.append(j)
    cuts.reverse()
    offsets = stats.row_offsets()
    return [int(offsets[c]) for c in cuts], total


def uniform_boundaries(n_rows: int, rows_per_group: int) -> list[int]:
    """The layout every Parquet writer gives you today: fixed size groups."""
    cuts = list(range(0, n_rows, rows_per_group))
    cuts.append(n_rows)
    return cuts


def evaluate_boundaries(boundaries: list[int], stats: BlockStats,
                        cost: np.ndarray) -> float:
    """Modeled bytes for an arbitrary layout, used to score the baseline."""
    offsets = list(stats.row_offsets())
    index = {off: i for i, off in enumerate(offsets)}
    total = 0.0
    for a, b in zip(boundaries, boundaries[1:]):
        if a not in index or b not in index:
            raise ValueError("boundaries must fall on block edges to be scored")
        total += cost[index[a], index[b]]
    return total
