"""The boundary program must be exactly optimal, not approximately."""

from __future__ import annotations

import itertools
import random

import numpy as np

from prunefit.layout import cost_matrix, dp_boundaries, evaluate_boundaries, uniform_boundaries
from prunefit.stats import BlockStats


def naive_group_cost(stats: BlockStats, intervals, weights, i: int, j: int) -> float:
    """Reference cost for one group, computed the obvious slow way."""
    rows = float(stats.rows[i:j].sum())
    total = 0.0
    for q, preds in enumerate(intervals):
        hit = True
        for col, lo, hi in preds:
            gmin = int(stats.mins[col][i:j].min())
            gmax = int(stats.maxs[col][i:j].max())
            if lo > hi or lo > gmax or hi < gmin:
                hit = False
                break
        if hit:
            total += weights[q]
    return rows * total


def random_instance(seed: int):
    rng = random.Random(seed)
    n_blocks = rng.randint(5, 9)
    cols = ["a", "b"]
    mins, maxs = {}, {}
    for c in cols:
        lows = [rng.randint(0, 20) for _ in range(n_blocks)]
        mins[c] = np.array(lows, dtype=np.int64)
        maxs[c] = np.array([v + rng.randint(0, 6) for v in lows], dtype=np.int64)
    rows = np.array([rng.randint(1, 4) for _ in range(n_blocks)], dtype=np.int64)
    stats = BlockStats(1, rows, mins, maxs)
    intervals = []
    for _ in range(rng.randint(1, 5)):
        preds = []
        for c in cols:
            if rng.random() < 0.7:
                lo = rng.randint(0, 20)
                preds.append((c, lo, lo + rng.randint(0, 5)))
        intervals.append(preds)
    weights = np.array([rng.uniform(0.5, 3.0) for _ in intervals])
    return stats, intervals, weights


def brute_force(stats, intervals, weights, max_groups, min_blocks=1):
    n = stats.n_blocks
    best = float("inf")
    for k in range(1, max_groups + 1):
        for cuts in itertools.combinations(range(1, n), k - 1):
            edges = [0, *cuts, n]
            if any(b - a < min_blocks for a, b in zip(edges, edges[1:])):
                continue
            total = sum(naive_group_cost(stats, intervals, weights, a, b)
                        for a, b in zip(edges, edges[1:]))
            best = min(best, total)
    return best


def test_cost_matrix_matches_naive():
    for seed in range(25):
        stats, intervals, weights = random_instance(seed)
        cost = cost_matrix(stats, intervals, weights)
        for i in range(stats.n_blocks):
            for j in range(i + 1, stats.n_blocks + 1):
                assert cost[i, j] == naive_group_cost(stats, intervals, weights, i, j)


def test_dp_is_optimal():
    for seed in range(25):
        stats, intervals, weights = random_instance(seed)
        cost = cost_matrix(stats, intervals, weights)
        for max_groups in (1, 2, 3, stats.n_blocks):
            _, total = dp_boundaries(cost, stats, max_groups, min_rows=1)
            assert abs(total - brute_force(stats, intervals, weights, max_groups)) < 1e-9


def test_dp_never_worse_than_uniform():
    for seed in range(20):
        stats, intervals, weights = random_instance(seed)
        cost = cost_matrix(stats, intervals, weights)
        n = stats.n_blocks
        uniform = uniform_boundaries(stats.n_rows, int(stats.rows[:2].sum()))
        uniform = [b for b in uniform if b in set(stats.row_offsets().tolist())]
        if uniform[-1] != stats.n_rows:
            continue
        reference = evaluate_boundaries(uniform, stats, cost)
        _, total = dp_boundaries(cost, stats, n, min_rows=1)
        assert total <= reference + 1e-9


def test_matching_is_monotone_in_group_growth():
    """The property the cost matrix pass relies on."""
    for seed in range(20):
        stats, intervals, weights = random_instance(seed)
        cost = cost_matrix(stats, intervals, weights)
        for i in range(stats.n_blocks):
            charged = []
            for j in range(i + 1, stats.n_blocks + 1):
                rows = float(stats.rows[i:j].sum())
                charged.append(cost[i, j] / rows)
            assert all(b >= a - 1e-9 for a, b in zip(charged, charged[1:]))


def test_group_budget_is_respected():
    stats, intervals, weights = random_instance(3)
    cost = cost_matrix(stats, intervals, weights)
    for budget in (1, 2, 3):
        boundaries, _ = dp_boundaries(cost, stats, budget, min_rows=1)
        assert len(boundaries) - 1 <= budget
