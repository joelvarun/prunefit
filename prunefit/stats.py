"""Per block statistics, the granularity the boundary search works at."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockStats:
    """Min and max codes for every candidate block of rows."""

    block_rows: int
    rows: np.ndarray            # rows per block
    mins: dict[str, np.ndarray]
    maxs: dict[str, np.ndarray]

    @property
    def n_blocks(self) -> int:
        return int(self.rows.shape[0])

    @property
    def n_rows(self) -> int:
        return int(self.rows.sum())

    def row_offsets(self) -> np.ndarray:
        """Row offset of every block boundary, length n_blocks + 1."""
        return np.concatenate([[0], np.cumsum(self.rows)])


def block_stats(codes: dict[str, np.ndarray], block_rows: int) -> BlockStats:
    """Reduce code arrays to per block min and max.

    Blocks are the atoms of the layout search: a row group is always a run of
    whole blocks, which caps the search at O(blocks^2) instead of O(rows^2).
    """
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    n_rows = len(next(iter(codes.values())))
    starts = np.arange(0, n_rows, block_rows, dtype=np.int64)
    ends = np.minimum(starts + block_rows, n_rows)
    mins = {c: np.minimum.reduceat(v, starts) for c, v in codes.items()}
    maxs = {c: np.maximum.reduceat(v, starts) for c, v in codes.items()}
    return BlockStats(block_rows, (ends - starts).astype(np.int64), mins, maxs)
