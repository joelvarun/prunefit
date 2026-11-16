"""Sort key specs, including coarsened time keys.

A flat sort has to choose between time locality and value clustering. A key
spec like `ts~7d` coarsens time into buckets, so an order such as
`ts~7d, provider, ts` keeps week level time pruning and still clusters each
week by provider. No Parquet writer exposes that today: all you get is
`row_group_size`.
"""

from __future__ import annotations

import numpy as np

UNITS = {"us": 1, "ms": 1_000, "s": 1_000_000, "m": 60_000_000,
         "h": 3_600_000_000, "d": 86_400_000_000, "w": 604_800_000_000}


def parse_key(spec: str) -> tuple[str, int | None]:
    """`provider` -> (provider, None); `ts~7d` -> (ts, 604800000000)."""
    if "~" not in spec:
        return spec, None
    column, width = spec.split("~", 1)
    for unit in sorted(UNITS, key=len, reverse=True):
        if width.endswith(unit):
            return column, int(float(width[: -len(unit)]) * UNITS[unit])
    raise ValueError(f"unknown bucket width {width!r} in key {spec!r}")


def key_columns(specs: list[str]) -> list[str]:
    """Underlying table columns a list of key specs needs."""
    out: list[str] = []
    for spec in specs:
        column, _ = parse_key(spec)
        if column not in out:
            out.append(column)
    return out


def key_values(codes: dict[str, np.ndarray], spec: str) -> np.ndarray:
    column, bucket = parse_key(spec)
    values = codes[column]
    if bucket is None:
        return values
    return values // bucket


def order_index(codes: dict[str, np.ndarray], specs: list[str]) -> np.ndarray:
    """Stable row permutation sorting by `specs`, ascending, left to right."""
    if not specs:
        return np.arange(len(next(iter(codes.values()))), dtype=np.int64)
    return np.lexsort([key_values(codes, s) for s in reversed(specs)])
