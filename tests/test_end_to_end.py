"""A fitted file must answer every query exactly like the baseline file."""

from __future__ import annotations

import datetime as dt
import random

import pyarrow as pa
import pyarrow.parquet as pq

from prunefit.generate import sample_workload
from prunefit.measure import benchmark, column_compressed_bytes
from prunefit.search import FitConfig, fit
from prunefit.writer import write_baseline, write_layout


def toy_table(rows: int = 6000, seed: int = 5) -> pa.Table:
    rng = random.Random(seed)
    providers = [f"p{i}" for i in range(12)]
    base = dt.datetime(2026, 1, 1)
    ts, prov, eid, lvl, pid, chan, payload = [], [], [], [], [], [], []
    for i in range(rows):
        ts.append(base + dt.timedelta(seconds=60 * i))
        prov.append(rng.choice(providers))
        eid.append(rng.choice([1, 2, 7, 4624]))
        lvl.append(rng.choice([1, 2, 3, 4]))
        pid.append(rng.choice([4, 900, 1500]))
        chan.append(rng.choice(["System", "Application"]))
        payload.append("x" * rng.randint(10, 60))
    return pa.table({
        "ts": pa.array(ts, pa.timestamp("us")), "provider": pa.array(prov),
        "event_id": pa.array(eid, pa.int32()), "level": pa.array(lvl, pa.int8()),
        "process_id": pa.array(pid, pa.int32()), "channel": pa.array(chan),
        "payload": pa.array(payload)})


def test_fitted_layout_answers_queries_identically(tmp_path):
    table = toy_table()
    train = sample_workload(table, 16, seed=1, prefix="tr")
    test = sample_workload(table, 16, seed=2, prefix="te")
    base_path = str(tmp_path / "baseline.parquet")
    write_baseline(table, base_path, rows_per_group=512)
    cfg = FitConfig(block_rows=256, rows_per_group=512, min_rows=256,
                    column_bytes=column_compressed_bytes(base_path))
    result = fit(table, train, cfg)
    fit_path = str(tmp_path / "fitted.parquet")
    info = write_layout(table, result.layout, fit_path)

    assert info["rows"] == table.num_rows
    assert info["row_groups"] <= result.max_groups
    measured = benchmark({"baseline": base_path, "prunefit": fit_path},
                         test, table.schema, repeats=1)
    for a, b in zip(measured["baseline"], measured["prunefit"]):
        assert a["rows_returned"] == b["rows_returned"]
    assert pq.read_table(fit_path).num_rows == table.num_rows


def test_fitted_layout_never_models_worse_than_baseline():
    table = toy_table(3000, seed=9)
    train = sample_workload(table, 20, seed=4, prefix="tr")
    result = fit(table, train, FitConfig(block_rows=256, rows_per_group=512, min_rows=256))
    assert result.layout.modeled_bytes <= result.baseline_modeled + 1e-9
    assert result.arrival_dp_modeled <= result.baseline_modeled + 1e-9
