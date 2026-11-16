"""Codes must order exactly like the values they replace."""

from __future__ import annotations

import datetime as dt

import pyarrow as pa
import pytest

from prunefit.encode import encode_table
from prunefit.keys import order_index, parse_key
from prunefit.workload import Predicate, Query, Workload


def sample_table() -> pa.Table:
    return pa.table({
        "ts": pa.array([dt.datetime(2026, 1, d) for d in (3, 1, 2, 4)], pa.timestamp("us")),
        "provider": pa.array(["b", "a", "c", "a"]),
        "level": pa.array([2, 4, 1, 3], pa.int8()),
    })


def test_string_codes_preserve_order():
    codes, codecs = encode_table(sample_table(), ["provider"])
    values = ["b", "a", "c", "a"]
    pairs = sorted(zip(codes["provider"].tolist(), values))
    assert [v for _, v in pairs] == sorted(values)
    assert codecs["provider"].code_bounds("a") == (0, 0)


def test_absent_string_value_yields_empty_interval():
    _, codecs = encode_table(sample_table(), ["provider"])
    lo, hi = codecs["provider"].code_bounds("bb")
    assert lo > hi


def test_predicate_intervals():
    _, codecs = encode_table(sample_table(), ["provider", "level", "ts"])
    eq = Predicate("level", "eq", 2).code_interval(codecs["level"])
    assert eq == (2, 2)
    rng = Predicate("level", "range", lo=1, hi=3).code_interval(codecs["level"])
    assert rng == (1, 3)
    inp = Predicate("provider", "in", ["a", "c"]).code_interval(codecs["provider"])
    assert inp == (0, 2)
    ge = Predicate("level", "ge", lo=2).code_interval(codecs["level"])
    assert ge[0] == 2 and ge[1] > 10**15


def test_bucketed_key_groups_time():
    codes, _ = encode_table(sample_table(), ["ts", "provider"])
    idx = order_index(codes, ["ts~7d", "provider"])
    providers = [["b", "a", "c", "a"][i] for i in idx]
    assert providers == sorted(providers)   # one week bucket, so provider decides
    assert parse_key("ts~12h") == ("ts", 12 * 3_600_000_000)


def test_workload_round_trip(tmp_path):
    workload = Workload([Query("q0", [Predicate("provider", "eq", "a"),
                                      Predicate("ts", "range", lo="2026-01-01T00:00:00",
                                                hi="2026-01-03T00:00:00")],
                               template="drill")])
    path = tmp_path / "w.json"
    workload.save(str(path))
    again = Workload.load(str(path))
    assert again.queries[0].to_json() == workload.queries[0].to_json()
    assert again.predicate_columns == ["provider", "ts"]


def test_unknown_bucket_unit_rejected():
    with pytest.raises(ValueError):
        parse_key("ts~3fortnights")
