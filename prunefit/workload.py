"""Query log representation: the thing a layout is fitted to."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field

import pyarrow as pa
import pyarrow.compute as pc

from prunefit.encode import NEG_INF, POS_INF, ColumnCodec

OPS = ("eq", "in", "range", "le", "ge")


@dataclass
class Predicate:
    column: str
    op: str
    value: object = None
    lo: object = None
    hi: object = None

    def __post_init__(self) -> None:
        if self.op not in OPS:
            raise ValueError(f"unsupported op {self.op!r}, expected one of {OPS}")

    def code_interval(self, codec: ColumnCodec) -> tuple[int, int]:
        """Inclusive code interval this predicate can match."""
        if self.op == "eq":
            return codec.code_bounds(self.value)
        if self.op == "in":
            bounds = [codec.code_bounds(v) for v in self.value]
            return min(b[0] for b in bounds), max(b[1] for b in bounds)
        if self.op == "range":
            return codec.lower(self.lo), codec.upper(self.hi)
        if self.op == "le":
            return NEG_INF, codec.upper(self.hi if self.hi is not None else self.value)
        return codec.lower(self.lo if self.lo is not None else self.value), POS_INF

    def expression(self, dtype: pa.DataType):
        """Equivalent pyarrow dataset filter expression."""
        f = pc.field(self.column)
        cast = lambda v: _native(v, dtype)  # noqa: E731
        if self.op == "eq":
            return f == cast(self.value)
        if self.op == "in":
            return f.isin([cast(v) for v in self.value])
        if self.op == "range":
            return (f >= cast(self.lo)) & (f <= cast(self.hi))
        if self.op == "le":
            return f <= cast(self.hi if self.hi is not None else self.value)
        return f >= cast(self.lo if self.lo is not None else self.value)

    def to_json(self) -> dict:
        out = {"column": self.column, "op": self.op}
        for key in ("value", "lo", "hi"):
            val = getattr(self, key)
            if val is not None:
                out[key] = _jsonable(val)
        return out

    @staticmethod
    def from_json(blob: dict) -> "Predicate":
        return Predicate(blob["column"], blob["op"], blob.get("value"),
                         blob.get("lo"), blob.get("hi"))


@dataclass
class Query:
    name: str
    predicates: list[Predicate]
    weight: float = 1.0
    template: str = ""
    projection: list[str] | None = None

    @property
    def columns(self) -> list[str]:
        return [p.column for p in self.predicates]

    def code_intervals(self, codecs: dict[str, ColumnCodec]) -> list[tuple[str, int, int]]:
        return [(p.column, *p.code_interval(codecs[p.column])) for p in self.predicates]

    def expression(self, schema: pa.Schema):
        expr = None
        for p in self.predicates:
            part = p.expression(schema.field(p.column).type)
            expr = part if expr is None else (expr & part)
        return expr

    def to_json(self) -> dict:
        return {"name": self.name, "template": self.template, "weight": self.weight,
                "projection": self.projection,
                "predicates": [p.to_json() for p in self.predicates]}

    @staticmethod
    def from_json(blob: dict) -> "Query":
        return Query(blob["name"], [Predicate.from_json(p) for p in blob["predicates"]],
                     blob.get("weight", 1.0), blob.get("template", ""),
                     blob.get("projection"))


@dataclass
class Workload:
    queries: list[Query] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)

    @property
    def predicate_columns(self) -> list[str]:
        """Predicate columns ordered by the total query weight that touches them."""
        weight: dict[str, float] = {}
        for q in self.queries:
            for col in q.columns:
                weight[col] = weight.get(col, 0.0) + q.weight
        return [c for c, _ in sorted(weight.items(), key=lambda kv: -kv[1])]

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"queries": [q.to_json() for q in self.queries]}, fh, indent=2)

    @staticmethod
    def load(path: str) -> "Workload":
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        return Workload([Query.from_json(q) for q in blob["queries"]])


def _jsonable(value):
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _native(value, dtype: pa.DataType):
    if pa.types.is_timestamp(dtype) and isinstance(value, str):
        stamp = dt.datetime.fromisoformat(value)
        return stamp.replace(tzinfo=None)
    if pa.types.is_timestamp(dtype) and isinstance(value, dt.datetime):
        return value.replace(tzinfo=None)
    return value
