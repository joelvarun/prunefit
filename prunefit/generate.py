"""Sample a realistic log-analytics query log from a table's own values.

The six templates are the shapes a log table actually gets queried with:
drill into one provider over a window, triage errors, look up an event id,
redraw a dashboard over the last day, attribute activity to a process, audit a
channel. Values are drawn from the table so selectivities are real.
"""

from __future__ import annotations

import collections
import datetime as dt
import random

import pyarrow as pa

from prunefit.workload import Predicate, Query, Workload

TEMPLATES = (
    ("provider_drilldown", 0.30),
    ("error_triage", 0.20),
    ("event_lookup", 0.20),
    ("dashboard_window", 0.15),
    ("process_attribution", 0.10),
    ("channel_audit", 0.05),
)

DASHBOARD_COLUMNS = ["ts", "provider", "level", "event_id"]


def _freq(table: pa.Table, column: str, top: int | None = None) -> tuple[list, list[float]]:
    counts = collections.Counter(table.column(column).to_pylist())
    items = counts.most_common(top)
    return [k for k, _ in items], [float(v) for _, v in items]


def _iso(stamp: dt.datetime) -> str:
    return stamp.replace(tzinfo=None).isoformat()


def sample_workload(table: pa.Table, n_queries: int = 64, seed: int = 11,
                    prefix: str = "q") -> Workload:
    rng = random.Random(seed)
    providers, provider_w = _freq(table, "provider")
    event_ids, event_w = _freq(table, "event_id")
    pids, pid_w = _freq(table, "process_id", top=40)
    channels, channel_w = _freq(table, "channel")
    stamps = table.column("ts").to_pylist()
    t_lo, t_hi = min(stamps), max(stamps)
    span = (t_hi - t_lo).total_seconds()

    names = [t for t, _ in TEMPLATES]
    probs = [w for _, w in TEMPLATES]
    queries: list[Query] = []
    for i in range(n_queries):
        template = rng.choices(names, probs)[0]
        preds: list[Predicate] = []
        projection = None
        if template == "provider_drilldown":
            preds.append(Predicate("provider", "eq", rng.choices(providers, provider_w)[0]))
            preds.append(_window(rng, t_lo, span, days=14))
        elif template == "error_triage":
            preds.append(Predicate("level", "range", lo=1, hi=2))
            preds.append(_window(rng, t_lo, span, days=7))
        elif template == "event_lookup":
            preds.append(Predicate("event_id", "eq", rng.choices(event_ids, event_w)[0]))
        elif template == "dashboard_window":
            preds.append(_window(rng, t_lo, span, days=1))
            projection = list(DASHBOARD_COLUMNS)
        elif template == "process_attribution":
            preds.append(Predicate("process_id", "eq", rng.choices(pids, pid_w)[0]))
        else:
            preds.append(Predicate("channel", "eq", rng.choices(channels, channel_w)[0]))
            preds.append(Predicate("level", "ge", lo=3))
        queries.append(Query(name=f"{prefix}{i:03d}", predicates=preds, weight=1.0,
                             template=template, projection=projection))
    return Workload(queries)


def _window(rng: random.Random, t_lo: dt.datetime, span: float, days: int) -> Predicate:
    width = days * 86400.0
    start = t_lo + dt.timedelta(seconds=rng.uniform(0, max(0.0, span - width)))
    return Predicate("ts", "range", lo=_iso(start), hi=_iso(start + dt.timedelta(seconds=width)))
