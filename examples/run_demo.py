"""End to end demo: fit a layout, write it, measure it, record the numbers.

Four layouts are written from the same rows with the same number of row
groups, so the only difference between them is row order and where the group
boundaries fall:

  A  arrival order, fixed size groups        what a Parquet writer gives you
  B  best single sort key, fixed size groups the manual clustering answer
  C  fitted order, fixed size groups         ordering only, no boundary search
  D  fitted order, fitted boundaries         prunefit

The layout is fitted on one sample of the query log and measured on a disjoint
held-out sample, so the reported numbers are not fitted to themselves.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import subprocess
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prunefit.generate import sample_workload                      # noqa: E402
from prunefit.layout import Layout, uniform_boundaries              # noqa: E402
from prunefit.measure import benchmark, column_compressed_bytes     # noqa: E402
from prunefit.search import FitConfig, fit                          # noqa: E402
from prunefit.writer import write_baseline, write_layout            # noqa: E402

LABELS = {
    "A_arrival_fixed": "Arrival order, fixed groups",
    "B_sorted_fixed": "Best single sort key, fixed groups",
    "C_fitorder_fixed": "Fitted order, fixed groups",
    "D_prunefit": "prunefit: fitted order and boundaries",
}


def ensure_source(path: str, builder: str) -> None:
    if os.path.exists(path):
        return
    print(f"{path} missing, building it")
    subprocess.run([sys.executable, builder, "--out", path], check=True)


def totals(rows: list[dict]) -> dict:
    return {
        "bytes_read": sum(r["bytes_read"] for r in rows),
        "row_groups_scanned": sum(r["row_groups_scanned"] for r in rows),
        "rows_scanned": sum(r["rows_scanned"] for r in rows),
        "rows_returned": sum(r["rows_returned"] for r in rows),
        "seconds": sum(r["seconds"] for r in rows),
    }


def by_template(rows: list[dict]) -> dict[str, dict]:
    agg: dict[str, dict] = collections.defaultdict(
        lambda: {"queries": 0, "bytes_read": 0, "row_groups_scanned": 0})
    for r in rows:
        slot = agg[r["template"]]
        slot["queries"] += 1
        slot["bytes_read"] += r["bytes_read"]
        slot["row_groups_scanned"] += r["row_groups_scanned"]
    return dict(agg)


def parse_args(root: str, here: str):
    ap = argparse.ArgumentParser(description="prunefit end to end demo")
    ap.add_argument("--source", default=os.path.join(root, "data", "events_source.parquet"))
    ap.add_argument("--workdir", default=os.path.join(root, "data"))
    ap.add_argument("--results", default=os.path.join(root, "docs", "results.json"))
    ap.add_argument("--rows-per-group", type=int, default=2048)
    ap.add_argument("--block-rows", type=int, default=1024)
    ap.add_argument("--queries", type=int, default=48)
    ap.add_argument("--repeats", type=int, default=5)
    return ap.parse_args()


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    args = parse_args(root, here)

    ensure_source(args.source, os.path.join(here, "build_event_dataset.py"))
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)

    table = pq.read_table(args.source)
    print(f"source {args.source}: {table.num_rows} rows, "
          f"{os.path.getsize(args.source)} bytes")

    train = sample_workload(table, args.queries, seed=11, prefix="tr")
    test = sample_workload(table, args.queries, seed=99, prefix="te")
    train.save(os.path.join(args.workdir, "workload_train.json"))
    test.save(os.path.join(args.workdir, "workload_test.json"))

    cfg = FitConfig(block_rows=args.block_rows, rows_per_group=args.rows_per_group,
                    min_rows=args.block_rows,
                    column_bytes=column_compressed_bytes(args.source))
    result = fit(table, train, cfg)
    layout = result.layout
    print(f"fit: order={layout.order_name} groups={layout.n_groups} "
          f"budget={result.max_groups} in {result.seconds:.1f}s over "
          f"{len(result.candidates)} candidate orders")
    print(f"modeled bytes scanned: baseline {result.baseline_modeled/1e6:.1f} MB to "
          f"fitted {layout.modeled_bytes/1e6:.1f} MB")

    single = next(r for r in result.candidates
                  if len(r["keys"]) == 1 and r["order"] != "arrival")
    fixed = uniform_boundaries(table.num_rows, args.rows_per_group)
    files = {}
    files["A_arrival_fixed"] = write_baseline(
        table, os.path.join(args.workdir, "layout_a_arrival_fixed.parquet"),
        args.rows_per_group)
    files["B_sorted_fixed"] = write_layout(
        table, Layout(order=single["keys"], boundaries=fixed, order_name=single["order"]),
        os.path.join(args.workdir, "layout_b_sorted_fixed.parquet"))
    files["C_fitorder_fixed"] = write_layout(
        table, Layout(order=layout.order, boundaries=fixed, order_name=layout.order_name),
        os.path.join(args.workdir, "layout_c_fitorder_fixed.parquet"))
    files["D_prunefit"] = write_layout(
        table, layout, os.path.join(args.workdir, "layout_d_prunefit.parquet"))
    for key, info in files.items():
        print(f"  {key:18s} {info['bytes']:>9d} bytes  {info['row_groups']:>3d} row groups")

    started = time.perf_counter()
    measured = benchmark({k: v["path"] for k, v in files.items()}, test,
                         table.schema, args.repeats)
    print(f"measured {len(test)} held-out queries on 4 layouts in "
          f"{time.perf_counter()-started:.1f}s")

    reference = measured["A_arrival_fixed"]
    for key, rows in measured.items():
        for a, b in zip(reference, rows):
            if a["rows_returned"] != b["rows_returned"]:
                raise SystemExit(f"layout {key} changed the answer for {a['query']}")

    out = {
        "environment": {
            "python": platform.python_version(), "pyarrow": pa.__version__,
            "platform": platform.platform(), "generated": time.strftime("%Y-%m-%d"),
        },
        "source": {"path": os.path.basename(args.source), "rows": table.num_rows,
                   "bytes": os.path.getsize(args.source),
                   "columns": table.column_names},
        "config": {"rows_per_group": args.rows_per_group, "block_rows": args.block_rows,
                   "group_budget": result.max_groups, "queries_per_workload": args.queries,
                   "repeats": args.repeats},
        "fit": {"order": layout.order, "order_name": layout.order_name,
                "groups": layout.n_groups,
                "group_rows_min": min(layout.group_rows),
                "group_rows_max": max(layout.group_rows),
                "modeled_bytes": layout.modeled_bytes,
                "baseline_modeled": result.baseline_modeled,
                "arrival_dp_modeled": result.arrival_dp_modeled,
                "candidates": result.candidates[:12], "seconds": result.seconds},
        "layouts": {}, "labels": LABELS,
    }
    base = totals(reference)
    for key, rows in measured.items():
        agg = totals(rows)
        out["layouts"][key] = {
            "label": LABELS[key], "file_bytes": files[key]["bytes"],
            "row_groups": files[key]["row_groups"], "totals": agg,
            "reduction_vs_baseline": 1.0 - agg["bytes_read"] / base["bytes_read"],
            "speedup_vs_baseline": base["seconds"] / agg["seconds"],
            "by_template": by_template(rows), "queries": rows,
        }

    with open(args.results, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.results}")

    header = ("layout", "file MB", "read MB", "groups", "ms", "factor")
    print()
    print(f"{header[0]:38s} {header[1]:>8s} {header[2]:>8s} {header[3]:>7s} "
          f"{header[4]:>8s} {header[5]:>7s}")
    for key, rows in measured.items():
        agg = totals(rows)
        print(f"{LABELS[key]:38s} {files[key]['bytes']/1e6:8.2f} "
              f"{agg['bytes_read']/1e6:8.2f} {agg['row_groups_scanned']:7d} "
              f"{agg['seconds']*1000:8.1f} "
              f"{base['bytes_read']/agg['bytes_read']:6.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
