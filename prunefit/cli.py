"""Command line entry points."""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

from prunefit.generate import sample_workload
from prunefit.measure import benchmark, column_compressed_bytes
from prunefit.search import FitConfig, fit
from prunefit.workload import Workload
from prunefit.writer import write_baseline, write_layout


def _config(args, input_path: str) -> FitConfig:
    return FitConfig(block_rows=args.block_rows, rows_per_group=args.rows_per_group,
                     min_rows=args.min_rows, max_rows=args.max_rows,
                     max_key_candidates=args.key_candidates,
                     max_key_depth=args.key_depth,
                     column_bytes=column_compressed_bytes(input_path))


def cmd_sample(args) -> int:
    table = pq.read_table(args.input)
    workload = sample_workload(table, args.queries, args.seed, args.prefix)
    workload.save(args.out)
    print(f"wrote {args.out} with {len(workload)} queries")
    return 0


def cmd_fit(args) -> int:
    table = pq.read_table(args.input)
    workload = Workload.load(args.workload)
    result = fit(table, workload, _config(args, args.input))
    layout = result.layout
    print(f"order            {layout.order_name}")
    print(f"row groups       {layout.n_groups} (budget {result.max_groups})")
    print(f"group rows       min {min(layout.group_rows)} max {max(layout.group_rows)}")
    print(f"modeled bytes    baseline {result.baseline_modeled/1e6:.1f} MB -> "
          f"fitted {layout.modeled_bytes/1e6:.1f} MB "
          f"({result.modeled_reduction*100:.1f} percent less)")
    print(f"search           {len(result.candidates)} orders in {result.seconds:.1f} s")
    if args.out:
        info = write_layout(table, layout, args.out, args.compression)
        print(f"wrote            {info['path']} ({info['bytes']} bytes, "
              f"{info['row_groups']} row groups)")
    if args.baseline_out:
        info = write_baseline(table, args.baseline_out, args.rows_per_group,
                              compression=args.compression)
        print(f"wrote            {info['path']} ({info['bytes']} bytes, "
              f"{info['row_groups']} row groups)")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump({"order": layout.order, "order_name": layout.order_name,
                       "boundaries": layout.boundaries,
                       "modeled_bytes": layout.modeled_bytes,
                       "baseline_modeled": result.baseline_modeled,
                       "arrival_dp_modeled": result.arrival_dp_modeled,
                       "candidates": result.candidates[:20],
                       "seconds": result.seconds}, fh, indent=2)
        print(f"wrote            {args.report}")
    return 0


def cmd_bench(args) -> int:
    workload = Workload.load(args.workload)
    paths = {}
    for item in args.layout:
        label, _, path = item.partition("=")
        if not path:
            label, path = os.path.basename(path or label), item
        paths[label] = path
    schema = pq.read_schema(next(iter(paths.values())))
    results = benchmark(paths, workload, schema, args.repeats)
    print(f"{'layout':22s} {'file MB':>9s} {'read MB':>9s} {'groups':>8s} {'time ms':>9s}")
    first = None
    for label, rows in results.items():
        read = sum(r["bytes_read"] for r in rows)
        groups = sum(r["row_groups_scanned"] for r in rows)
        seconds = sum(r["seconds"] for r in rows)
        first = read if first is None else first
        print(f"{label:22s} {os.path.getsize(paths[label])/1e6:9.2f} {read/1e6:9.2f} "
              f"{groups:8d} {seconds*1000:9.1f}   x{first/max(read,1):.2f}")
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"wrote {args.report}")
    return 0


def cmd_explain(args) -> int:
    table = pq.read_table(args.input)
    workload = Workload.load(args.workload)
    result = fit(table, workload, _config(args, args.input))
    print(f"{'order':38s} {'modeled MB':>11s} {'groups':>7s}")
    for row in result.candidates[: args.top]:
        print(f"{row['order']:38s} {row['modeled_bytes']/1e6:11.1f} {row['groups']:7d}")
    print(f"\nbaseline (arrival order, fixed groups): "
          f"{result.baseline_modeled/1e6:.1f} MB modeled")
    print(f"arrival order with fitted boundaries:   "
          f"{result.arrival_dp_modeled/1e6:.1f} MB modeled")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="prunefit",
                                 description="Fit a Parquet row-group layout to a query log")
    sub = ap.add_subparsers(dest="command", required=True)

    def add_fit_flags(p):
        p.add_argument("--block-rows", type=int, default=1024)
        p.add_argument("--rows-per-group", type=int, default=2048)
        p.add_argument("--min-rows", type=int, default=1024)
        p.add_argument("--max-rows", type=int, default=None)
        p.add_argument("--key-candidates", type=int, default=4)
        p.add_argument("--key-depth", type=int, default=3)

    p = sub.add_parser("sample-workload", help="draw a query log from a table's values")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--queries", type=int, default=64)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--prefix", default="q")
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("fit", help="fit a layout and optionally write it")
    p.add_argument("--input", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--out")
    p.add_argument("--baseline-out")
    p.add_argument("--report")
    p.add_argument("--compression", default="snappy")
    add_fit_flags(p)
    p.set_defaults(func=cmd_fit)

    p = sub.add_parser("bench", help="measure layouts on a query log")
    p.add_argument("--workload", required=True)
    p.add_argument("--layout", action="append", required=True,
                   help="label=path, repeatable")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--report")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("explain", help="rank candidate orders without writing files")
    p.add_argument("--input", required=True)
    p.add_argument("--workload", required=True)
    p.add_argument("--top", type=int, default=12)
    add_fit_flags(p)
    p.set_defaults(func=cmd_explain)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
