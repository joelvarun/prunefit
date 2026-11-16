"""Build a realistic log-analytics Parquet dataset.

On Windows the rows come from the machine's own event logs, which is what the
benchmark in the README uses. Anywhere else, a synthetic generator with the
same shape (skewed providers, bursty arrivals, rare error levels) stands in so
the demo still runs end to end.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import re
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq

RE_TIME = re.compile(r"SystemTime=['\"]([^'\"]+)['\"]")
RE_PROVIDER = re.compile(r"<Provider Name=['\"]([^'\"]*)['\"]")
RE_EVENTID = re.compile(r"<EventID[^>]*>(\d+)</EventID>")
RE_LEVEL = re.compile(r"<Level>(\d+)</Level>")
RE_TASK = re.compile(r"<Task>(\d+)</Task>")
RE_EXEC = re.compile(r"<Execution ProcessID=['\"](\d+)['\"] ThreadID=['\"](\d+)['\"]")
RE_CHANNEL = re.compile(r"<Channel>([^<]*)</Channel>")
RE_COMPUTER = re.compile(r"<Computer>([^<]*)</Computer>")
RE_RECORD = re.compile(r"<EventRecordID>(\d+)</EventRecordID>")
RE_BODY = re.compile(r"<(?:EventData|UserData)[^>]*>(.*)</(?:EventData|UserData)>", re.S)
RE_TAG = re.compile(r"<[^>]+>")

CHANNELS = [
    "System",
    "Application",
    "Security",
    "Microsoft-Windows-Store/Operational",
    "Microsoft-Windows-StateRepository/Operational",
    "Setup",
    "Microsoft-Windows-Winlogon/Operational",
    "Microsoft-Windows-PowerShell/Operational",
    "Microsoft-Windows-NetworkProfile/Operational",
    "Microsoft-Windows-WindowsUpdateClient/Operational",
]

PAYLOAD_CHARS = 256


def _epoch_us(stamp: str) -> int:
    """Parse an event log SystemTime into microseconds since the epoch."""
    import datetime as dt

    stamp = stamp.rstrip("Z")
    if "." in stamp:
        head, frac = stamp.split(".", 1)
        frac = (frac + "000000")[:6]
    else:
        head, frac = stamp, "000000"
    base = dt.datetime.strptime(head, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=dt.timezone.utc)
    return int(base.timestamp()) * 1_000_000 + int(frac)


def iter_records(path: str, chunk: int = 1 << 22):
    """Yield one XML record per event, independent of how the dump is wrapped."""
    buf = ""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        while True:
            data = fh.read(chunk)
            if not data:
                break
            buf += data
            parts = buf.split("</Event>")
            buf = parts.pop()
            for part in parts:
                yield part


def parse_dump(paths: list[str]) -> dict[str, list]:
    cols: dict[str, list] = {k: [] for k in (
        "ts", "channel", "provider", "event_id", "level", "task",
        "process_id", "thread_id", "record_id", "computer", "payload")}
    for path in paths:
        for record in iter_records(path):
            if "<Event" not in record:
                continue
            mt = RE_TIME.search(record)
            mp = RE_PROVIDER.search(record)
            me = RE_EVENTID.search(record)
            if not (mt and mp and me):
                continue
            ml = RE_LEVEL.search(record)
            mtask = RE_TASK.search(record)
            mx = RE_EXEC.search(record)
            mc = RE_CHANNEL.search(record)
            mcomp = RE_COMPUTER.search(record)
            mr = RE_RECORD.search(record)
            mb = RE_BODY.search(record)
            body = RE_TAG.sub(" ", mb.group(1)) if mb else ""
            cols["ts"].append(_epoch_us(mt.group(1)))
            cols["channel"].append(mc.group(1) if mc else "")
            cols["provider"].append(mp.group(1))
            cols["event_id"].append(int(me.group(1)))
            cols["level"].append(int(ml.group(1)) if ml else 0)
            cols["task"].append(int(mtask.group(1)) if mtask else 0)
            cols["process_id"].append(int(mx.group(1)) if mx else 0)
            cols["thread_id"].append(int(mx.group(2)) if mx else 0)
            cols["record_id"].append(int(mr.group(1)) if mr else 0)
            cols["computer"].append(mcomp.group(1) if mcomp else "")
            cols["payload"].append(" ".join(body.split())[:PAYLOAD_CHARS])
    return cols


def harvest_windows(dump_dir: str, per_channel: int) -> list[str]:
    os.makedirs(dump_dir, exist_ok=True)
    paths = []
    for channel in CHANNELS:
        out = os.path.join(dump_dir, channel.replace("/", "_").replace(":", "_") + ".xml")
        if not os.path.exists(out):
            with open(out, "wb") as fh:
                proc = subprocess.run(
                    ["wevtutil", "qe", channel, "/f:xml", f"/c:{per_channel}", "/rd:true"],
                    stdout=fh, stderr=subprocess.DEVNULL, check=False)
            if proc.returncode != 0 or os.path.getsize(out) == 0:
                os.remove(out)
                continue
        paths.append(out)
    return paths


def synthesize(rows: int, seed: int = 7) -> dict[str, list]:
    """Synthetic rows with the same shape as a real event log."""
    rng = random.Random(seed)
    providers = [f"Provider-{i:03d}" for i in range(220)]
    weights = [1.0 / (i + 1) ** 1.1 for i in range(len(providers))]
    words = [f"tok{i:04d}" for i in range(900)]
    cols: dict[str, list] = {k: [] for k in (
        "ts", "channel", "provider", "event_id", "level", "task",
        "process_id", "thread_id", "record_id", "computer", "payload")}
    now = 1_760_000_000_000_000
    ts = now - rows * 4_000_000
    for i in range(rows):
        ts += int(rng.expovariate(1 / 4_000_000.0))
        provider = rng.choices(providers, weights)[0]
        cols["ts"].append(ts)
        cols["channel"].append(CHANNELS[hash(provider) % len(CHANNELS)])
        cols["provider"].append(provider)
        cols["event_id"].append(rng.choice([1, 7, 16, 100, 1000, 1001, 4624, 4634, 7036, 10016]))
        cols["level"].append(rng.choices([0, 1, 2, 3, 4], [40, 2, 8, 45, 5])[0])
        cols["task"].append(rng.randrange(0, 64))
        cols["process_id"].append(rng.choice([4, 892, 1204, 2680, 3344, 5120, 7788, 9012]))
        cols["thread_id"].append(rng.randrange(100, 60000))
        cols["record_id"].append(i + 1)
        cols["computer"].append("HOST-01")
        n = rng.randrange(6, 30)
        cols["payload"].append(" ".join(rng.choice(words) for _ in range(n))[:PAYLOAD_CHARS])
    return cols


SCHEMA = pa.schema([
    ("ts", pa.timestamp("us")),
    ("channel", pa.string()),
    ("provider", pa.string()),
    ("event_id", pa.int32()),
    ("level", pa.int8()),
    ("task", pa.int32()),
    ("process_id", pa.int32()),
    ("thread_id", pa.int32()),
    ("record_id", pa.int64()),
    ("computer", pa.string()),
    ("payload", pa.string()),
])


def to_table(cols: dict[str, list]) -> pa.Table:
    table = pa.table({
        "ts": pa.array(cols["ts"], pa.timestamp("us")),
        "channel": pa.array(cols["channel"], pa.string()),
        "provider": pa.array(cols["provider"], pa.string()),
        "event_id": pa.array(cols["event_id"], pa.int32()),
        "level": pa.array(cols["level"], pa.int8()),
        "task": pa.array(cols["task"], pa.int32()),
        "process_id": pa.array(cols["process_id"], pa.int32()),
        "thread_id": pa.array(cols["thread_id"], pa.int32()),
        "record_id": pa.array(cols["record_id"], pa.int64()),
        "computer": pa.array(cols["computer"], pa.string()),
        "payload": pa.array(cols["payload"], pa.string()),
    }, schema=SCHEMA)
    # Arrival order: a log pipeline appends rows as events land.
    return table.sort_by([("ts", "ascending")])


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the log-analytics source table")
    ap.add_argument("--out", default="data/events_source.parquet")
    ap.add_argument("--dump-dir", default="_evtdump")
    ap.add_argument("--per-channel", type=int, default=45000)
    ap.add_argument("--synthetic-rows", type=int, default=160000)
    ap.add_argument("--force-synthetic", action="store_true")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dump_dir, "*.xml")))
    if not args.force_synthetic and not paths and sys.platform == "win32":
        print("harvesting event logs via wevtutil")
        paths = harvest_windows(args.dump_dir, args.per_channel)
    if args.force_synthetic or not paths:
        print(f"synthesizing {args.synthetic_rows} rows")
        cols = synthesize(args.synthetic_rows)
        source = "synthetic"
    else:
        print(f"parsing {len(paths)} event log dumps")
        cols = parse_dump(paths)
        source = "windows-event-log"

    table = to_table(cols)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pq.write_table(table, args.out, compression="snappy")
    print(f"source={source} rows={table.num_rows} "
          f"providers={len(set(cols['provider']))} bytes={os.path.getsize(args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
