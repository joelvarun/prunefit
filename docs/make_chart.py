"""Render docs/before_after.png from the measured numbers in docs/results.json."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches                               # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
BEFORE = "#eb6834"
AFTER = "#2a78d6"
NEUTRAL = "#c9c8c3"

ORDER = ["A_arrival_fixed", "B_sorted_fixed", "C_fitorder_fixed", "D_prunefit"]
SHORT = {
    "A_arrival_fixed": "Arrival order,\nfixed groups",
    "B_sorted_fixed": "Best single sort key,\nfixed groups",
    "C_fitorder_fixed": "Fitted order,\nfixed groups",
    "D_prunefit": "Fitted order,\nfitted boundaries",
}


def panel_layouts(ax, blob) -> None:
    layouts = blob["layouts"]
    groups = blob["layouts"]["A_arrival_fixed"]["row_groups"]
    values = [layouts[k]["totals"]["bytes_read"] / 1e6 for k in ORDER]
    base = values[0]
    colors = [BEFORE, NEUTRAL, NEUTRAL, AFTER]
    y = list(range(len(ORDER)))[::-1]
    ax.barh(y, values, height=0.58, color=colors, zorder=3)
    for pos, value, key in zip(y, values, ORDER):
        ax.text(value + base * 0.022, pos, f"{value:,.1f} MB", va="center",
                ha="left", color=INK, fontsize=10.5, fontweight="bold")
        if key != ORDER[0]:
            ax.text(base * 1.45, pos, f"{base/value:.2f}x less",
                    va="center", ha="right", color=INK_SOFT, fontsize=9.5)
    ax.text(values[0] * 0.5, y[0], "before", va="center", ha="center",
            color=SURFACE, fontsize=10, fontweight="bold")
    ax.text(values[-1] * 0.5, y[-1], "after", va="center", ha="center",
            color=SURFACE, fontsize=10, fontweight="bold")
    ax.set_yticks(y, [SHORT[k] for k in ORDER], fontsize=10, color=INK)
    ax.set_xlim(0, base * 1.48)
    ax.set_xlabel("Bytes read from disk, total over the workload (MB)",
                  fontsize=10, color=INK_SOFT)
    ax.set_title(f"Same rows, same {groups} row groups, identical answers",
                 fontsize=11, color=INK_SOFT, loc="left", pad=8)


def panel_templates(ax, blob) -> None:
    before = blob["layouts"]["A_arrival_fixed"]["by_template"]
    after = blob["layouts"]["D_prunefit"]["by_template"]
    keys = sorted(before, key=lambda k: -before[k]["bytes_read"])
    y = list(range(len(keys)))[::-1]
    height = 0.36
    for pos, key in zip(y, keys):
        b = before[key]["bytes_read"] / 1e6
        a = after[key]["bytes_read"] / 1e6
        ax.barh(pos + height / 2 + 0.02, b, height=height, color=BEFORE, zorder=3)
        ax.barh(pos - height / 2 - 0.02, a, height=height, color=AFTER, zorder=3)
        factor = b / a if a else float("inf")
        note = f"{factor:.2f}x less" if factor >= 1 else f"{1/factor:.2f}x more"
        ax.text(max(a, b) + 0.6, pos, note, va="center", ha="left",
                color=INK_SOFT, fontsize=9)
    labels = [f"{k.replace('_', ' ')}\n({before[k]['queries']} queries)" for k in keys]
    ax.set_yticks(y, labels, fontsize=9.5, color=INK)
    ax.set_xlim(0, max(v["bytes_read"] for v in before.values()) / 1e6 * 1.45)
    ax.set_xlabel("Bytes read from disk, by query shape (MB)",
                  fontsize=10, color=INK_SOFT)
    ax.set_title("Where the saving comes from, and the one shape that pays for it",
                 fontsize=11, color=INK_SOFT, loc="left", pad=8)
    handles = [mpatches.Patch(color=BEFORE, label="Arrival order, fixed groups"),
               mpatches.Patch(color=AFTER, label="prunefit")]
    ax.legend(handles=handles, frameon=False, fontsize=9.5, loc="lower right",
              labelcolor=INK_SOFT, handlelength=1.4, handleheight=0.9)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(axis="x", colors=INK_SOFT, labelsize=9.5, length=0)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="render the before and after chart")
    ap.add_argument("--results", default=os.path.join(here, "results.json"))
    ap.add_argument("--out", default=os.path.join(here, "before_after.png"))
    args = ap.parse_args()

    with open(args.results, "r", encoding="utf-8") as fh:
        blob = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4), dpi=150,
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    fig.patch.set_facecolor(SURFACE)
    panel_layouts(axes[0], blob)
    panel_templates(axes[1], blob)
    for ax in axes:
        style(ax)

    cfg, src = blob["config"], blob["source"]
    fit = blob["fit"]
    fig.suptitle(f"Bytes read answering {cfg['queries_per_workload']} held-out "
                 f"log queries over {src['rows']:,} rows of Windows event log",
                 fontsize=14, color=INK, x=0.012, ha="left", y=0.975,
                 fontweight="bold")
    fig.text(0.012, 0.915,
             f"Parquet, snappy, {cfg['rows_per_group']} rows per group "
             f"({cfg['group_budget']} groups) in every layout. "
             f"prunefit order: {fit['order_name']}. "
             f"Layout fitted on a disjoint sample of the same query log.",
             fontsize=10, color=INK_SOFT, ha="left")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.885))
    fig.savefig(args.out, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
