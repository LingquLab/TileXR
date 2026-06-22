#!/usr/bin/env python3
"""Plot TileXR UDMA P2P performance CSV files."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", help="p2p_perf.csv files to merge")
    parser.add_argument("--output", default="tilexr_udma_p2p_perf_curve.png", help="bandwidth PNG path")
    parser.add_argument("--latency-output", default=None, help="latency PNG path")
    parser.add_argument("--latency-max-bytes", type=int, default=1024 * 1024,
        help="only plot latency rows up to this size; default: 1048576")
    parser.add_argument("--direction", default=None, help="only plot one direction, for example 0to1")
    parser.add_argument("--labels", default=None,
        help="comma-separated labels matching input CSV files, for example direct_urma,memory")
    return parser.parse_args()


def infer_label(path):
    text = str(path)
    if "direct_urma" in text:
        return "direct_urma"
    if "memory" in text:
        return "memory"
    return Path(path).parent.name or Path(path).stem


def load_rows(paths, labels=None, direction_filter=None):
    if labels is not None and len(labels) != len(paths):
        raise SystemExit("--labels count must match input CSV count")
    merged = {}
    for index, path in enumerate(paths):
        label = labels[index] if labels is not None else infer_label(path)
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if direction_filter is not None and row["direction"] != direction_filter:
                    continue
                series = label if labels is not None or len(paths) > 1 else row["direction"]
                key = (series, int(row["bytes"]))
                merged[key] = {
                    "series": series,
                    "direction": row["direction"],
                    "bytes": int(row["bytes"]),
                    "avg_us": float(row["avg_us"]),
                    "bw_GBps": float(row["bw_GBps"]),
                    "src": row["src"],
                    "dst": row["dst"],
                    "ranks": row["ranks"],
                    "status": int(row["status"]),
                    "errors": int(row["errors"]),
                }
    grouped = defaultdict(list)
    for row in merged.values():
        grouped[row["series"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda item: item["bytes"])
    return grouped


def default_latency_output(output):
    path = Path(output)
    suffix = path.suffix or ".png"
    return str(path.with_name(path.stem + "_latency" + suffix))


def format_bytes(value):
    units = [("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)]
    for suffix, scale in units:
        if value >= scale and value % scale == 0:
            return f"{value // scale}{suffix}"
    return f"{value}B"


def byte_ticks(grouped):
    values = sorted({row["bytes"] for rows in grouped.values() for row in rows})
    return values, [format_bytes(value) for value in values]


def filter_by_max_bytes(grouped, max_bytes):
    filtered = defaultdict(list)
    for direction, rows in grouped.items():
        filtered[direction] = [row for row in rows if row["bytes"] <= max_bytes]
    return {direction: rows for direction, rows in filtered.items() if rows}


def plot_metric(grouped, metric, ylabel, title, output):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.2))
    for direction in sorted(grouped):
        rows = grouped[direction]
        plt.plot([row["bytes"] for row in rows], [row[metric] for row in rows], marker="o", label=direction)

    plt.xscale("log", base=2)
    ticks, labels = byte_ticks(grouped)
    plt.xticks(ticks, labels, rotation=35, ha="right")
    plt.xlabel("bytes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    print(output)


def main():
    args = parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise SystemExit("matplotlib is required to plot the curve: python3 -m pip install matplotlib") from exc

    labels = args.labels.split(",") if args.labels else None
    grouped = load_rows(args.csv, labels=labels, direction_filter=args.direction)
    if not grouped:
        raise SystemExit("no rows found")

    bad_rows = [
        row for rows in grouped.values() for row in rows
        if row["status"] != 0 or row["errors"] != 0
    ]
    if bad_rows:
        raise SystemExit("refuse to plot rows with nonzero status/errors")

    plot_metric(grouped, "bw_GBps", "bw_GBps", "TileXR UDMA P2P bandwidth, rank_size=2", args.output)
    latency_grouped = filter_by_max_bytes(grouped, args.latency_max_bytes)
    if not latency_grouped:
        raise SystemExit(f"no rows found for latency <= {args.latency_max_bytes} bytes")
    plot_metric(latency_grouped, "avg_us", "avg_us",
        f"TileXR UDMA P2P latency, rank_size=2, <= {format_bytes(args.latency_max_bytes)}",
        args.latency_output or default_latency_output(args.output))


if __name__ == "__main__":
    main()
