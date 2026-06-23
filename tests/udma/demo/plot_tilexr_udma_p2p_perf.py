#!/usr/bin/env python3
"""Plot TileXR UDMA P2P performance CSV files."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from glob import glob


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
    parser.add_argument("--series-by", default="auto", choices=["auto", "label", "transport", "traffic", "block_dim"],
        help="series grouping for new CSVs; default uses transport+traffic+block_dim")
    parser.add_argument("--metric", default="bw_GBps", choices=["bw_GBps", "per_flow_bw_GBps"],
        help="bandwidth metric to plot; default plots aggregate bandwidth")
    return parser.parse_args()


def infer_label(path):
    text = str(path)
    if "direct_urma" in text:
        return "direct_urma"
    if "memory" in text:
        return "memory"
    return Path(path).parent.name or Path(path).stem


def expand_paths(paths):
    expanded = []
    for path in paths:
        matches = sorted(glob(path))
        expanded.extend(matches if matches else [path])
    return expanded


def row_value(row, key, default):
    value = row.get(key)
    return default if value is None or value == "" else value


def build_series(row, path, fallback_label, series_by):
    transport = row_value(row, "transport", fallback_label)
    traffic = row_value(row, "traffic", "unidir")
    block_dim = row_value(row, "block_dim", "1")
    if series_by == "label":
        return fallback_label
    if series_by == "transport":
        return transport
    if series_by == "traffic":
        return traffic
    if series_by == "block_dim":
        return f"bd={block_dim}"
    return f"{transport} {traffic} bd={block_dim}"


def load_rows(paths, labels=None, direction_filter=None, series_by="auto"):
    if labels is not None and len(labels) != len(paths):
        raise SystemExit("--labels count must match input CSV count")
    merged = {}
    for index, path in enumerate(paths):
        label = labels[index] if labels is not None else infer_label(path)
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                direction = row_value(row, "direction", "")
                if direction_filter is not None and direction != direction_filter:
                    continue
                series = label if labels is not None else build_series(row, path, label, series_by)
                key = (series, int(row["bytes"]))
                merged[key] = {
                    "series": series,
                    "direction": direction,
                    "transport": row_value(row, "transport", label),
                    "traffic": row_value(row, "traffic", "unidir"),
                    "block_dim": int(row_value(row, "block_dim", "1")),
                    "bytes": int(row["bytes"]),
                    "avg_us": float(row["avg_us"]),
                    "bw_GBps": float(row["bw_GBps"]),
                    "per_flow_bw_GBps": float(row_value(row, "per_flow_bw_GBps", row["bw_GBps"])),
                    "src": row_value(row, "src", ""),
                    "dst": row_value(row, "dst", ""),
                    "ranks": row_value(row, "ranks", ""),
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
    plt.xlabel("message size")
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

    paths = expand_paths(args.csv)
    labels = args.labels.split(",") if args.labels else None
    grouped = load_rows(paths, labels=labels, direction_filter=args.direction, series_by=args.series_by)
    if not grouped:
        raise SystemExit("no rows found")

    bad_rows = [
        row for rows in grouped.values() for row in rows
        if row["status"] != 0 or row["errors"] != 0
    ]
    if bad_rows:
        raise SystemExit("refuse to plot rows with nonzero status/errors")

    metric_label = "aggregate bw_GBps" if args.metric == "bw_GBps" else "per-flow bw_GBps"
    plot_metric(grouped, args.metric, metric_label, "TileXR P2P bandwidth, rank_size=2", args.output)
    latency_grouped = filter_by_max_bytes(grouped, args.latency_max_bytes)
    if not latency_grouped:
        raise SystemExit(f"no rows found for latency <= {args.latency_max_bytes} bytes")
    plot_metric(latency_grouped, "avg_us", "avg_us",
        f"TileXR UDMA P2P latency, rank_size=2, <= {format_bytes(args.latency_max_bytes)}",
        args.latency_output or default_latency_output(args.output))


if __name__ == "__main__":
    main()
