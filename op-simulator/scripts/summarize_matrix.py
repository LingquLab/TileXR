#!/usr/bin/env python3
"""Summarize op-simulator matrix results and trace_core0.json files.

The matrix TSV is intentionally simple. A row should include `variant`,
`status`, and timing columns such as `run_time_s`; if `trace_json` is present,
the script also parses the trace and reports barrier/span/category statistics.
"""

import argparse
import csv
import json
import re
from pathlib import Path


DEFAULT_STATUS_OK = "SUCCESS"
TRACE_NAME = "trace_core0.json"


def parse_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def maybe_number(value):
    int_value = parse_int(value)
    if int_value is not None and str(int_value) == str(value):
        return int_value
    float_value = parse_float(value)
    if float_value is not None:
        return float_value
    return value


def load_matrix(summary_tsv):
    with summary_tsv.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def infer_key_columns(rows, requested):
    if requested:
        return [col.strip() for col in requested.split(",") if col.strip()]
    if rows and all(col in rows[0] for col in ("ep", "bs")):
        return ["ep", "bs"]
    ignored = {
        "variant",
        "kernel",
        "status",
        "run_time_s",
        "wall_s",
        "hw_sim_0_s",
        "hw_sim_1_s",
        "trace_json",
        "case_dir",
    }
    return [col for col in rows[0].keys() if col not in ignored] if rows else []


def read_rank_stats(record_log):
    if not record_log.exists():
        return None
    text = record_log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(
        r"RankStats ep=(\d+) totalExperts=(\d+) expertsPerRank=(\d+) bs=(\d+) topK=(\d+) "
        r"blockCntPerToken=(\d+) hAlignWinSize=(\d+) nonZeroRanks=(\d+) maxRankCount=(\d+)",
        text,
    )
    if not matches:
        return None
    ep, total, per_rank, bs, topk, block_cnt, win_size, non_zero, max_count = matches[-1]
    return {
        "ep": int(ep),
        "total_experts": int(total),
        "experts_per_rank": int(per_rank),
        "bs": int(bs),
        "top_k": int(topk),
        "block_cnt_per_token": int(block_cnt),
        "h_align_win_size": int(win_size),
        "non_zero_ranks": int(non_zero),
        "max_rank_count": int(max_count),
    }


def find_case_dir(row, result_dir):
    case_dir = row.get("case_dir")
    if case_dir:
        candidate = Path(case_dir)
        if candidate.exists():
            return candidate
        candidate = result_dir / candidate.name
        if candidate.exists():
            return candidate
    if {"ep", "bs", "variant"}.issubset(row):
        candidate = result_dir / f"ep{row['ep']}_bs{row['bs']}_{row['variant']}"
        if candidate.exists():
            return candidate
    return None


def resolve_trace(row, result_dir):
    trace = row.get("trace_json")
    if trace:
        candidate = Path(trace)
        if candidate.exists():
            return candidate
        case_dir = find_case_dir(row, result_dir)
        if case_dir is not None:
            local_trace = case_dir / "report" / TRACE_NAME
            if local_trace.exists():
                return local_trace
    case_dir = find_case_dir(row, result_dir)
    if case_dir is None:
        return None
    candidate = case_dir / "report" / TRACE_NAME
    return candidate if candidate.exists() else None


def top_items(mapping, limit):
    return [
        {"name": key, "value": value}
        for key, value in sorted(mapping.items(), key=lambda item: item[1], reverse=True)[:limit]
    ]


def trace_stats(trace_path, top_limit):
    if trace_path is None or not trace_path.exists():
        return None
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    min_ts = None
    max_ts = None
    x_events = 0
    cat_durations = {}
    name_counts = {}
    bar_events = 0
    barrier_pipes = {}
    events_by_cat = {}

    for event in data:
        name = str(event.get("name", ""))
        cat = str(event.get("cat", ""))
        ph = event.get("ph")
        name_counts[name] = name_counts.get(name, 0) + 1

        if name == "BAR" or "BAR" in cat:
            bar_events += 1
            extra_info = str(event.get("args", {}).get("extra_info", ""))
            pipe_match = re.search(r"PIPE:([^,\s]+)", extra_info)
            pipe_name = pipe_match.group(1) if pipe_match else "UNKNOWN"
            barrier_pipes[pipe_name] = barrier_pipes.get(pipe_name, 0) + 1

        if ph != "X":
            continue

        ts = event.get("ts")
        dur = event.get("dur", 0)
        if not isinstance(ts, (int, float)):
            continue

        x_events += 1
        min_ts = ts if min_ts is None else min(min_ts, ts)
        max_ts = ts + dur if max_ts is None else max(max_ts, ts + dur)
        cat_durations[cat] = cat_durations.get(cat, 0) + dur
        events_by_cat.setdefault(cat, []).append((ts, dur))

    gap_by_cat = {}
    for cat, events in events_by_cat.items():
        events.sort(key=lambda item: item[0])
        last_end = None
        total_gap = 0
        for ts, dur in events:
            if last_end is not None and ts > last_end:
                total_gap += ts - last_end
            last_end = max(last_end or ts, ts + dur)
        if total_gap:
            gap_by_cat[cat] = total_gap

    return {
        "trace_json": str(trace_path),
        "events": len(data),
        "x_events": x_events,
        "span_ticks": None if min_ts is None or max_ts is None else max_ts - min_ts,
        "bar_events": bar_events,
        "cat_durations": cat_durations,
        "barrier_pipes": top_items(barrier_pipes, top_limit),
        "top_cat_durations": top_items(cat_durations, top_limit),
        "top_cat_gaps": top_items(gap_by_cat, top_limit),
        "top_event_names": top_items(name_counts, top_limit),
    }


def normalize_row(row, result_dir, top_limit):
    case_dir = find_case_dir(row, result_dir)
    trace_path = resolve_trace(row, result_dir)
    normalized = {key: maybe_number(value) for key, value in row.items()}
    normalized["run_time_s"] = parse_float(row.get("run_time_s"))
    normalized["wall_s"] = parse_float(row.get("wall_s"))
    normalized["hw_sim_s"] = [
        parse_float(row.get("hw_sim_0_s")),
        parse_float(row.get("hw_sim_1_s")),
    ]
    normalized["case_dir"] = str(case_dir) if case_dir is not None else row.get("case_dir")
    normalized["trace_json"] = str(trace_path) if trace_path is not None else None
    normalized["rank_stats"] = read_rank_stats(case_dir / "record.log") if case_dir is not None else None
    normalized["trace_stats"] = trace_stats(trace_path, top_limit)
    return normalized


def key_tuple(row, key_columns):
    return tuple(str(row.get(col, "")) for col in key_columns)


def cat_duration_map(row):
    stats = row.get("trace_stats") or {}
    if "cat_durations" in stats:
        return stats["cat_durations"]
    return {item["name"]: item["value"] for item in stats.get("top_cat_durations", [])}


def compare_rows(rows, key_columns, baseline_variant, candidate_variant, success_status):
    grouped = {}
    for row in rows:
        grouped.setdefault(key_tuple(row, key_columns), {})[str(row.get("variant"))] = row

    comparisons = []
    for key, variants in sorted(grouped.items()):
        baseline = variants.get(baseline_variant)
        candidate = variants.get(candidate_variant)
        base_time = baseline.get("run_time_s") if baseline else None
        cand_time = candidate.get("run_time_s") if candidate else None
        baseline_success = baseline is not None and baseline.get("status") == success_status
        candidate_success = candidate is not None and candidate.get("status") == success_status
        delta = None
        improvement = None
        if baseline_success and candidate_success and base_time is not None and cand_time is not None and base_time != 0:
            delta = cand_time - base_time
            improvement = (base_time - cand_time) / base_time * 100.0

        base_trace = (baseline or {}).get("trace_stats") or {}
        cand_trace = (candidate or {}).get("trace_stats") or {}
        base_cat = cat_duration_map(baseline or {})
        cand_cat = cat_duration_map(candidate or {})
        cat_delta = {
            cat: cand_cat.get(cat, 0) - base_cat.get(cat, 0)
            for cat in sorted(set(base_cat) | set(cand_cat))
        }

        comparisons.append({
            "key": dict(zip(key_columns, key)),
            "baseline_variant": baseline_variant,
            "candidate_variant": candidate_variant,
            "baseline_status": None if baseline is None else baseline.get("status"),
            "candidate_status": None if candidate is None else candidate.get("status"),
            "baseline_run_time_s": base_time,
            "candidate_run_time_s": cand_time,
            "delta_s_candidate_minus_baseline": delta,
            "improvement_percent": improvement,
            "baseline_success": baseline_success,
            "candidate_success": candidate_success,
            "trace_delta": {
                "span_ticks": subtract_optional(cand_trace.get("span_ticks"), base_trace.get("span_ticks")),
                "bar_events": subtract_optional(cand_trace.get("bar_events"), base_trace.get("bar_events")),
                "top_cat_durations": cat_delta,
            },
        })
    return comparisons


def subtract_optional(lhs, rhs):
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def format_number(value, digits=1):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path, key_columns, comparisons, baseline_variant, candidate_variant):
    headers = key_columns + [
        f"{baseline_variant} time(s)",
        f"{candidate_variant} time(s)",
        "delta(s)",
        "improvement",
        "bar delta",
        "span delta(ticks)",
        "status",
    ]
    lines = [
        "# Op-simulator Matrix Summary",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for item in comparisons:
        key_values = [str(item["key"].get(col, "")) for col in key_columns]
        improvement = item["improvement_percent"]
        status = f"{item['baseline_status']} / {item['candidate_status']}"
        row = key_values + [
            format_number(item["baseline_run_time_s"]),
            format_number(item["candidate_run_time_s"]),
            format_number(item["delta_s_candidate_minus_baseline"]),
            "" if improvement is None else f"{improvement:.2f}%",
            format_number(item["trace_delta"].get("bar_events"), 0),
            format_number(item["trace_delta"].get("span_ticks"), 0),
            status,
        ]
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", help="Directory containing matrix_summary.tsv or copied result directory.")
    parser.add_argument("--summary-tsv", help="Matrix TSV path. Defaults to <result_root>/matrix_summary.tsv.")
    parser.add_argument("--output-json", help="Output JSON path. Defaults to <result_root>/comparison_summary.json.")
    parser.add_argument("--output-md", help="Output Markdown path. Defaults to <result_root>/comparison_summary.md.")
    parser.add_argument("--key", default="", help="Comma-separated comparison key columns. Defaults to ep,bs if present.")
    parser.add_argument("--baseline-variant", default="baseline")
    parser.add_argument("--candidate-variant", default="final")
    parser.add_argument("--success-status", default=DEFAULT_STATUS_OK)
    parser.add_argument("--top-limit", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    summary_tsv = Path(args.summary_tsv).resolve() if args.summary_tsv else result_root / "matrix_summary.tsv"
    if not summary_tsv.exists():
        candidates = sorted(result_root.glob("*/matrix_summary.tsv"))
        if len(candidates) == 1:
            summary_tsv = candidates[0]
        else:
            raise FileNotFoundError(summary_tsv)

    result_dir = summary_tsv.parent
    raw_rows = load_matrix(summary_tsv)
    key_columns = infer_key_columns(raw_rows, args.key)
    rows = [normalize_row(row, result_dir, args.top_limit) for row in raw_rows]
    comparisons = compare_rows(
        rows,
        key_columns,
        args.baseline_variant,
        args.candidate_variant,
        args.success_status,
    )

    output = {
        "input": {
            "result_root": str(result_root),
            "summary_tsv": str(summary_tsv),
            "key_columns": key_columns,
            "baseline_variant": args.baseline_variant,
            "candidate_variant": args.candidate_variant,
            "success_status": args.success_status,
        },
        "rows": rows,
        "comparisons": comparisons,
    }

    output_json = Path(args.output_json).resolve() if args.output_json else result_root / "comparison_summary.json"
    output_md = Path(args.output_md).resolve() if args.output_md else result_root / "comparison_summary.md"
    output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_md, key_columns, comparisons, args.baseline_variant, args.candidate_variant)
    print(output_json)


if __name__ == "__main__":
    raise SystemExit(main())
