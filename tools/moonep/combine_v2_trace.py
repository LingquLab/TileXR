#!/usr/bin/env python3

"""Convert MoonEP Combine V2 profile logs to Chrome Trace Event JSON."""

import argparse
import json
import math
import sys
from pathlib import Path


SAMPLE_PREFIX = "COMBINE_V2_SAMPLE"
PROFILE_PREFIX = "COMBINE_V2_PROFILE"
RANK_PERF_PREFIX = "COMBINE_V2_RANK_PERF"
PERF_PREFIX = "COMBINE_V2_PERF"

TIME_POINT_COUNT = 26
METRIC_NAMES = (
    "selection_load_us",
    "selection_select_us",
    "self_route_decode_us",
    "self_copy_us",
    "remote_route_decode_us",
    "remote_descriptor_us",
    "remote_wqe_build_us",
    "remote_submit_us",
)
FULLMESH_TIME_FIELDS = (
    "fm_wqe_build_end",
    "fm_submit_end",
    "fm_cq_success",
    "clos_grant_submit",
    "clos_grant_cq_success",
)
FULLMESH_EVENT_NAMES = (
    "fullmesh_wqe_build_end",
    "fullmesh_submit_end",
    "fullmesh_cq_success",
    "clos_grant_submit",
    "clos_grant_cq_success",
)
FULLMESH_ROUTE_FIELDS = (
    "transport",
    "fm_step",
    "fm_peer",
    "fm_successor",
    "fm_logical_qp",
)


def parse_fields(line):
    fields = {}
    for item in line.strip().split()[1:]:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        fields[key] = value
    return fields


def parse_log(paths, batch_size):
    samples = {}
    profiles = {}
    correctness = {}
    perf_records = []
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.startswith(SAMPLE_PREFIX):
                    fields = parse_fields(line)
                    if int(fields.get("bs", -1)) != batch_size:
                        continue
                    key = (int(fields["iteration"]), int(fields["rank"]))
                    value = float(fields["elapsed_ms"])
                    timing_source = fields.get("timing_source", "acl_event")
                    if timing_source not in ("acl_event", "kernel_profile"):
                        raise ValueError(
                            "unsupported timing_source {} at {}:{}".format(
                                timing_source, path, line_number))
                    sample = {
                        "elapsed_ms": value,
                        "timing_source": timing_source,
                    }
                    if key in samples and samples[key] != sample:
                        raise ValueError(
                            "conflicting sample {} at {}:{}".format(
                                key, path, line_number))
                    samples[key] = sample
                elif line.startswith(PROFILE_PREFIX):
                    fields = parse_fields(line)
                    if int(fields.get("bs", -1)) != batch_size:
                        continue
                    key = (
                        int(fields["iteration"]),
                        int(fields["rank"]),
                        int(fields["core"]),
                    )
                    cycles_per_us = int(fields["cycles_per_us"])
                    points = tuple(int(fields["t{}".format(index)])
                                   for index in range(TIME_POINT_COUNT))
                    metrics = {
                        name: float(fields[name]) for name in METRIC_NAMES
                    }
                    fullmesh = None
                    fullmesh_fields = FULLMESH_ROUTE_FIELDS + \
                        FULLMESH_TIME_FIELDS
                    if any(name in fields for name in fullmesh_fields):
                        missing = [name for name in fullmesh_fields
                                   if name not in fields]
                        if missing:
                            raise ValueError(
                                "incomplete Fullmesh profile fields {} at {}:{}".
                                format(missing, path, line_number))
                        fullmesh = {
                            "profile_version": int(
                                fields.get("profile_version", 0)),
                            "transport": fields["transport"],
                            "step": int(fields["fm_step"]),
                            "peer": int(fields["fm_peer"]),
                            "successor": int(fields["fm_successor"]),
                            "logical_qp": int(fields["fm_logical_qp"]),
                            "time_points": tuple(
                                int(fields[name])
                                for name in FULLMESH_TIME_FIELDS),
                        }
                    record = {
                        "cycles_per_us": cycles_per_us,
                        "points": points,
                        "metrics": metrics,
                        "fullmesh": fullmesh,
                    }
                    if key in profiles and profiles[key] != record:
                        raise ValueError(
                            "conflicting profile {} at {}:{}".format(
                                key, path, line_number))
                    profiles[key] = record
                elif line.startswith(RANK_PERF_PREFIX):
                    fields = parse_fields(line)
                    if int(fields.get("bs", -1)) == batch_size:
                        correctness[int(fields["rank"])] = fields["correctness"]
                elif line.startswith(PERF_PREFIX):
                    fields = parse_fields(line)
                    if int(fields.get("bs", -1)) == batch_size:
                        perf_records.append(fields)
    return samples, profiles, correctness, perf_records


def validate_perf_record(perf_records, batch_size, world_size, topk,
                         hidden_size, experts, reduce_mode):
    if len(perf_records) != 1:
        raise ValueError(
            "expected one matching COMBINE_V2_PERF record, found {}".format(
                len(perf_records)))
    record = perf_records[0]
    expected = {
        "bs": str(batch_size),
        "k": str(topk),
        "h": str(hidden_size),
        "experts": str(experts),
        "dtype": "bf16",
        "ranks": str(world_size),
        "reduce": reduce_mode,
    }
    mismatches = [
        "{}={} (expected {})".format(key, record.get(key, "missing"), value)
        for key, value in expected.items() if record.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            "COMBINE_V2_PERF provenance mismatch: {}".format(
                ", ".join(mismatches)))
    if not record.get("correctness"):
        raise ValueError("COMBINE_V2_PERF provenance mismatch: correctness=missing")
    return record


def nearest_rank_p50_index(count):
    if count <= 0:
        raise ValueError("rank count must be positive")
    return int(math.ceil(0.5 * count)) - 1


def active_core_count(world_size):
    return world_size if world_size <= 8 else 16


def select_ranks(samples, iteration, world_size):
    elapsed = []
    for rank in range(world_size):
        key = (iteration, rank)
        if key not in samples:
            raise ValueError("missing final-iteration sample for rank {}".format(rank))
        elapsed.append((samples[key]["elapsed_ms"], rank))
    elapsed.sort(key=lambda item: (item[0], item[1]))
    selected = (
        ("fastest", elapsed[0]),
        ("p50", elapsed[nearest_rank_p50_index(world_size)]),
        ("slowest", elapsed[-1]),
    )
    return elapsed, selected


def schedule_step_count(world_size):
    if 2 <= world_size <= 8:
        return 1
    if world_size in (16, 32, 64, 128):
        return world_size // 16
    raise ValueError("unsupported Combine V2 world size {}".format(world_size))


def stage_ranges(world_size, reduce_mode="disabled"):
    step_count = schedule_step_count(world_size)
    ranges = [
        ("init", "setup", 0, 1),
        ("prepare", "setup", 1, 2),
        ("full_sync_enter", "sync", 2, 3),
        ("full_sync_submit", "sync", 3, 4),
        ("full_sync_receive", "sync", 4, 5),
        ("full_sync_core_barrier", "sync", 5, 6),
    ]
    previous = 6
    for step in range(step_count):
        send_end = 7 + step * 2
        ready_end = send_end + 1
        ranges.append(("step{}_send".format(step), "send", previous, send_end))
        ranges.append(("step{}_wait".format(step), "wait", send_end, ready_end))
        previous = ready_end
    ranges.extend((
        ("step_loop_finalize", "finalize", previous, 23),
        ("inbound_wait", "wait", 23, 24),
        ("reduce_hidden" if reduce_mode == "enabled" else
            "finalize_no_reduce", "reduce" if reduce_mode == "enabled" else
            "finalize", 24, 25),
    ))
    return ranges


def complete_event(name, category, pid, tid, start_us, end_us, args=None):
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": round(start_us, 6),
        "dur": round(max(0.0, end_us - start_us), 6),
        "args": args or {},
    }


def metadata_event(name, pid, args, tid=0):
    return {
        "name": name,
        "ph": "M",
        "pid": pid,
        "tid": tid,
        "args": args,
    }


def validate_fullmesh_profile(record, rank, core, world_size):
    fullmesh = record["fullmesh"]
    if fullmesh is None:
        return
    transport = fullmesh["transport"]
    route = (
        fullmesh["step"],
        fullmesh["peer"],
        fullmesh["successor"],
        fullmesh["logical_qp"],
    )
    time_points = fullmesh["time_points"]
    if transport == "none":
        if route != (-1, -1, -1, -1) or any(time_points):
            raise ValueError(
                "rank {} core {} has nonzero non-Fullmesh diagnostics".
                format(rank, core))
        return
    if transport != "fullmesh" or fullmesh["profile_version"] < 5:
        raise ValueError(
            "rank {} core {} has unsupported Fullmesh profile schema".
            format(rank, core))
    local_rank_size = min(world_size, 8)
    if not 0 <= fullmesh["step"] < schedule_step_count(world_size) or \
            not 0 <= fullmesh["peer"] < world_size or \
            not 0 <= fullmesh["successor"] < world_size or \
            fullmesh["peer"] == rank or \
            rank // local_rank_size != fullmesh["peer"] // local_rank_size or \
            fullmesh["logical_qp"] != \
            32 + fullmesh["peer"] % local_rank_size:
        raise ValueError(
            "rank {} core {} has invalid Fullmesh route {}".
            format(rank, core, route))
    fullmesh_points = time_points[:3]
    grant_points = time_points[3:]
    grant_absent = grant_points == (0, 0)
    grant_valid = all(point > 0 for point in grant_points) and \
        fullmesh_points[-1] < grant_points[0] < grant_points[1]
    if any(point <= 0 for point in fullmesh_points) or any(
            fullmesh_points[index] >= fullmesh_points[index + 1]
            for index in range(len(fullmesh_points) - 1)) or \
            (not grant_absent and not grant_valid):
        raise ValueError(
            "rank {} core {} has invalid Fullmesh event order".
            format(rank, core))


def append_fullmesh_events(events, record, rank, core, iteration,
                           origin, divisor, pid, tid):
    fullmesh = record["fullmesh"]
    if fullmesh is None or fullmesh["transport"] != "fullmesh":
        return
    grant_absent = fullmesh["time_points"][3:] == (0, 0)
    args = {
        "iteration": iteration,
        "rank": rank,
        "core": core,
        "data_transport": "fullmesh",
        "grant_transport": (
            "none" if grant_absent else
            "local" if fullmesh["successor"] == rank else "clos"),
        "step": fullmesh["step"],
        "peer": fullmesh["peer"],
        "successor": fullmesh["successor"],
        "logical_qp": fullmesh["logical_qp"],
    }
    for index, (name, point) in enumerate(zip(
            FULLMESH_EVENT_NAMES, fullmesh["time_points"])):
        if point == 0:
            continue
        event_args = dict(args)
        if name == "clos_grant_cq_success" and \
                fullmesh["successor"] == rank:
            event_args["completion"] = "local_publication"
        events.append({
            "name": name,
            "cat": "fullmesh" if index < 3 else "grant",
            "ph": "i",
            "s": "t",
            "pid": pid,
            "tid": tid,
            "ts": round((point - origin) / divisor, 6),
            "args": event_args,
        })


def build_trace(samples, profiles, correctness, perf_record, batch_size,
                world_size, iteration, topk, hidden_size, experts, reduce_mode):
    elapsed_order, selected = select_ranks(samples, iteration, world_size)
    timing_sources = {
        samples[(iteration, rank)]["timing_source"]
        for rank in range(world_size)
    }
    if len(timing_sources) != 1:
        raise ValueError("final-iteration samples use mixed timing sources")
    timing_source = next(iter(timing_sources))
    for rank in range(world_size):
        if rank not in correctness:
            raise ValueError("rank {} correctness is missing".format(rank))

    selected_rows = []
    events = []
    expected_cores = set(range(active_core_count(world_size)))
    for role_index, (role, (elapsed_ms, rank)) in enumerate(selected):
        rank_records = {
            core: profiles[(iteration, rank, core)]
            for core in expected_cores
            if (iteration, rank, core) in profiles
        }
        if set(rank_records) != expected_cores:
            missing = sorted(expected_cores - set(rank_records))
            raise ValueError(
                "rank {} final iteration missing profile cores {}".format(
                    rank, missing))
        divisors = {record["cycles_per_us"] for record in rank_records.values()}
        if len(divisors) != 1 or next(iter(divisors)) <= 0:
            raise ValueError("rank {} has inconsistent cycle divisors".format(rank))
        divisor = next(iter(divisors))
        rank_origin = min(record["points"][0] for record in rank_records.values())
        critical_core = max(
            rank_records,
            key=lambda core: (
                rank_records[core]["points"][-1] -
                rank_records[core]["points"][0],
                -core,
            ),
        )
        critical_record = rank_records[critical_core]
        critical_points = critical_record["points"]
        critical_stages = {
            name: round(
                (critical_points[end] - critical_points[begin]) / divisor, 6)
            for name, _category, begin, end in stage_ranges(
                world_size, reduce_mode)
        }
        pid = role_index + 1
        events.append(metadata_event("process_name", pid, {
            "name": "{} rank {} ({:.6f} ms)".format(role, rank, elapsed_ms),
        }))
        events.append(metadata_event("process_sort_index", pid, {
            "sort_index": role_index,
        }))
        selected_rows.append({
            "role": role,
            "rank": rank,
            "elapsed_ms": elapsed_ms,
            "timing_source": timing_source,
            "correctness": correctness[rank],
            "sorted_position": next(
                index + 1 for index, item in enumerate(elapsed_order)
                if item[1] == rank),
            "critical_core": critical_core,
            "profile_total_us": round(
                (critical_points[-1] - critical_points[0]) / divisor, 6),
            "critical_core_stages_us": critical_stages,
            "critical_core_metrics_us": critical_record["metrics"],
        })
        for core in sorted(rank_records):
            record = rank_records[core]
            validate_fullmesh_profile(record, rank, core, world_size)
            points = record["points"]
            if any(point <= 0 for point in points):
                raise ValueError(
                    "rank {} core {} has non-positive timestamps".format(rank, core))
            if any(points[index] > points[index + 1]
                   for index in range(len(points) - 1)):
                raise ValueError(
                    "rank {} core {} has non-monotonic timestamps".format(rank, core))
            tid = core + 1
            events.append(metadata_event("thread_name", pid, {
                "name": "AIV core {}".format(core),
            }, tid=tid))
            events.append(metadata_event("thread_sort_index", pid, {
                "sort_index": core,
            }, tid=tid))
            start_us = (points[0] - rank_origin) / divisor
            end_us = (points[-1] - rank_origin) / divisor
            total_args = {
                "role": role,
                "rank": rank,
                "core": core,
                "iteration": iteration,
                "selection_elapsed_ms": elapsed_ms,
                "timing_source": timing_source,
                "correctness": correctness[rank],
                "reduce": reduce_mode,
            }
            total_args[
                "acl_event_elapsed_ms" if timing_source == "acl_event" else
                "kernel_profile_elapsed_ms"] = elapsed_ms
            total_args.update(record["metrics"])
            events.append(complete_event(
                "combine_v2_reduce" if reduce_mode == "enabled" else
                "combine_v2_no_reduce", "kernel", pid, tid,
                start_us, end_us, total_args))
            for name, category, begin, end in stage_ranges(
                    world_size, reduce_mode):
                events.append(complete_event(
                    name, category, pid, tid,
                    (points[begin] - rank_origin) / divisor,
                    (points[end] - rank_origin) / divisor,
                    {"rank": rank, "core": core, "iteration": iteration}))
            append_fullmesh_events(
                events, record, rank, core, iteration, rank_origin,
                divisor, pid, tid)

    sorted_rows = [
        {"sorted_position": index + 1, "rank": rank, "elapsed_ms": elapsed}
        for index, (elapsed, rank) in enumerate(elapsed_order)
    ]
    other_data = {
        "schema": "tilexr_moonep_combine_v2_chrome_trace.v1",
        "shape": {
            "world_size": world_size,
            "batch_size": batch_size,
            "topk": topk,
            "hidden_size": hidden_size,
            "dtype": "bf16",
            "experts": experts,
            "schedule_steps": schedule_step_count(world_size),
        },
        "iteration": iteration,
        "timing_source": timing_source,
        "reduce": reduce_mode,
        "correctness": perf_record["correctness"],
        "rank_correctness": [
            {"rank": rank, "correctness": correctness[rank]}
            for rank in range(world_size)
        ],
        "selection_method": (
            "ascending final-iteration {} elapsed time; fastest=min; "
            "p50=nearest-rank ceil(0.5*N); slowest=max; ties by rank".format(
                "ACL Event" if timing_source == "acl_event" else
                "slowest-core kernel profile")),
        "clock_alignment": (
            "each selected NPU rank is normalized independently to its earliest "
            "AIV core t0; cross-NPU absolute cycle offsets are not compared"),
        "profile_warning": (
            "profile instrumentation perturbs timing; use for attribution, not "
            "uninstrumented production latency"),
        "rank_order": sorted_rows,
        "selected_ranks": selected_rows,
    }
    return {
        "displayTimeUnit": "ms",
        "traceEvents": events,
        "otherData": other_data,
    }


def edge_stage_ranges(world_size, reduce_mode):
    names = {
        "init": "Init",
        "prepare": "Prepare",
        "full_sync_enter": "Full sync enter",
        "full_sync_submit": "Full sync submit",
        "full_sync_receive": "Full sync receive",
        "full_sync_core_barrier": "Full sync core barrier",
        "step_loop_finalize": "Final CQ",
        "inbound_wait": "Inbound done",
        "finalize_no_reduce": "Finalize (no reduce)",
        "reduce_hidden": "Reduce hidden",
    }
    result = []
    for name, category, begin, end in stage_ranges(world_size, reduce_mode):
        if name.startswith("step") and name.endswith("_send"):
            display_name = "Step {} send".format(name[4:-5])
        elif name.startswith("step") and name.endswith("_wait"):
            display_name = "Step {} wait".format(name[4:-5])
        else:
            display_name = names[name]
        result.append((display_name, category, begin, end))
    return result


def build_edge_trace(role, elapsed_ms, rank, samples, profiles, correctness,
                     batch_size, world_size, iteration, topk, hidden_size,
                     experts, reduce_mode, host):
    core_count = active_core_count(world_size)
    rank_records = {
        core: profiles[(iteration, rank, core)] for core in range(core_count)
        if (iteration, rank, core) in profiles
    }
    if len(rank_records) != core_count:
        missing = sorted(set(range(core_count)) - set(rank_records))
        raise ValueError(
            "rank {} final iteration missing profile cores {}".format(
                rank, missing))
    divisors = {record["cycles_per_us"] for record in rank_records.values()}
    if len(divisors) != 1 or next(iter(divisors)) <= 0:
        raise ValueError("rank {} has inconsistent cycle divisors".format(rank))
    divisor = next(iter(divisors))
    timing_source = samples[(iteration, rank)]["timing_source"]
    timing_label = "ACL" if timing_source == "acl_event" else "PROFILE"
    origin = min(record["points"][0] for record in rank_records.values())
    events = [
        metadata_event("process_name", rank, {
            "name": "rank {} | {} | {} {:.6f} ms | {} device {}".format(
                rank, role, timing_label, elapsed_ms, host, rank % 8),
        }),
        metadata_event("process_sort_index", rank, {"sort_index": 0}),
    ]
    for core in sorted(rank_records):
        record = rank_records[core]
        validate_fullmesh_profile(record, rank, core, world_size)
        points = record["points"]
        if any(point <= 0 for point in points) or any(
                points[index] > points[index + 1]
                for index in range(len(points) - 1)):
            raise ValueError(
                "rank {} core {} has invalid timestamps".format(rank, core))
        events.append(metadata_event("thread_name", rank, {
            "name": "AIV core {}".format(core),
        }, tid=core))
        events.append(metadata_event("thread_sort_index", rank, {
            "sort_index": core,
        }, tid=core))
        for name, category, begin, end in edge_stage_ranges(
                world_size, reduce_mode):
            start_us = (points[begin] - origin) / divisor
            end_us = (points[end] - origin) / divisor
            events.append(complete_event(
                name, category, rank, core, start_us, end_us, {
                    "iteration": iteration,
                    "rank": rank,
                    "core": core,
                    "start_point": "t{}".format(begin),
                    "end_point": "t{}".format(end),
                    "duration_us": round(max(0.0, end_us - start_us), 6),
                }))
        events.append({
            "name": "Core accumulated metrics",
            "cat": "metrics",
            "ph": "i",
            "s": "t",
            "ts": round((points[-1] - origin) / divisor, 6),
            "pid": rank,
            "tid": core,
            "args": record["metrics"],
        })
        append_fullmesh_events(
            events, record, rank, core, iteration, origin,
            divisor, rank, core)
    edge_trace = {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "format": "Chrome Trace Event Format",
            "selection": role,
            "selection_rule": "{} rank by iteration-{} {} elapsed_ms".format(
                role, iteration, timing_source),
            "iteration": iteration,
            "rank": rank,
            "host": host,
            "device": rank % 8,
            "selection_elapsed_ms": elapsed_ms,
            "timing_source": timing_source,
            "bs": batch_size,
            "k": topk,
            "h": hidden_size,
            "dtype": "bf16",
            "experts": experts,
            "active_cores": core_count,
            "profile_cycles_per_us": divisor,
            "reduce": reduce_mode,
            "correctness": correctness[rank],
        },
    }
    edge_trace["metadata"][
        "acl_event_elapsed_ms" if timing_source == "acl_event" else
        "kernel_profile_elapsed_ms"] = elapsed_ms
    return edge_trace


def write_edge_traces(output_dir, prefix, host, samples, profiles,
                      correctness, batch_size, world_size, iteration, topk,
                      hidden_size, experts, reduce_mode):
    _elapsed_order, selected = select_ranks(samples, iteration, world_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for role, (elapsed_ms, rank) in selected:
        trace = build_edge_trace(
            role, elapsed_ms, rank, samples, profiles, correctness,
            batch_size, world_size, iteration, topk, hidden_size, experts,
            reduce_mode, host)
        path = output_dir / (
            "{}_iteration{}_{}_rank{}_edge_trace.json".format(
                prefix, iteration, role, rank))
        path.write_text(json.dumps(trace, indent=4) + "\n", encoding="utf-8")
        outputs.append(path)
        print("COMBINE_V2_EDGE_TRACE role={} rank={} elapsed_ms={:.6f} "
              "output={}".format(role, rank, elapsed_ms, path))
    return outputs


def write_summary(path, trace):
    data = trace["otherData"]
    summary = {
        "schema": "tilexr_moonep_combine_v2_trace_summary.v1",
        "shape": data["shape"],
        "iteration": data["iteration"],
        "timing_source": data["timing_source"],
        "reduce": data["reduce"],
        "correctness": data["correctness"],
        "rank_correctness": data["rank_correctness"],
        "selection_method": data["selection_method"],
        "rank_order": data["rank_order"],
        "selected_ranks": data["selected_ranks"],
        "clock_alignment": data["clock_alignment"],
        "profile_warning": data["profile_warning"],
    }
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--split-output-dir", type=Path)
    parser.add_argument("--prefix", default="combine_v2_reduce")
    parser.add_argument("--host", default="unknown")
    parser.add_argument("--reduce", choices=("enabled", "disabled"),
                        default="disabled")
    parser.add_argument("--bs", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--iteration", type=int,
                        help="timed iteration to export; default: last sample")
    parser.add_argument("--topk", default=16, type=int)
    parser.add_argument("--hidden-size", default=3584, type=int)
    parser.add_argument("--experts", required=True, type=int)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output is None and args.split_output_dir is None:
        raise ValueError("one of --output or --split-output-dir is required")
    if args.bs <= 0 or args.world_size <= 0 or args.topk <= 0 or \
            args.hidden_size <= 0 or args.experts <= 0:
        raise ValueError("shape and world-size arguments must be positive")
    missing = [str(path) for path in args.logs if not path.is_file()]
    if missing:
        raise ValueError("input logs do not exist: {}".format(", ".join(missing)))
    samples, profiles, correctness, perf_records = parse_log(args.logs, args.bs)
    if not samples:
        raise ValueError("no matching Combine V2 samples found")
    perf_record = validate_perf_record(
        perf_records, args.bs, args.world_size, args.topk,
        args.hidden_size, args.experts, args.reduce)
    iteration = args.iteration
    if iteration is None:
        iteration = max(key[0] for key in samples)
    trace = build_trace(
        samples, profiles, correctness, perf_record, args.bs, args.world_size,
        iteration, args.topk, args.hidden_size, args.experts, args.reduce)
    summary_path = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        summary_path = args.summary or args.output.with_name(
            args.output.stem + "_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_summary(summary_path, trace)
    if args.split_output_dir is not None:
        write_edge_traces(
            args.split_output_dir, args.prefix, args.host, samples, profiles,
            correctness, args.bs, args.world_size, iteration, args.topk,
            args.hidden_size, args.experts, args.reduce)
    for row in trace["otherData"]["selected_ranks"]:
        print("COMBINE_V2_TRACE_SELECTION role={role} rank={rank} "
              "sorted_position={sorted_position} elapsed_ms={elapsed_ms:.6f}".format(
                  **row))
    if args.output is not None:
        print("COMBINE_V2_TRACE output={} summary={} iteration={} reduce={}".format(
            args.output, summary_path, iteration, args.reduce))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, TypeError, ValueError) as error:
        print("error: {}".format(error), file=sys.stderr)
        sys.exit(2)
