#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path


TRACE_MAGIC = 0x47545243
TRACE_VERSION = 3
LEGACY_TRACE_VERSION = 1
TRACE_BYTES = 128 * 1024 * 1024
HEADER_BYTES = 4096
MAX_ITERATIONS = 50
MAX_CORES = 64
SEND_WORKER_COUNT = 32
LANE_COUNT = 16
PHASE_COUNT = 5
TRACE_V2_PHASE_COUNT = 6
TRACE_V3_PHASE_COUNT = 8
CURRENT_PHASE_COUNT = TRACE_V3_PHASE_COUNT
SPAN_BYTES = 16
CACHE_LINE_BYTES = 128
TASK_FORMAT = "<QQiI"
TASK_BYTES = struct.calcsize(TASK_FORMAT)
HEADER_FORMAT = "<8I4Q"
TASK_BASE_OFFSET = HEADER_BYTES + MAX_ITERATIONS * MAX_CORES * CACHE_LINE_BYTES
NO_QP = 0xFFFFFFFF
ITERATION_GAP_US = 1.0

PHASE_NAMES = (
    "self-copy",
    "send-put-signal",
    "send-quiet",
    "receive-wait",
    "receive-copy",
    "credit-wait",
    "sdma-submit",
    "sdma-wait",
)


def kernel_span_offset(iteration, core):
    return HEADER_BYTES + (iteration * MAX_CORES + core) * CACHE_LINE_BYTES


def task_span_offset(
        iteration, core, group, pass_index, phase, group_count, pass_count,
        phase_count=CURRENT_PHASE_COUNT):
    raw_core_bytes = group_count * pass_count * phase_count * TASK_BYTES
    core_bytes = (raw_core_bytes + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1)
    core_index = iteration * MAX_CORES + core
    task_index = ((group * pass_count + pass_index) * phase_count) + phase
    return TASK_BASE_OFFSET + core_index * core_bytes + task_index * TASK_BYTES


def layout_bytes(
        iteration_count, group_count, pass_count,
        phase_count=CURRENT_PHASE_COUNT):
    raw_core_bytes = group_count * pass_count * phase_count * TASK_BYTES
    core_bytes = (raw_core_bytes + CACHE_LINE_BYTES - 1) & ~(CACHE_LINE_BYTES - 1)
    return TASK_BASE_OFFSET + iteration_count * MAX_CORES * core_bytes


def _read_span(data, offset, label):
    if offset < 0 or offset + SPAN_BYTES > len(data):
        raise ValueError(f"span offset out of range for {label}: {offset}")
    begin, end = struct.unpack_from("<QQ", data, offset)
    if begin == 0 and end == 0:
        return None
    if begin == 0 or end == 0:
        raise ValueError(f"incomplete {label}: begin={begin} end={end}")
    if end < begin:
        raise ValueError(f"invalid {label}: begin={begin} end={end}")
    return begin, end


def _read_task(data, offset, label):
    if offset < 0 or offset + TASK_BYTES > len(data):
        raise ValueError(f"task offset out of range for {label}: {offset}")
    begin, end, peer, qp = struct.unpack_from(TASK_FORMAT, data, offset)
    if begin == 0 and end == 0:
        return None
    if begin == 0 or end == 0:
        raise ValueError(f"incomplete {label}: begin={begin} end={end}")
    if end < begin:
        raise ValueError(f"invalid {label}: begin={begin} end={end}")
    return begin, end, peer, qp


def read_rank_trace(path):
    path = Path(path)
    file_size = path.stat().st_size
    if file_size != TRACE_BYTES:
        raise ValueError(f"invalid trace size {file_size} in {path}, expected {TRACE_BYTES}")
    with path.open("rb") as stream:
        header_data = stream.read(struct.calcsize(HEADER_FORMAT))
    fields = struct.unpack_from(HEADER_FORMAT, header_data, 0)
    header = {
        "magic": fields[0],
        "version": fields[1],
        "rank": fields[2],
        "iteration_count": fields[3],
        "group_count": fields[4],
        "pass_count": fields[5],
        "core_count": fields[6],
        "phase_count": fields[7],
        "cycles_per_us": fields[8],
        "trace_bytes": fields[9],
        "kernel_span_offset": fields[10],
        "task_span_offset": fields[11],
    }
    if header["magic"] != TRACE_MAGIC:
        raise ValueError(f"invalid trace magic in {path}")
    if header["version"] not in (LEGACY_TRACE_VERSION, 2, TRACE_VERSION):
        raise ValueError(f"unsupported trace version {header['version']} in {path}")
    if header["trace_bytes"] != TRACE_BYTES:
        raise ValueError(f"trace byte dimension mismatch in {path}")
    if not 0 < header["iteration_count"] <= MAX_ITERATIONS:
        raise ValueError(f"iteration dimension mismatch in {path}")
    if header["group_count"] <= 0 or header["pass_count"] <= 0:
        raise ValueError(f"group/pass dimension mismatch in {path}")
    expected_phase_count = {
        LEGACY_TRACE_VERSION: PHASE_COUNT,
        2: TRACE_V2_PHASE_COUNT,
        TRACE_VERSION: TRACE_V3_PHASE_COUNT,
    }[header["version"]]
    if (header["core_count"] != MAX_CORES or
            header["phase_count"] != expected_phase_count):
        raise ValueError(f"core/phase dimension mismatch in {path}")
    if (header["kernel_span_offset"] != HEADER_BYTES or
            header["task_span_offset"] != TASK_BASE_OFFSET):
        raise ValueError(f"trace offset mismatch in {path}")
    if header["cycles_per_us"] == 0:
        raise ValueError(f"invalid cycle frequency in {path}")
    required = layout_bytes(
        header["iteration_count"], header["group_count"],
        header["pass_count"], header["phase_count"])
    if required > TRACE_BYTES:
        raise ValueError(f"trace capacity exceeded in {path}: required={required}")
    with path.open("rb") as stream:
        data = stream.read(required)
    if len(data) != required:
        raise ValueError(f"short trace read in {path}: read={len(data)} required={required}")
    return {"path": str(path), "header": header, "data": data}


def _metadata(name, pid, tid, value):
    return {"name": name, "ph": "M", "pid": pid, "tid": tid, "args": {"name": value}}


def _event(name, category, pid, tid, begin, end, base, cycles_per_us, args, offset_us):
    return {
        "name": name,
        "cat": category,
        "ph": "X",
        "pid": pid,
        "tid": tid,
        "ts": offset_us + (begin - base) / cycles_per_us,
        "dur": (end - begin) / cycles_per_us,
        "args": {"beginCycle": begin, "endCycle": end, **args},
    }


def build_chrome_trace(rank_traces):
    events = []
    sources = []
    bases = {}
    iteration_durations = {}
    for rank_trace in rank_traces:
        header = rank_trace["header"]
        rank = header["rank"]
        for iteration in range(header["iteration_count"]):
            spans = [
                _read_span(
                    rank_trace["data"], kernel_span_offset(iteration, core),
                    f"kernel rank={rank} iter={iteration} core={core}")
                for core in range(MAX_CORES)
            ]
            spans = [span for span in spans if span is not None]
            if not spans:
                raise ValueError(f"rank {rank} iteration {iteration} contains no kernel spans")
            base = min(begin for begin, _ in spans)
            bases[(rank, iteration)] = base
            duration = (max(end for _, end in spans) - base) / header["cycles_per_us"]
            iteration_durations[iteration] = max(
                iteration_durations.get(iteration, 0.0), duration)

    iteration_offsets = {}
    cursor = 0.0
    for iteration in sorted(iteration_durations):
        iteration_offsets[iteration] = cursor
        cursor += iteration_durations[iteration] + ITERATION_GAP_US

    for rank_trace in sorted(rank_traces, key=lambda item: item["header"]["rank"]):
        header = rank_trace["header"]
        data = rank_trace["data"]
        rank = header["rank"]
        sources.append(rank_trace["path"])
        events.append(_metadata("process_name", rank, 0, f"rank {rank}"))
        for core in range(MAX_CORES):
            role = "send" if core < SEND_WORKER_COUNT else "receive"
            events.append(_metadata("thread_name", rank, core, f"core{core:02d} {role}"))

        for iteration in range(header["iteration_count"]):
            base = bases[(rank, iteration)]
            offset_us = iteration_offsets[iteration]
            for core in range(MAX_CORES):
                role = "send" if core < SEND_WORKER_COUNT else "receive"
                kernel = _read_span(
                    data, kernel_span_offset(iteration, core),
                    f"kernel rank={rank} iter={iteration} core={core}")
                if kernel is not None:
                    events.append(_event(
                        "kernel", "kernel", rank, core, kernel[0], kernel[1], base,
                        header["cycles_per_us"],
                        {"iteration": iteration, "role": role}, offset_us))
                for group in range(header["group_count"]):
                    for pass_index in range(header["pass_count"]):
                        for phase in range(header["phase_count"]):
                            label = (
                                f"task rank={rank} iter={iteration} core={core} "
                                f"group={group} pass={pass_index} phase={phase}")
                            task = _read_task(
                                data,
                                task_span_offset(
                                    iteration, core, group, pass_index, phase,
                                    header["group_count"], header["pass_count"],
                                    header["phase_count"]),
                                label,
                            )
                            if task is None:
                                continue
                            begin, end, peer, qp = task
                            events.append(_event(
                                PHASE_NAMES[phase], role, rank, core, begin, end, base,
                                header["cycles_per_us"],
                                {
                                    "iteration": iteration,
                                    "group": group,
                                    "pass": pass_index,
                                    "lane": core % LANE_COUNT,
                                    "peer": peer,
                                    "qp": None if qp == NO_QP else qp,
                                    "role": role,
                                },
                                offset_us,
                            ))
    return {
        "traceEvents": events,
        "otherData": {
            "displayTimeUnit": "ns",
            "sources": sources,
            "clock": "GetSystemCycle normalized per rank and iteration",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert TileXR grouped AllToAll GM traces to Chrome trace JSON")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    trace = build_chrome_trace([read_rank_trace(path) for path in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(trace, stream, separators=(",", ":"))
    print(f"wrote {args.output} events={len(trace['traceEvents'])}")


if __name__ == "__main__":
    main()
