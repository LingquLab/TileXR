#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path


TRACE_MAGIC = 0x464D5452
TRACE_VERSION = 1
TRACE_BYTES = 8 * 1024 * 1024
HEADER_BYTES = 4096
MAX_ITERATIONS = 50
MAX_CORES = 35
KERNEL_REGIONS = 2
PHASE_COUNT = 14
SPAN_BYTES = 16
HEADER_FORMAT = "<8I4Q"
TASK_BASE_OFFSET = HEADER_BYTES + MAX_ITERATIONS * MAX_CORES * KERNEL_REGIONS * SPAN_BYTES

PHASE_NAMES = (
    "pass",
    "self-copy",
    "peer-copy",
    "publish-copy-ready",
    "wait-copy-ready",
    "data-put",
    "quiet",
    "segment-done",
    "publish-ready",
    "wait-ready",
    "output-copy",
    "publish-recv-done",
    "wait-recv-done",
    "ACK",
)
ITERATION_GAP_US = 1.0


def kernel_span_offset(iteration, core, region):
    index = (iteration * MAX_CORES + core) * KERNEL_REGIONS + region
    return HEADER_BYTES + index * SPAN_BYTES


def task_span_offset(iteration, core, pass_index, peer, phase, pass_count, rank_size):
    index = (((((iteration * MAX_CORES + core) * pass_count + pass_index) *
               rank_size + peer) * PHASE_COUNT) + phase)
    return TASK_BASE_OFFSET + index * SPAN_BYTES


def layout_bytes(iteration_count, pass_count, rank_size):
    return TASK_BASE_OFFSET + (
        iteration_count * MAX_CORES * pass_count * rank_size * PHASE_COUNT * SPAN_BYTES)


def _core_role(core):
    if core < 16:
        return "copy"
    if core == 16:
        return "remote-primary"
    if core == 17:
        return "remote-secondary"
    if core == 18:
        return "local-send"
    return "recv-copy"


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


def read_rank_trace(path):
    path = Path(path)
    data = path.read_bytes()
    if len(data) != TRACE_BYTES:
        raise ValueError(f"invalid trace size {len(data)} in {path}, expected {TRACE_BYTES}")
    fields = struct.unpack_from(HEADER_FORMAT, data, 0)
    header = {
        "magic": fields[0],
        "version": fields[1],
        "rank": fields[2],
        "iteration_count": fields[3],
        "pass_count": fields[4],
        "core_count": fields[5],
        "rank_size": fields[6],
        "phase_count": fields[7],
        "cycles_per_us": fields[8],
        "trace_bytes": fields[9],
        "kernel_span_offset": fields[10],
        "task_span_offset": fields[11],
    }
    if header["magic"] != TRACE_MAGIC:
        raise ValueError(f"invalid trace magic in {path}")
    if header["version"] != TRACE_VERSION:
        raise ValueError(f"unsupported trace version {header['version']} in {path}")
    if header["trace_bytes"] != TRACE_BYTES:
        raise ValueError(f"trace byte dimension mismatch in {path}")
    if not 0 < header["iteration_count"] <= MAX_ITERATIONS:
        raise ValueError(f"iteration dimension mismatch in {path}")
    if header["pass_count"] <= 0 or header["rank_size"] <= 0:
        raise ValueError(f"pass/rank dimension mismatch in {path}")
    if header["core_count"] != MAX_CORES or header["phase_count"] != PHASE_COUNT:
        raise ValueError(f"core/phase dimension mismatch in {path}")
    if header["kernel_span_offset"] != HEADER_BYTES or header["task_span_offset"] != TASK_BASE_OFFSET:
        raise ValueError(f"trace offset mismatch in {path}")
    if header["cycles_per_us"] == 0:
        raise ValueError(f"invalid cycle frequency in {path}")
    required = layout_bytes(
        header["iteration_count"], header["pass_count"], header["rank_size"])
    if required > TRACE_BYTES:
        raise ValueError(f"trace capacity exceeded in {path}: required={required} capacity={TRACE_BYTES}")
    return {"path": str(path), "header": header, "data": data}


def _metadata(name, pid, tid, value):
    return {"name": name, "ph": "M", "pid": pid, "tid": tid, "args": {"name": value}}


def _span_event(name, category, pid, tid, begin, end, base, cycles_per_us, args, offset_us=0.0):
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
        data = rank_trace["data"]
        rank = header["rank"]
        for iteration in range(header["iteration_count"]):
            spans = []
            for core in range(header["core_count"]):
                span = _read_span(
                    data, kernel_span_offset(iteration, core, 0),
                    f"kernel rank={rank} iter={iteration} core={core}")
                if span is not None:
                    spans.append(span)
            if not spans:
                raise ValueError(f"rank {rank} iteration {iteration} contains no kernel spans")
            base = min(begin for begin, _ in spans)
            bases[(rank, iteration)] = base
            duration = (max(end for _, end in spans) - base) / header["cycles_per_us"]
            iteration_durations[iteration] = max(iteration_durations.get(iteration, 0.0), duration)

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
        for core in range(header["core_count"]):
            events.append(_metadata("thread_name", rank, core, f"core{core} {_core_role(core)}"))

        for iteration in range(header["iteration_count"]):
            base = bases[(rank, iteration)]
            offset_us = iteration_offsets[iteration]
            for core in range(header["core_count"]):
                role = _core_role(core)
                for region, name in ((0, "kernel"), (1, "work")):
                    span = _read_span(
                        data, kernel_span_offset(iteration, core, region),
                        f"kernel rank={rank} iter={iteration} core={core} region={region}")
                    if span is not None:
                        events.append(_span_event(
                            name, "kernel", rank, core, span[0], span[1], base,
                            header["cycles_per_us"],
                            {"iteration": iteration, "role": role}, offset_us))
                for pass_index in range(header["pass_count"]):
                    for peer in range(header["rank_size"]):
                        for phase in range(header["phase_count"]):
                            span = _read_span(
                                data,
                                task_span_offset(
                                    iteration, core, pass_index, peer, phase,
                                    header["pass_count"], header["rank_size"]),
                                f"task rank={rank} iter={iteration} core={core} "
                                f"pass={pass_index} peer={peer} phase={phase}")
                            if span is None:
                                continue
                            events.append(_span_event(
                                PHASE_NAMES[phase], role, rank, core,
                                span[0], span[1], base, header["cycles_per_us"],
                                {"iteration": iteration, "pass": pass_index,
                                 "peer": peer, "phase": phase, "role": role}, offset_us))
    return {
        "traceEvents": events,
        "otherData": {
            "sources": sources,
            "clock": "GetSystemCycle normalized per rank and iteration, iterations laid out sequentially",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Convert TileXR full-mesh GM traces to Chrome trace JSON")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    trace = build_chrome_trace([read_rank_trace(path) for path in args.inputs])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.output} events={len(trace['traceEvents'])}")


if __name__ == "__main__":
    main()
