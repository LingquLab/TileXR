from __future__ import annotations

import ctypes
import os
import struct
import time
from pathlib import Path

import torch
import torch.distributed as dist

from tools.moonep.test_npu_e2e import Buffer, make_inputs, setup


def mismatch_summary(actual, expected):
    mismatch = actual != expected
    count = int(mismatch.sum().item())
    if count == 0:
        return {"count": 0}
    first = mismatch.nonzero()[0]
    index = tuple(int(value) for value in first.cpu().tolist())
    delta = (actual.float() - expected.float()).abs()
    return {
        "count": count,
        "index": index,
        "actual": float(actual[index].float().item()),
        "expected": float(expected[index].float().item()),
        "max_abs": float(delta.max().item()),
    }


def value_summary(tensor, limit=12):
    values, counts = torch.unique(tensor.float(), return_counts=True)
    pairs = list(zip(values.cpu().tolist(), counts.cpu().tolist()))
    return {
        "unique": len(pairs),
        "values": pairs[:limit],
    }


def hidden_scratch_views(buffer, tokens, hidden_size, topk, nv_s):
    runtime = buffer._context.runtime
    workspace_bytes = ctypes.c_uint64()
    profile_offset = ctypes.c_uint64()
    scratch0_offset = ctypes.c_uint64()
    scratch1_offset = ctypes.c_uint64()
    ret = runtime._combine_v2_lib.TileXRMoonEpCombineGetWorkspaceSizeV2(
        ctypes.c_int64(tokens),
        ctypes.c_int64(hidden_size),
        ctypes.c_int64(topk),
        ctypes.c_int64(nv_s),
        ctypes.c_uint32(11),
        ctypes.byref(workspace_bytes),
        ctypes.byref(profile_offset),
        ctypes.byref(scratch0_offset),
        ctypes.byref(scratch1_offset),
    )
    runtime._check("TileXRMoonEpCombineGetWorkspaceSizeV2", ret)
    context = buffer._context
    raw = context._dispatch_workspace_owner
    aligned_offset = context._dispatch_workspace_ptr - int(raw.data_ptr())
    workspace = raw.narrow(0, aligned_offset, context._dispatch_workspace_bytes)
    scratch_bytes = nv_s * hidden_size * 2
    return tuple(
        workspace.narrow(0, int(offset.value), scratch_bytes)
        .cpu()
        .view(torch.bfloat16)
        .reshape(nv_s, hidden_size)
        for offset in (scratch0_offset, scratch1_offset)
    )


def combine_failure_records(buffer, hidden_size, nv_s):
    profile_bytes = 16 * 320
    expert_bytes = nv_s * hidden_size * 2
    profile_offset = (expert_bytes + 63) // 64 * 64
    scratch0_offset = profile_offset + (profile_bytes + 63) // 64 * 64
    scratch1_offset = scratch0_offset + expert_bytes
    done_offset = (scratch1_offset + expert_bytes + 63) // 64 * 64
    done_bytes = 2 * 128 * 2 * 64
    grant_bytes = 2 * 16 * 2 * 7 * 512
    control_source_bytes = 16 * 2 * 64
    failure_offset = done_offset + done_bytes + grant_bytes + control_source_bytes

    context = buffer._context
    raw = context._dispatch_workspace_owner
    aligned_offset = context._dispatch_workspace_ptr - int(raw.data_ptr())
    workspace = raw.narrow(0, aligned_offset, context._dispatch_workspace_bytes)
    records = workspace.narrow(0, failure_offset, 2 * 16 * 64).cpu().tolist()
    result = []
    for epoch in range(2):
        for core in range(16):
            offset = (epoch * 16 + core) * 64
            fields = struct.unpack_from("<Q8I2Q2I", bytes(records), offset)
            (
                magic, status, rank, record_core, step, peer, lane, qp,
                cq_status, expected, observed, poison, marker,
            ) = fields
            if marker & ~1 == 0x47505632:
                result.append({
                    "epoch": epoch,
                    "magic": magic,
                    "status": status,
                    "rank": rank,
                    "core": record_core,
                    "step": step,
                    "peer": peer,
                    "lane": lane,
                    "qp": qp,
                    "cq_status": cq_status,
                    "expected": expected,
                    "observed": observed,
                    "poison": poison,
                })
    return result


def main():
    rank, rank_size = setup()
    tokens, hidden_size, topk, expert_count = 256, 1024, 4, rank_size * 4
    buffer = Buffer(tokens, hidden_size, topk, expert_count, rank_size,
        B=2, num_sms=32)
    try:
        hidden, weights, topk_experts, tokens_per_expert = make_inputs(
            rank, rank_size, tokens, hidden_size, topk, expert_count)
        constant_hidden = os.environ.get(
            "TILEXR_COMBINE_DIAG_CONSTANT_HIDDEN", "0") == "1"
        if constant_hidden:
            hidden.fill_(rank + 1)
        _, _, _, plan = buffer.dispatch(
            hidden, weights, topk_experts, tokens_per_expert)
        plan = plan.clone()
        exchange_dir = Path(os.environ["TILEXR_COMBINE_DIAG_DIR"])
        exchange_dir.mkdir(parents=True, exist_ok=True)
        rank_path = exchange_dir / f"rank_{rank}.pt"
        temporary_path = exchange_dir / f"rank_{rank}.tmp"
        torch.save(plan.dst.cpu(), temporary_path)
        os.replace(temporary_path, rank_path)
        deadline = time.monotonic() + 30.0
        while len(list(exchange_dir.glob("rank_*.pt"))) != rank_size:
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out exchanging Planner dst snapshots")
            time.sleep(0.01)
        gathered_dst = [
            torch.load(exchange_dir / f"rank_{source_rank}.pt",
                map_location="cpu", weights_only=True)
            for source_rank in range(rank_size)
        ]
        expected_dst_local = torch.full(
            (plan.NvS,), -1, dtype=torch.int32, device="cpu")
        for source_rank, source_dst in enumerate(gathered_dst):
            for route, encoded_tensor in enumerate(source_dst):
                encoded = int(encoded_tensor.item())
                if encoded // plan.NvS == rank:
                    expected_dst_local[encoded % plan.NvS] = (
                        source_rank * plan.NvS + route)
        native_plan = plan._require_native()
        dst_local_bytes = native_plan.workspace[
            native_plan.dst_local_offset:
            native_plan.dst_local_offset + plan.NvS * 4
        ].cpu()
        actual_dst_local = dst_local_bytes.view(torch.int32)

        dispatched_first, weights_first, _, _ = buffer.dispatch(
            hidden, weights, topk_experts, tokens_per_expert)
        output_first, _, _ = buffer.combine(
            plan=plan, hidden_nvsh=dispatched_first)
        torch.npu.synchronize()
        dispatched_first = dispatched_first.clone()
        output_first = output_first.clone()
        scratch_epochs = hidden_scratch_views(
            buffer, tokens, hidden_size, topk, plan.NvS)
        expected_scratch = hidden.repeat_interleave(topk, dim=0).cpu()
        scratch_summaries = [
            mismatch_summary(epoch[:tokens * topk], expected_scratch)
            for epoch in scratch_epochs
        ]

        dispatched_second, weights_second, _, _ = buffer.dispatch(
            hidden, weights, topk_experts, tokens_per_expert)
        output_second, gathered_weights, _ = buffer.combine(
            plan=plan, hidden_nvsh=dispatched_second,
            route_weights_nvs=weights_second)
        torch.npu.synchronize()

        reference = (hidden.float() * float(topk)).to(torch.bfloat16)
        print(f"[combine-v2-diagnostic rank {rank}] "
            f"dst_local={mismatch_summary(actual_dst_local, expected_dst_local)} "
            f"dst_local_routes={int((actual_dst_local >= 0).sum().item())} "
            f"dispatch={mismatch_summary(dispatched_second, dispatched_first)} "
            f"first_vs_reference={mismatch_summary(output_first, reference)} "
            f"second_vs_reference={mismatch_summary(output_second, reference)} "
            f"second_vs_first={mismatch_summary(output_second, output_first)} "
            f"weights={mismatch_summary(gathered_weights, weights)} "
            f"scratch={scratch_summaries} "
            f"first_values={value_summary(output_first)}",
            flush=True)

        for iteration in range(8):
            hidden_loop, weights_loop, _, plan_loop = buffer.dispatch(
                hidden, weights, topk_experts, tokens_per_expert,
                zero_copy=True)
            output_loop, gathered_loop, _ = buffer.combine(
                plan=plan_loop,
                hidden_nvsh=hidden_loop,
                route_weights_nvs=weights_loop,
                zero_copy=True,
            )
            torch.npu.synchronize()
            failures = combine_failure_records(buffer, hidden_size, plan_loop.NvS)
            record_magics = {
                epoch: max((record["magic"] for record in failures
                    if record["epoch"] == epoch), default=0)
                for epoch in range(2)
            }
            nonzero_records = [
                record for record in failures if record["status"] != 0
            ]
            output_summary = mismatch_summary(output_loop, reference)
            weight_summary = mismatch_summary(gathered_loop, weights)
            print(f"[combine-v2-loop rank {rank} iteration {iteration}] "
                f"hidden={output_summary} weights={weight_summary} "
                f"record_magics={record_magics} "
                f"nonzero_records={nonzero_records}", flush=True)
            if output_summary["count"] or weight_summary["count"] or any(
                    record["status"] != 0 for record in failures):
                raise AssertionError("Combine V2 repeated-call diagnostic failed")
    finally:
        buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
