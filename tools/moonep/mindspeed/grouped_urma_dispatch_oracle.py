"""Exact 8-rank oracle for repeated grouped URMA MoonEP Dispatch."""

from __future__ import annotations

import ctypes
import os
import struct
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(os.environ.get("TILEXR_ORACLE_SOURCE_ROOT", Path(__file__).resolve().parents[3]))
INTEGRATION = ROOT / "integrations" / "moonep_torch"
for path in (ROOT, INTEGRATION):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from tilexr_moonep import Buffer, ProjectionBuffers


RANK_SIZE = 8
TOKENS_PER_RANK = 4096
HIDDEN_SIZE = int(os.environ.get("TILEXR_ORACLE_HIDDEN_SIZE", "32"))
TOPK = 8
EXPERT_COUNT = 32
ITERATIONS = int(os.environ.get("TILEXR_ORACLE_ITERATIONS", "16"))
ROUTE_MODE = os.environ.get("TILEXR_ORACLE_ROUTE_MODE", "balanced")
WITH_ROUTE_WEIGHTS = os.environ.get("TILEXR_ORACLE_WITH_ROUTE_WEIGHTS", "1") == "1"
WITH_COMBINE = os.environ.get("TILEXR_ORACLE_WITH_COMBINE", "1") == "1"
MAGIC_ADVANCE_AFTER_FIRST = int(
    os.environ.get("TILEXR_ORACLE_MAGIC_ADVANCE_AFTER_FIRST", "0")
)
SWITCH_REGISTRATION = (
    os.environ.get("TILEXR_ORACLE_SWITCH_REGISTRATION", "0") == "1"
)
EXTRA_PLAN_COUNT = int(os.environ.get("TILEXR_ORACLE_EXTRA_PLAN_COUNT", "0"))
WITH_PREFETCH = os.environ.get("TILEXR_ORACLE_WITH_PREFETCH", "0") == "1"
PROJECTION_SIZE = int(os.environ.get("TILEXR_ORACLE_PROJECTION_SIZE", "256"))
WAIT_SECONDS = int(os.environ.get("TILEXR_ORACLE_WAIT_SECONDS", "120"))


def make_inputs(rank: int, device: str):
    token = torch.arange(TOKENS_PER_RANK, dtype=torch.int64).view(-1, 1)
    column = torch.arange(HIDDEN_SIZE, dtype=torch.int64).view(1, -1)
    route = torch.arange(TOPK, dtype=torch.int64).view(1, -1)
    hidden = ((token * 17 + column * 3 + rank * 29) % 251).to(torch.bfloat16)
    weights = (
        ((token * TOPK + route * 11 + rank * 37) % 1024).to(torch.float32)
        / 1024.0
    )
    if ROUTE_MODE == "balanced":
        local_expert = (token + rank) % (EXPERT_COUNT // RANK_SIZE)
        topk = (route * (EXPERT_COUNT // RANK_SIZE) + local_expert).to(torch.int32)
    elif ROUTE_MODE == "model_skew":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(20260811 + rank)
        logits = torch.randn(
            (TOKENS_PER_RANK, EXPERT_COUNT), generator=generator
        )
        logits += torch.linspace(0.6, -0.6, EXPERT_COUNT).reshape(1, -1)
        topk = torch.topk(logits, TOPK, dim=1).indices.to(torch.int32)
    else:
        raise ValueError(f"unknown TILEXR_ORACLE_ROUTE_MODE={ROUTE_MODE}")
    tpe = torch.bincount(topk.flatten().to(torch.int64), minlength=EXPERT_COUNT)
    return (
        hidden.contiguous().to(device),
        weights.contiguous().to(device),
        topk.contiguous().to(device),
        tpe.to(torch.int32).contiguous().to(device),
    )


def wait_for(paths: list[Path], label: str) -> None:
    deadline = time.monotonic() + WAIT_SECONDS
    while not all(path.exists() for path in paths):
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"timed out waiting for {label}: {missing}")
        time.sleep(0.05)


def publish_tensor(path: Path, value: torch.Tensor, rank: int) -> None:
    temporary = path.with_name(f".{path.name}.rank{rank}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def file_barrier(run_dir: Path, label: str, rank: int) -> None:
    marker = run_dir / f"{label}.rank{rank}.ready"
    marker.touch()
    wait_for(
        [run_dir / f"{label}.rank{peer}.ready" for peer in range(RANK_SIZE)],
        label,
    )


def build_expected(run_dir: Path, target_rank: int, nv_s: int):
    expected_hidden = torch.empty(
        (nv_s, HIDDEN_SIZE), dtype=torch.bfloat16, device="cpu"
    )
    expected_weights = torch.empty((nv_s,), dtype=torch.float32, device="cpu")
    filled = torch.zeros((nv_s,), dtype=torch.bool, device="cpu")
    route_ids = torch.arange(TOKENS_PER_RANK * TOPK, dtype=torch.int64)

    for source_rank in range(RANK_SIZE):
        dst = torch.load(run_dir / f"dst.rank{source_rank}.pt", map_location="cpu")
        dst = dst.to(torch.int64)
        if tuple(dst.shape) != (TOKENS_PER_RANK * TOPK,):
            raise AssertionError(f"rank {source_rank} dst shape is {tuple(dst.shape)}")
        if bool(((dst < 0) | (dst >= RANK_SIZE * nv_s)).any().item()):
            bad = int(torch.nonzero((dst < 0) | (dst >= RANK_SIZE * nv_s))[0].item())
            raise AssertionError(
                f"rank {source_rank} dst[{bad}]={int(dst[bad])} is out of range"
            )

        selected = torch.div(dst, nv_s, rounding_mode="floor") == target_rank
        slots = torch.remainder(dst[selected], nv_s)
        selected_routes = route_ids[selected]
        if bool(filled[slots].any().item()):
            slot = int(slots[torch.nonzero(filled[slots])[0]].item())
            raise AssertionError(f"destination slot {slot} has multiple writers")

        source_hidden, source_weights, _, _ = make_inputs(source_rank, "cpu")
        expected_hidden[slots] = source_hidden[selected_routes // TOPK]
        expected_weights[slots] = source_weights.flatten()[selected_routes]
        filled[slots] = True

    if not bool(filled.all().item()):
        slot = int(torch.nonzero(~filled)[0].item())
        raise AssertionError(f"destination slot {slot} has no writer")
    return expected_hidden, expected_weights


def compare_exact(
    rank: int,
    iteration: int,
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    actual_cpu = actual.detach().cpu()
    if torch.equal(actual_cpu, expected):
        print(
            f"[grouped-urma-oracle rank={rank}] iteration={iteration} "
            f"{name}=exact",
            flush=True,
        )
        return
    mismatch = actual_cpu != expected
    first = tuple(int(value) for value in torch.nonzero(mismatch, as_tuple=False)[0])
    raise AssertionError(
        f"rank {rank} iteration {iteration} {name} mismatch at {first}: "
        f"actual={actual_cpu[first].item()} expected={expected[first].item()}"
    )


def advance_magic(buffer: Buffer, rank: int) -> None:
    runtime = buffer._native_buffer.runtime
    function = runtime._comm_lib.TileXRCommNextMagic
    function.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64)]
    function.restype = ctypes.c_int
    values = []
    for _ in range(MAGIC_ADVANCE_AFTER_FIRST):
        value = ctypes.c_int64()
        ret = int(function(ctypes.c_void_p(runtime.comm_ptr), ctypes.byref(value)))
        if ret != 0:
            raise RuntimeError(f"TileXRCommNextMagic failed with ret={ret}")
        values.append(int(value.value))
    if values:
        print(
            f"[grouped-urma-oracle rank={rank}] advanced_magic={values}",
            flush=True,
        )


def activate_dummy_registration(buffer: Buffer, workspace: torch.Tensor) -> None:
    native = buffer._native_buffer
    native._synchronize_device()
    native.runtime._activate_udma_region(
        int(workspace.data_ptr()),
        int(workspace.numel()),
        "oracle_dummy",
        f"workspace_bytes={int(workspace.numel())}",
    )


def dump_hidden_scratch(
    buffer: Buffer,
    plan,
    rank: int,
    iteration: int,
    expected_hidden: torch.Tensor,
) -> None:
    context = buffer._native_buffer.context
    raw = context._dispatch_workspace_owner
    aligned_offset = int(context._dispatch_workspace_ptr) - int(raw.data_ptr())
    source_bytes = TOKENS_PER_RANK * HIDDEN_SIZE * 2
    scratch_offset = (source_bytes + 63) // 64 * 64
    scratch_slot_bytes = TOKENS_PER_RANK * TOPK * HIDDEN_SIZE * 2
    raw_cpu = raw.detach().cpu()
    print(
        f"[grouped-urma-oracle rank={rank}] iteration={iteration} "
        f"workspace_bytes={context._dispatch_workspace_bytes} "
        f"aligned_offset={aligned_offset} hidden_scratch_offset={scratch_offset} "
        f"hidden_scratch_slot_bytes={scratch_slot_bytes}",
        flush=True,
    )
    for scratch_index in range(2):
        byte_offset = aligned_offset + scratch_offset + scratch_index * scratch_slot_bytes
        scratch = (
            raw_cpu.narrow(0, byte_offset, scratch_slot_bytes)
            .view(torch.bfloat16)
            .reshape(TOKENS_PER_RANK * TOPK, HIDDEN_SIZE)
        )
        mismatch_count = int((scratch != expected_hidden).sum().item())
        print(
            f"[grouped-urma-oracle rank={rank}] iteration={iteration} "
            f"scratch={scratch_index} mismatch_elements={mismatch_count} "
            f"row0={scratch[0, :8].tolist()}",
            flush=True,
        )
    raw_bytes = raw_cpu.numpy().tobytes()
    status_marker = struct.pack("<I", 0x54584453)
    status_offset = raw_bytes.find(status_marker)
    while status_offset >= 0 and status_offset + 64 <= len(raw_bytes):
        marker, version, record_bytes, status, payload_mode, magic = struct.unpack_from(
            "<IHHiiQ", raw_bytes, status_offset
        )
        print(
            f"[grouped-urma-oracle rank={rank}] iteration={iteration} "
            f"kernel_status_offset={status_offset - aligned_offset} "
            f"marker=0x{marker:08x} version={version} record_bytes={record_bytes} "
            f"status={status} payload_mode={payload_mode} magic={magic}",
            flush=True,
        )
        status_offset = raw_bytes.find(status_marker, status_offset + 1)
    zero_fill = plan.zero_fill_ranges.detach().cpu()
    active = torch.nonzero(zero_fill[:, 1] != 0, as_tuple=False).flatten()
    print(
        f"[grouped-urma-oracle rank={rank}] iteration={iteration} "
        f"zero_fill_active={active.tolist()} "
        f"zero_fill_values={zero_fill[active].tolist()}",
        flush=True,
    )


def main() -> int:
    if os.environ.get("TILEXR_MOONEP_DISPATCH_TRANSPORT"):
        raise RuntimeError("Dispatch transport override must be unset for this URMA oracle")
    if os.environ.get("TILEXR_MOONEP_DISPATCH_PEER_MODE") != "group":
        raise RuntimeError("TILEXR_MOONEP_DISPATCH_PEER_MODE must be group")
    if os.environ.get("TILEXR_MOONEP_DISPATCH_GROUP_WIDTH") != "16":
        raise RuntimeError("TILEXR_MOONEP_DISPATCH_GROUP_WIDTH must be 16")

    import torch_npu  # noqa: F401

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl")
    if dist.get_world_size() != RANK_SIZE:
        raise RuntimeError(f"expected {RANK_SIZE} ranks, got {dist.get_world_size()}")

    run_dir = Path(os.environ["TILEXR_ORACLE_RUN_DIR"])
    buffer = None
    dummy_workspace = None
    dummy_allocation = None
    projections = None
    try:
        hidden, weights, topk, tpe = make_inputs(rank, "npu")
        buffer = Buffer(
            TOKENS_PER_RANK,
            HIDDEN_SIZE,
            TOPK,
            EXPERT_COUNT,
            RANK_SIZE,
            B=EXPERT_COUNT // RANK_SIZE,
            num_sms=32,
            token_padding=1,
        )
        if SWITCH_REGISTRATION:
            dummy_workspace, dummy_allocation = (
                buffer._native_buffer._aligned_workspace(2 * 1024 * 1024, 2 * 1024 * 1024)
            )
        dispatch_weights = weights if WITH_ROUTE_WEIGHTS else None
        hidden_out, weight_out, _, plan = buffer.dispatch(
            hidden, dispatch_weights, topk, tpe
        )
        torch.npu.synchronize()
        nv_s = int(hidden_out.shape[0])
        if nv_s != TOKENS_PER_RANK * TOPK:
            raise AssertionError(f"expected NvS=32768, got {nv_s}")

        publish_tensor(run_dir / f"dst.rank{rank}.pt", plan.dst.detach().cpu(), rank)
        file_barrier(run_dir, "plans", rank)
        expected_hidden, expected_weights = build_expected(run_dir, rank, nv_s)

        expected_combine_hidden = (hidden.detach().cpu().to(torch.float32) * TOPK).to(
            torch.bfloat16
        )
        expected_combine_weights = weights.detach().cpu()

        if WITH_PREFETCH:
            local_experts = EXPERT_COUNT // RANK_SIZE
            local_weights = tuple(
                torch.zeros(
                    (local_experts, HIDDEN_SIZE, PROJECTION_SIZE),
                    dtype=torch.bfloat16,
                    device="npu",
                )
                for _ in range(3)
            )
            projections = ProjectionBuffers.from_local_weights(
                buffer._context, *local_weights, torch_module=torch
            )
            buffer._native_buffer.register_projection_buffers(projections)
            buffer._native_buffer.prefetch_weight(
                plan._require_native(), projections, async_finish=False
            )

        compare_exact(rank, 0, "hidden", hidden_out, expected_hidden)
        if WITH_ROUTE_WEIGHTS:
            compare_exact(rank, 0, "route_weights", weight_out, expected_weights)
        if WITH_COMBINE:
            combined_hidden, combined_weights, _ = buffer.combine(
                plan=plan,
                hidden_nvsh=hidden_out,
                route_weights_nvs=weight_out,
            )
            torch.npu.synchronize()
            compare_exact(
                rank, 0, "combined_hidden", combined_hidden, expected_combine_hidden
            )
            if WITH_ROUTE_WEIGHTS:
                compare_exact(
                    rank,
                    0,
                    "combined_route_weights",
                    combined_weights,
                    expected_combine_weights,
                )
        for extra_plan in range(EXTRA_PLAN_COUNT):
            extra_topk = torch.remainder(
                topk + (extra_plan + 1) * 3, EXPERT_COUNT
            ).to(torch.int32)
            extra_tpe = torch.bincount(
                extra_topk.flatten().to(torch.int64), minlength=EXPERT_COUNT
            ).to(torch.int32)
            hidden_out, weight_out, _, plan = buffer.dispatch(
                hidden,
                dispatch_weights,
                extra_topk.contiguous(),
                extra_tpe.contiguous(),
            )
            torch.npu.synchronize()
            if WITH_PREFETCH:
                buffer._native_buffer.prefetch_weight(
                    plan._require_native(), projections, async_finish=False
                )
            if WITH_COMBINE:
                combined_hidden, _, _ = buffer.combine(
                    plan=plan,
                    hidden_nvsh=hidden_out,
                    route_weights_nvs=weight_out,
                )
                torch.npu.synchronize()
                compare_exact(
                    rank,
                    -(extra_plan + 1),
                    "extra_plan_combined_hidden",
                    combined_hidden,
                    expected_combine_hidden,
                )
            if SWITCH_REGISTRATION:
                activate_dummy_registration(buffer, dummy_workspace)

        if EXTRA_PLAN_COUNT:
            publish_tensor(run_dir / f"dst.rank{rank}.pt", plan.dst.detach().cpu(), rank)
            file_barrier(run_dir, "extra_plans", rank)
            expected_hidden, expected_weights = build_expected(run_dir, rank, nv_s)

        advance_magic(buffer, rank)
        if SWITCH_REGISTRATION:
            activate_dummy_registration(buffer, dummy_workspace)
        file_barrier(run_dir, "iteration0", rank)

        reuse_status_mode = os.environ.get(
            "TILEXR_ORACLE_SYNC_REUSE_STATUS", "0"
        )
        if reuse_status_mode != "0":
            native_plan = plan._require_native()
            native_plan.status.zero_()
            if reuse_status_mode == "sync":
                torch.npu.synchronize()
            buffer._native_buffer._pending_statuses.pop(id(native_plan), None)

        for iteration in range(1, ITERATIONS):
            hidden_out, weight_out, _, returned_plan = buffer.dispatch(
                hidden, dispatch_weights, plan=plan
            )
            if returned_plan is not plan:
                raise AssertionError("reused Dispatch did not return the same plan")
            torch.npu.synchronize()
            buffer._native_buffer._check_plan_status(
                returned_plan._require_native()
            )
            if WITH_PREFETCH:
                buffer._native_buffer.prefetch_weight(
                    plan._require_native(), projections, async_finish=False
                )
            try:
                compare_exact(rank, iteration, "hidden", hidden_out, expected_hidden)
            except AssertionError:
                dump_hidden_scratch(buffer, plan, rank, iteration, expected_hidden)
                raise
            if WITH_ROUTE_WEIGHTS:
                compare_exact(
                    rank, iteration, "route_weights", weight_out, expected_weights
                )
            if WITH_COMBINE:
                combined_hidden, combined_weights, _ = buffer.combine(
                    plan=plan,
                    hidden_nvsh=hidden_out,
                    route_weights_nvs=weight_out,
                )
                torch.npu.synchronize()
                compare_exact(
                    rank,
                    iteration,
                    "combined_hidden",
                    combined_hidden,
                    expected_combine_hidden,
                )
                if WITH_ROUTE_WEIGHTS:
                    compare_exact(
                        rank,
                        iteration,
                        "combined_route_weights",
                        combined_weights,
                        expected_combine_weights,
                    )
            if SWITCH_REGISTRATION:
                activate_dummy_registration(buffer, dummy_workspace)
            file_barrier(run_dir, f"iteration{iteration}", rank)

        if rank == 0:
            print(
                f"[grouped-urma-oracle] PASS iterations={ITERATIONS} "
                f"S=4096 K=8 H={HIDDEN_SIZE} rank_size=8 "
                f"peer_mode=group group_width=16 weights={int(WITH_ROUTE_WEIGHTS)} "
                f"combine={int(WITH_COMBINE)} "
                f"route_mode={ROUTE_MODE} "
                f"switch_registration={int(SWITCH_REGISTRATION)} "
                f"extra_plans={EXTRA_PLAN_COUNT} "
                f"prefetch={int(WITH_PREFETCH)} "
                f"magic_advance={MAGIC_ADVANCE_AFTER_FIRST}",
                flush=True,
            )
        return 0
    finally:
        if buffer is not None and not buffer.destroyed:
            buffer.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
