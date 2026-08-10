"""Ascend NPU end-to-end tool adapted from MoonEP's public API E2E.

Run directly from the repository root. The parent process launches eight local
workers by default:

    python tools/moonep/test_npu_e2e.py

The same file can also run under an existing torchrun environment:

    torchrun --nproc-per-node=8 tools/moonep/test_npu_e2e.py
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from pathlib import Path

import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[2]
INTEGRATION = ROOT / "integrations" / "moonep_torch"
for path in (ROOT, INTEGRATION):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

from tilexr_moonep import Buffer, MoonEPCommPlan


DEDUP_PLAN_FIELDS = ("dup_groups", "dup_loffs", "dup_counts")
REQUIRED_UDMA_QP_ROUTE_SPEC = "port_count:6,port_count:2"
DEFAULT_NPROC_PER_NODE = 8


def local_device_index() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))


def setup():
    import torch_npu  # noqa: F401

    route_spec = os.environ.setdefault(
        "TILEXR_UDMA_QP_ROUTE_SPEC", REQUIRED_UDMA_QP_ROUTE_SPEC
    )
    if route_spec != REQUIRED_UDMA_QP_ROUTE_SPEC:
        raise RuntimeError(
            "test_npu_e2e requires TILEXR_UDMA_QP_ROUTE_SPEC="
            f"{REQUIRED_UDMA_QP_ROUTE_SPEC}, got {route_spec}"
        )
    torch.npu.set_device(local_device_index())
    dist.init_process_group(backend="hccl")
    return dist.get_rank(), dist.get_world_size()


def make_topk(rank, R, S, K, E, *, device):
    token = torch.arange(S, dtype=torch.int64, device=device).view(S, 1)
    route = torch.arange(K, dtype=torch.int64, device=device).view(1, K)
    experts_per_rank = E // R
    owner = route.expand(S, K)
    local_expert = (rank + token) % experts_per_rank
    return (owner * experts_per_rank + local_expert).to(torch.int32).contiguous()


def make_inputs(rank, R, S, H, K, E, seed=0):
    if R <= K:
        raise ValueError(
            f"active duplicate-free prefetch requires R > K, got R={R}, K={K}"
        )
    dev = "npu"
    generator = torch.Generator(device=dev).manual_seed(seed + rank)
    hidden = torch.randn(S, H, dtype=torch.bfloat16, device=dev, generator=generator)
    weights = torch.rand(S, K, dtype=torch.float32, device=dev, generator=generator)
    topk = make_topk(rank, R, S, K, E, device=dev)
    tpe = torch.bincount(topk.flatten().to(torch.int64), minlength=E).to(torch.int32)
    return hidden, weights, topk, tpe


def validate_topk_routing():
    R, S, K, E = 8, 256, 4, 32
    experts_per_rank = E // R
    owner_rows = []
    for rank in range(R):
        topk = make_topk(rank, R, S, K, E, device="cpu")
        owners = torch.div(topk, experts_per_rank, rounding_mode="floor")
        assert torch.equal(torch.sort(owners, dim=1).values, torch.arange(K).expand(S, K))
        owner_rows.append(owners)
    owner_counts = torch.bincount(torch.stack(owner_rows).flatten(), minlength=R)
    assert torch.equal(
        owner_counts,
        torch.tensor([R * S] * K + [0] * (R - K), dtype=torch.int64),
    )


def make_global_experts(E, H, Hp):
    generator = torch.Generator(device="npu").manual_seed(5000)
    return torch.randn(
        E, H, Hp, dtype=torch.bfloat16, device="npu", generator=generator
    )


def make_full_weight(global_experts, B, source_offset):
    E, H, Hp = global_experts.shape
    full_weight = torch.empty(
        E + B, H, Hp, dtype=torch.bfloat16, device=global_experts.device
    )
    full_weight[:E].copy_(global_experts)
    if source_offset:
        full_weight[:E].add_(source_offset)
    full_weight[E:].zero_()
    return full_weight


def make_prefetch_args(global_experts, B):
    return {
        "full_gate_weight": make_full_weight(global_experts, B, 0.0),
        "full_up_weight": make_full_weight(global_experts, B, 1.0),
        "full_down_weight": make_full_weight(global_experts, B, 2.0),
    }


def assert_prefetched(prefetch_args, experts_to_copy):
    assert experts_to_copy.dim() == 1
    B = experts_to_copy.numel()
    for slot in range(B):
        expert = int(experts_to_copy[slot].item())
        if expert < 0:
            continue
        for name, full_weight in prefetch_args.items():
            E = full_weight.shape[0] - B
            assert torch.equal(full_weight[E + slot], full_weight[expert]), (
                f"{name} prefetch buffer {slot} does not match expert {expert}"
            )


def assert_remote_prefetch_planned(plan, rank, R, K):
    active = plan.experts_to_copy >= 0
    assert bool(active.any().item()), "Planner produced no remote PrefetchWeight slots"
    local_count = int(active[rank].sum().item())
    if rank >= K:
        assert local_count > 0, f"rank {rank} should receive a migrated remote expert"
    print(f"[test_npu_e2e rank {rank}] live_prefetch_slots={local_count}", flush=True)


def fill_prefetch_slots(prefetch_args, B, value):
    for full_weight in prefetch_args.values():
        full_weight[-B:].fill_(value)


def assert_prefetch_slots_equal(prefetch_args, B, value):
    for name, full_weight in prefetch_args.items():
        expected = torch.full_like(full_weight[-B:], value)
        assert torch.equal(full_weight[-B:], expected), (
            f"{name} prefetch slots changed without prefetch_weight"
        )


def clone_dedup_plan_fields(plan):
    return {name: getattr(plan, name).clone() for name in DEDUP_PLAN_FIELDS}


def dedup_plan_fields_equal(plan, snapshot):
    return all(
        torch.equal(getattr(plan, name), snapshot[name])
        for name in DEDUP_PLAN_FIELDS
    )


def _dedup_group_map(plan):
    groups = plan.dup_groups.cpu()
    dup_loffs = plan.dup_loffs.cpu()
    counts = plan.dup_counts.cpu()
    NvS = dup_loffs.numel()
    errors = []
    mapping = {}
    group_count = int(counts[0].item())
    dup_count_total = int(counts[1].item())
    if not 0 <= group_count <= NvS:
        errors.append(f"dup group count {group_count} is out of range")
        group_count = max(0, min(group_count, NvS))
    if not 0 <= dup_count_total <= NvS:
        errors.append(f"dup loff count {dup_count_total} is out of range")
        dup_count_total = max(0, min(dup_count_total, NvS))

    seen = []
    for group_index in range(group_count):
        primary, begin, count = (int(value) for value in groups[group_index].tolist())
        if not 0 <= primary < NvS or begin < 0 or count <= 0:
            errors.append(f"invalid duplicate group {group_index}")
            continue
        if begin + count > dup_count_total:
            errors.append(f"duplicate group {group_index} exceeds compact prefix")
            continue
        duplicates = tuple(
            sorted(int(value) for value in dup_loffs[begin:begin + count].tolist())
        )
        if any(value < 0 or value >= NvS for value in duplicates):
            errors.append(f"duplicate group {group_index} has out-of-range rows")
            continue
        mapping[primary] = duplicates
        seen.extend(duplicates)
    if len(seen) != len(set(seen)) or len(seen) != dup_count_total:
        errors.append("duplicate compact prefix is inconsistent")
    return mapping, errors


def dedup_plan_semantic_errors(name, actual, expected):
    errors = []
    for field in DEDUP_PLAN_FIELDS:
        actual_tensor = getattr(actual, field)
        expected_tensor = getattr(expected, field)
        if actual_tensor.dtype != expected_tensor.dtype:
            errors.append(f"{name}.{field} dtype mismatch")
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            errors.append(f"{name}.{field} shape mismatch")
    if errors:
        return errors
    if not torch.equal(actual.dup_counts.cpu(), expected.dup_counts.cpu()):
        errors.append(f"{name}.dup_counts mismatch")
    actual_map, actual_errors = _dedup_group_map(actual)
    expected_map, expected_errors = _dedup_group_map(expected)
    errors.extend(f"actual {value}" for value in actual_errors)
    errors.extend(f"expected {value}" for value in expected_errors)
    if actual_map != expected_map:
        errors.append(f"{name} duplicate group map differs")
    return errors


def assert_dedup_plan_semantic_equal(actual, expected):
    errors = dedup_plan_semantic_errors("dedup plan", actual, expected)
    assert not errors, "; ".join(errors[:5])


def grad_base(E, H, Hp, offset, device):
    expert = torch.arange(E, dtype=torch.float32, device=device).view(E, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=device).view(1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=device).view(1, 1, Hp)
    return offset + expert * 17.0 + row * 0.125 + col * 0.0078125


def reduce_base(R, B, H, Hp, offset, device):
    source = torch.arange(R, dtype=torch.float32, device=device).view(R, 1, 1, 1)
    slot = torch.arange(B, dtype=torch.float32, device=device).view(1, B, 1, 1)
    row = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
    col = torch.arange(Hp, dtype=torch.float32, device=device).view(1, 1, 1, Hp)
    return (
        offset
        + (source + 1.0) * 101.0
        + slot * 11.0
        + row * 0.03125
        + col * 0.00390625
    )


def make_grad_reduce_args(R, E, B, H, Hp, offsets):
    device = "npu"
    full_E = E + B
    return {
        "full_gate_grad": grad_base(full_E, H, Hp, offsets[0], device).contiguous(),
        "full_up_grad": grad_base(full_E, H, Hp, offsets[1], device).contiguous(),
        "full_down_grad": grad_base(full_E, H, Hp, offsets[2], device).contiguous(),
        "gate_reduce_buffer": reduce_base(R, B, H, Hp, offsets[3], device).contiguous(),
        "up_reduce_buffer": reduce_base(R, B, H, Hp, offsets[4], device).contiguous(),
        "down_reduce_buffer": reduce_base(R, B, H, Hp, offsets[5], device).contiguous(),
    }


def expected_local_grad(rank, R, E, H, Hp, full_offset, reduce_offset, experts_to_copy):
    device = "npu"
    expected = grad_base(E, H, Hp, full_offset, device)
    values = reduce_base(R, experts_to_copy.shape[1], H, Hp, reduce_offset, device)
    for source_rank in range(R):
        for slot in range(experts_to_copy.shape[1]):
            expert = int(experts_to_copy[source_rank, slot].item())
            if expert >= 0:
                expected[expert].add_(values[source_rank, slot])
    experts_per_rank = E // R
    begin = rank * experts_per_rank
    return expected[begin:begin + experts_per_rank].contiguous()


def assert_grad_reduced(rank, R, E, H, Hp, experts_to_copy, args, offsets):
    experts_per_rank = E // R
    local_begin = rank * experts_per_rank
    local_end = local_begin + experts_per_rank
    pairs = (
        ("gate", args["full_gate_grad"], args["gate_reduce_buffer"], offsets[0], offsets[3]),
        ("up", args["full_up_grad"], args["up_reduce_buffer"], offsets[1], offsets[4]),
        ("down", args["full_down_grad"], args["down_reduce_buffer"], offsets[2], offsets[5]),
    )
    for name, full_grad, reduce_buffer, full_offset, reduce_offset in pairs:
        expected = expected_local_grad(
            rank, R, E, H, Hp, full_offset, reduce_offset, experts_to_copy
        )
        assert torch.allclose(
            full_grad[local_begin:local_end], expected, rtol=0.0, atol=1e-5
        ), f"{name} local grad reduce mismatch"
        initial = grad_base(E + experts_to_copy.shape[1], H, Hp, full_offset, full_grad.device)
        assert torch.equal(full_grad[:local_begin], initial[:local_begin])
        assert torch.equal(full_grad[local_end:], initial[local_end:])

        original = reduce_base(
            R, experts_to_copy.shape[1], H, Hp, reduce_offset, reduce_buffer.device
        )
        for source_rank in range(R):
            for slot in range(experts_to_copy.shape[1]):
                expert = int(experts_to_copy[source_rank, slot].item())
                if expert >= 0 and source_rank == rank:
                    assert torch.equal(
                        reduce_buffer[source_rank, slot],
                        torch.zeros_like(reduce_buffer[source_rank, slot]),
                    ), f"{name} consumed reduce slot was not cleared"
                else:
                    assert torch.equal(
                        reduce_buffer[source_rank, slot], original[source_rank, slot]
                    ), f"{name} non-local reduce slot changed"


def assert_raises_assertion(expected_substr, fn):
    try:
        fn()
    except AssertionError as error:
        assert expected_substr in str(error), (
            f"expected AssertionError containing {expected_substr!r}, got {error!r}"
        )
        return
    raise AssertionError(f"expected AssertionError containing {expected_substr!r}")


def _run_npu_e2e_body(resources):
    rank, R = setup()
    S, H, K, E = 256, 1024, 4, R * 4
    B = 2
    Hp = 128
    buffer = Buffer(S, H, K, E, R, B=B, num_sms=32)
    resources["buffer"] = buffer
    global_experts = make_global_experts(E, H, Hp)
    sync_prefetch_args = make_prefetch_args(global_experts, B)
    async_prefetch_args = make_prefetch_args(global_experts, B)
    reuse_prefetch_args = make_prefetch_args(global_experts, B)
    no_prefetch_args = make_prefetch_args(global_experts, B)
    hidden, weights, topk, tpe = make_inputs(rank, R, S, H, K, E)

    h_sync, w_sync, cu_sync, plan_sync = buffer.dispatch(hidden, weights, topk, tpe)
    h_sync_snapshot = h_sync.clone()
    w_sync_snapshot = w_sync.clone()
    cu_sync_snapshot = cu_sync.clone()
    assert isinstance(plan_sync, MoonEPCommPlan)
    assert_remote_prefetch_planned(plan_sync, rank, R, K)
    plan_snapshot = plan_sync.clone()
    buffer.prefetch_weight(plan=plan_snapshot, **sync_prefetch_args)
    torch.npu.synchronize()
    assert_prefetched(sync_prefetch_args, plan_snapshot.experts_to_copy[rank])

    hidden2, weights2, topk2, tpe2 = make_inputs(rank, R, S, H, K, E)
    assert torch.equal(hidden, hidden2)
    assert torch.equal(weights, weights2)
    h_async, w_async, cu_async, plan_async, _dispatch_event = buffer.dispatch(
        hidden2, weights2, topk2, tpe2, async_finish=True
    )
    prefetch_event = buffer.prefetch_weight(
        plan=plan_async, async_finish=True, **async_prefetch_args
    )
    prefetch_event.wait(torch.npu.current_stream())
    h_async_snapshot = h_async.clone()
    w_async_snapshot = w_async.clone()
    torch.npu.synchronize()
    assert_prefetched(async_prefetch_args, plan_async.experts_to_copy[rank])

    assert torch.equal(h_sync_snapshot, h_async_snapshot)
    assert torch.equal(w_sync_snapshot, w_async_snapshot)
    assert torch.equal(cu_sync_snapshot, cu_async)
    for name in ("dst", "experts_to_copy", "zero_fill_ranges", "remote_stats"):
        assert torch.equal(getattr(plan_snapshot, name), getattr(plan_async, name))
    assert_dedup_plan_semantic_equal(plan_async, plan_snapshot)

    reuse_dedup = clone_dedup_plan_fields(plan_snapshot)
    h_reuse, w_reuse, cu_reuse, echoed_plan = buffer.dispatch(hidden, plan=plan_snapshot)
    buffer.prefetch_weight(plan=plan_snapshot, **reuse_prefetch_args)
    torch.npu.synchronize()
    assert torch.equal(h_reuse, h_sync_snapshot)
    assert w_reuse is None and cu_reuse is None and echoed_plan is plan_snapshot
    assert dedup_plan_fields_equal(plan_snapshot, reuse_dedup)
    assert_prefetched(reuse_prefetch_args, plan_snapshot.experts_to_copy[rank])

    fill_prefetch_slots(no_prefetch_args, B, 7.0)
    no_prefetch_dedup = clone_dedup_plan_fields(plan_snapshot)
    h_no_prefetch, w_no_prefetch, cu_no_prefetch, no_prefetch_plan = buffer.dispatch(
        hidden, plan=plan_snapshot
    )
    torch.npu.synchronize()
    assert torch.equal(h_no_prefetch, h_sync_snapshot)
    assert w_no_prefetch is None and cu_no_prefetch is None
    assert no_prefetch_plan is plan_snapshot
    assert dedup_plan_fields_equal(plan_snapshot, no_prefetch_dedup)
    assert_prefetch_slots_equal(no_prefetch_args, B, 7.0)

    h_no_prefetch_async, _, _, _, no_prefetch_event = buffer.dispatch(
        hidden, plan=plan_snapshot, async_finish=True
    )
    no_prefetch_event.wait(torch.npu.current_stream())
    torch.npu.synchronize()
    assert torch.equal(h_no_prefetch_async, h_sync_snapshot)

    h_for_combine, w_for_combine, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
    out_sync, _, _ = buffer.combine(plan=plan_snapshot, hidden_nvsh=h_for_combine)
    out_sync_snapshot = out_sync.clone()
    torch.npu.synchronize()

    h_for_combine, w_for_combine, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
    out_with_weights, gathered_weights, _ = buffer.combine(
        plan=plan_snapshot,
        hidden_nvsh=h_for_combine,
        route_weights_nvs=w_for_combine,
    )
    torch.npu.synchronize()
    assert torch.equal(out_sync_snapshot, out_with_weights)
    assert torch.equal(gathered_weights, weights)

    h_zc, w_zc, _, plan_zc = buffer.dispatch(
        hidden, weights, topk, tpe, zero_copy=True
    )
    assert h_zc.data_ptr() == buffer._require_ctx()["hidden_buf_local"].data_ptr()
    assert torch.equal(h_zc, h_for_combine)
    assert torch.equal(w_zc, w_for_combine)
    out_zc, gathered_weights_zc, _ = buffer.combine(
        plan=plan_zc,
        hidden_nvsh=h_zc,
        route_weights_nvs=w_zc,
        zero_copy=True,
    )
    torch.npu.synchronize()
    assert torch.equal(out_sync_snapshot, out_zc)
    assert torch.equal(gathered_weights_zc, weights)
    assert_raises_assertion(
        "alias",
        lambda: buffer.combine(
            plan=plan_zc, hidden_nvsh=h_for_combine, zero_copy=True
        ),
    )

    h_for_combine, _, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
    grad_offsets = (1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
    grad_args = make_grad_reduce_args(R, E, B, H, Hp, grad_offsets)
    out_grad_sync, _, _ = buffer.combine(
        plan=plan_snapshot, hidden_nvsh=h_for_combine
    )
    buffer.reduce_grad(plan=plan_snapshot, **grad_args)
    torch.npu.synchronize()
    assert torch.equal(out_sync_snapshot, out_grad_sync)
    assert_grad_reduced(
        rank, R, E, H, Hp, plan_snapshot.experts_to_copy, grad_args, grad_offsets
    )

    h_for_combine, _, _, _ = buffer.dispatch(hidden, weights, topk, tpe)
    async_offsets = (11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0)
    async_grad_args = make_grad_reduce_args(R, E, B, H, Hp, async_offsets)
    out_async, _, _combine_event = buffer.combine(
        plan=plan_snapshot, hidden_nvsh=h_for_combine, async_finish=True
    )
    reduce_event = buffer.reduce_grad(
        plan=plan_snapshot, async_finish=True, **async_grad_args
    )
    reduce_event.wait(torch.npu.current_stream())
    torch.npu.synchronize()
    assert torch.equal(out_sync_snapshot, out_async)
    assert_grad_reduced(
        rank,
        R,
        E,
        H,
        Hp,
        plan_snapshot.experts_to_copy,
        async_grad_args,
        async_offsets,
    )

    if rank == 0:
        print("[test_npu_e2e] PASS: TileXR matches the MoonEP public E2E contract.")
    buffer.destroy()
    dist.destroy_process_group()


def _cleanup(resources):
    try:
        buffer = resources["buffer"]
        if buffer is not None and not buffer.destroyed:
            buffer.destroy()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def run_npu_e2e():
    validate_topk_routing()
    resources = {"buffer": None}
    try:
        _run_npu_e2e_body(resources)
    finally:
        _cleanup(resources)


def build_launch_command(
    *,
    python_executable,
    script_path,
    nproc_per_node,
    master_addr,
    master_port,
):
    if int(nproc_per_node) <= 0:
        raise ValueError("nproc_per_node must be positive")
    if int(master_port) <= 0 or int(master_port) > 65535:
        raise ValueError("master_port must be in [1, 65535]")
    return [
        str(python_executable),
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        f"--nproc-per-node={int(nproc_per_node)}",
        f"--master-addr={master_addr}",
        f"--master-port={int(master_port)}",
        str(script_path),
    ]


def _free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=int(
            os.environ.get("TILEXR_MOONEP_NPROC_PER_NODE", DEFAULT_NPROC_PER_NODE)
        ),
    )
    parser.add_argument(
        "--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1")
    )
    parser.add_argument(
        "--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "0"))
    )
    return parser.parse_args(argv)


def main(argv=None):
    if "RANK" in os.environ:
        run_npu_e2e()
        return 0

    args = _parse_args(argv)
    master_port = args.master_port or _free_local_port()
    command = build_launch_command(
        python_executable=sys.executable,
        script_path=Path(__file__).resolve(),
        nproc_per_node=args.nproc_per_node,
        master_addr=args.master_addr,
        master_port=master_port,
    )
    return int(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    raise SystemExit(main())
