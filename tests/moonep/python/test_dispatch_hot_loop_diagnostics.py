from __future__ import annotations

import ctypes
from types import SimpleNamespace

from tools.moonep import dispatch_hot_loop


def _profile(payload_mode: int, *, shared_owner: bool):
    record = dispatch_hot_loop.DispatchProfileRecord()
    record.marker = dispatch_hot_loop.PROFILE_MARKER
    record.version = dispatch_hot_loop.DIAGNOSTIC_VERSION
    record.record_bytes = ctypes.sizeof(dispatch_hot_loop.DispatchProfileRecord)
    record.payload_mode = payload_mode
    record.rank = 0
    record.core = 0
    record.block_dim = 1
    record.select_mode = 1
    record.fallback_reason = 0
    record.magic = 17
    record.processed = 8
    record.put_count = 8
    if shared_owner:
        record.scanned = 16
        record.matched = 8
        record.selected = 8
        record.visited_peers = 1
        record.completion_flags = 2
        record.kernel_cycles = 110
        record.staging_cycles = 10
        record.put_issue_cycles = 20
        record.flag_wait_cycles = 30
        record.output_copy_cycles = 40
        record.quiet_cycles = 10
        for index in range(11):
            record.reserved[index] = (index + 1) * 10
    return record


def _dfx(payload_mode: int):
    record = dispatch_hot_loop.DispatchDfxRecord()
    record.marker = dispatch_hot_loop.DFX_MARKER
    record.version = dispatch_hot_loop.DIAGNOSTIC_VERSION
    record.record_bytes = ctypes.sizeof(dispatch_hot_loop.DispatchDfxRecord)
    record.payload_mode = payload_mode
    record.rank = 0
    record.core = 0
    record.magic = 17
    record.expected_routes = 16
    record.processed_routes = 8
    return record


def test_paired_fused_diagnostics_marks_one_shared_epoch_owner(monkeypatch) -> None:
    monkeypatch.setenv("TILEXR_MOONEP_DISPATCH_AIV_CORE_COUNT", "1")
    context = SimpleNamespace(
        planner_group_rank=0,
        _dispatch_workspace_bytes=(
            dispatch_hot_loop.COMPLETION_BYTES
            + dispatch_hot_loop.SIGNAL_BYTES
            + 2 * dispatch_hot_loop.PROFILE_COUNT
            * ctypes.sizeof(dispatch_hot_loop.DispatchProfileRecord)
            + 2 * dispatch_hot_loop.DFX_COUNT
            * ctypes.sizeof(dispatch_hot_loop.DispatchDfxRecord)
            + ctypes.sizeof(dispatch_hot_loop.DispatchKernelStatus)
        ),
    )
    offsets = dispatch_hot_loop._diagnostic_layout(context)

    hidden_profile = bytearray(
        dispatch_hot_loop.PROFILE_COUNT
        * ctypes.sizeof(dispatch_hot_loop.DispatchProfileRecord)
    )
    weight_profile = bytearray(hidden_profile)
    hidden_profile[: ctypes.sizeof(dispatch_hot_loop.DispatchProfileRecord)] = bytes(
        _profile(0, shared_owner=False)
    )
    weight_profile[: ctypes.sizeof(dispatch_hot_loop.DispatchProfileRecord)] = bytes(
        _profile(1, shared_owner=True)
    )
    hidden_dfx = bytearray(
        dispatch_hot_loop.DFX_COUNT
        * ctypes.sizeof(dispatch_hot_loop.DispatchDfxRecord)
    )
    weight_dfx = bytearray(hidden_dfx)
    hidden_dfx[: ctypes.sizeof(dispatch_hot_loop.DispatchDfxRecord)] = bytes(
        _dfx(0)
    )
    weight_dfx[: ctypes.sizeof(dispatch_hot_loop.DispatchDfxRecord)] = bytes(
        _dfx(1)
    )
    status = dispatch_hot_loop.DispatchKernelStatus()
    status.marker = dispatch_hot_loop.KERNEL_STATUS_MARKER
    status.version = dispatch_hot_loop.DIAGNOSTIC_VERSION
    status.record_bytes = ctypes.sizeof(dispatch_hot_loop.DispatchKernelStatus)
    status.payload_mode = 1
    status.magic = 17
    status.reserved[0] = (
        dispatch_hot_loop.KERNEL_STATUS_FEATURE_DFX_ENABLED
        | dispatch_hot_loop.KERNEL_STATUS_FEATURE_PROFILING_ENABLED
        | dispatch_hot_loop.KERNEL_STATUS_FEATURE_FUSED_EPOCH
    )
    blobs = {
        offsets["hidden_profile"]: bytes(hidden_profile),
        offsets["weight_profile"]: bytes(weight_profile),
        offsets["hidden_dfx"]: bytes(hidden_dfx),
        offsets["weight_dfx"]: bytes(weight_dfx),
        offsets["kernel_status"]: bytes(status),
    }

    monkeypatch.setattr(
        dispatch_hot_loop,
        "_workspace_blob",
        lambda unused_context, byte_offset, byte_count: blobs[byte_offset][
            :byte_count
        ],
    )

    result = dispatch_hot_loop._diagnostics(context)

    assert result["fused_epoch"] is True
    assert result["shared_owner_mode"] == "weight"
    assert result["kernel_status"]["payload_mode"] == 1
    assert result["hidden"]["shared_epoch_owner"] is False
    assert result["weight"]["shared_epoch_owner"] is True
    assert result["hidden"]["profile"][0]["kernel_cycles"] == 0
    assert result["hidden"]["profile"][0]["reserved"] == [0] * 11
    assert result["weight"]["profile"][0]["kernel_cycles"] == 110
    assert len(result["hidden"]["dfx"]) == 1
    assert len(result["weight"]["dfx"]) == 1


def test_non_fused_diagnostics_do_not_claim_a_shared_owner(monkeypatch) -> None:
    monkeypatch.setenv("TILEXR_MOONEP_DISPATCH_AIV_CORE_COUNT", "1")
    context = SimpleNamespace(
        planner_group_rank=0,
        _dispatch_workspace_bytes=(
            dispatch_hot_loop.COMPLETION_BYTES
            + dispatch_hot_loop.SIGNAL_BYTES
            + 2 * dispatch_hot_loop.PROFILE_COUNT
            * ctypes.sizeof(dispatch_hot_loop.DispatchProfileRecord)
            + 2 * dispatch_hot_loop.DFX_COUNT
            * ctypes.sizeof(dispatch_hot_loop.DispatchDfxRecord)
            + ctypes.sizeof(dispatch_hot_loop.DispatchKernelStatus)
        ),
    )
    offsets = dispatch_hot_loop._diagnostic_layout(context)
    status = dispatch_hot_loop.DispatchKernelStatus()
    status.marker = dispatch_hot_loop.KERNEL_STATUS_MARKER
    status.version = dispatch_hot_loop.DIAGNOSTIC_VERSION
    status.record_bytes = ctypes.sizeof(dispatch_hot_loop.DispatchKernelStatus)
    status.payload_mode = 0
    status.magic = 17
    status.reserved[0] = dispatch_hot_loop.KERNEL_STATUS_FEATURE_DFX_ENABLED
    empty_profile = bytes(
        dispatch_hot_loop.PROFILE_COUNT
        * ctypes.sizeof(dispatch_hot_loop.DispatchProfileRecord)
    )
    empty_dfx = bytes(
        dispatch_hot_loop.DFX_COUNT
        * ctypes.sizeof(dispatch_hot_loop.DispatchDfxRecord)
    )
    blobs = {
        offsets["hidden_profile"]: empty_profile,
        offsets["weight_profile"]: empty_profile,
        offsets["hidden_dfx"]: empty_dfx,
        offsets["weight_dfx"]: empty_dfx,
        offsets["kernel_status"]: bytes(status),
    }
    monkeypatch.setattr(
        dispatch_hot_loop,
        "_workspace_blob",
        lambda unused_context, byte_offset, byte_count: blobs[byte_offset][
            :byte_count
        ],
    )

    result = dispatch_hot_loop._diagnostics(
        context, required_modes=()
    )

    assert result["fused_epoch"] is False
    assert result["shared_owner_mode"] is None
    assert result["hidden"]["shared_epoch_owner"] is False
    assert result["weight"]["shared_epoch_owner"] is False


def test_skewed_hot_loop_routes_concentrate_on_the_same_experts() -> None:
    case = SimpleNamespace(
        tokens_per_rank=3,
        topk=2,
        expert_count=8,
        routing_pattern="skewed",
        route_distribution="rank_shifted_uniform",
    )

    rank_zero = dispatch_hot_loop._case_rank_topk(case, 0, 4)
    rank_three = dispatch_hot_loop._case_rank_topk(case, 3, 4)

    assert rank_zero == (0, 1, 0, 1, 0, 1)
    assert rank_three == rank_zero
    assert dispatch_hot_loop._all_case_topk(case, 4) == rank_zero * 4


def test_reference_slots_leave_padding_for_zero_fill() -> None:
    case = SimpleNamespace(
        tokens_per_rank=2,
        topk=1,
        expert_count=2,
        routing_pattern="balanced",
        route_distribution="moonep_combine_balanced",
    )
    context = SimpleNamespace(
        planner_group_rank=0,
        planner_group_size=2,
        nv_s=8,
        prefetch_slots=1,
        token_padding=4,
    )

    sources, tokens, routes = dispatch_hot_loop._reference_slot_assignments(
        case, context
    )

    assert len(sources) == len(tokens) == len(routes) == context.nv_s
    assert sum(source >= 0 for source in sources) == 2
    assert sum(source < 0 for source in sources) == 6


class _FakePlan:
    def __init__(self) -> None:
        import torch

        self.dst = torch.tensor([0, 1], dtype=torch.int32)


class _FakeBuffer:
    def __init__(self) -> None:
        self.runtime = SimpleNamespace()
        self.context = SimpleNamespace(dispatch_workspace=(object(), 4096))
        self.quiesce_calls = 0

    def _stream_ptr(self):
        return object()

    def quiesce(self) -> None:
        self.quiesce_calls += 1


def test_repeated_exact_check_compares_every_round_and_alternates_plan(
    monkeypatch,
) -> None:
    import torch

    case = SimpleNamespace(topk=1)
    plan = _FakePlan()
    buffer = _FakeBuffer()
    hidden_out = torch.zeros((2, 1), dtype=torch.bfloat16)
    weights_out = torch.zeros((2,), dtype=torch.float32)
    observed = []

    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None),
        raising=False)
    monkeypatch.setattr(
        dispatch_hot_loop,
        "_launch_dispatch",
        lambda unused_runtime, unused_context, active_plan, *unused_args: (
            observed.append(tuple(active_plan.dst.tolist()))
        ),
    )
    monkeypatch.setattr(
        dispatch_hot_loop,
        "_expected",
        lambda *unused_args, destination_roll=0: (
            hidden_out.clone(), weights_out.clone(), [0, 0]
        ),
    )

    result = dispatch_hot_loop._repeated_exact_check(
        torch, case, buffer, plan, object(), hidden_out, object(), weights_out,
        "pair", 4
    )

    assert result["rounds"] == 4
    assert result["hidden_exact"] is True
    assert result["weight_exact"] is True
    assert observed == [(0, 1), (1, 0), (0, 1), (1, 0)]
    assert buffer.quiesce_calls == 4
    assert plan.dst.tolist() == [0, 1]
