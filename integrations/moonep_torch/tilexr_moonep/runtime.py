from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .abi import (
    TILEXR_MOONEP_ABI_VERSION,
    TILEXR_MOONEP_FLAG_BUILD_DEDUP,
    TILEXR_MOONEP_FLAG_NONE,
    TILEXR_SUCCESS,
    TileXRMoonEPCombineArgsV1,
    TileXRMoonEPDispatchArgsV1,
    TileXRMoonEPPlanV1,
    TileXRMoonEPPlanningArgsV1,
    TileXRMoonEPPrefetchWeightArgsV1,
    TileXRMoonEPReduceGradArgsV1,
    TileXRMoonEPStage,
    TileXRMoonEPTensorV1,
    initialize_struct,
    make_tensor_v1,
    tensor_nbytes,
    tensor_ptr,
    void_p,
)


class TileXRMoonEPError(RuntimeError):
    def __init__(self, operation: str, ret: int, detail: str = ""):
        suffix = f": {detail}" if detail else ""
        super().__init__(f"{operation} failed with TileXR ret={int(ret)}{suffix}")
        self.operation = operation
        self.ret = int(ret)
        self.detail = detail


@dataclass(frozen=True)
class NativeCapabilities:
    abi_version: int
    stage_mask: int
    stub_mask: int

    def implementation(self, stage: str) -> str:
        bits = {
            "planning": TileXRMoonEPStage.PLANNING,
            "dispatch": TileXRMoonEPStage.DISPATCH,
            "prefetch_weight": TileXRMoonEPStage.PREFETCH_WEIGHT,
            "combine": TileXRMoonEPStage.COMBINE,
            "reduce_grad": TileXRMoonEPStage.REDUCE_GRAD,
        }
        if stage not in bits:
            raise KeyError(f"unknown MoonEP stage: {stage}")
        bit = int(bits[stage])
        if self.stub_mask & bit:
            return "stub"
        if self.stage_mask & bit:
            return "native"
        return "unavailable"

    def as_dict(self) -> dict[str, object]:
        implementations = {
            stage: self.implementation(stage)
            for stage in (
                "planning",
                "dispatch",
                "prefetch_weight",
                "combine",
                "reduce_grad",
            )
        }
        return {
            "abi_version": self.abi_version,
            "stage_mask": self.stage_mask,
            "stub_mask": self.stub_mask,
            "implementations": implementations,
            "transport_correctness_valid": self.transport_correctness_valid,
            "transport_performance_valid": self.transport_performance_valid,
        }

    @property
    def transport_correctness_valid(self) -> bool:
        required = int(
            TileXRMoonEPStage.PLANNING
            | TileXRMoonEPStage.DISPATCH
            | TileXRMoonEPStage.PREFETCH_WEIGHT
            | TileXRMoonEPStage.COMBINE
            | TileXRMoonEPStage.REDUCE_GRAD
        )
        return (self.stage_mask & required) == required and (self.stub_mask & required) == 0

    @property
    def transport_performance_valid(self) -> bool:
        # Native coverage is necessary but does not replace precision and profiling gates.
        return False


def _resolve_install_prefix(value: str | os.PathLike[str] | None) -> Path:
    if value is not None:
        return Path(value).resolve()
    env_value = os.environ.get("TILEXR_INSTALL_PREFIX")
    if env_value:
        return Path(env_value).resolve()
    return Path.cwd().resolve() / "install"


def _resolve_library(
    names: Sequence[str],
    env_name: str,
    install_prefix: Path,
    explicit: str | os.PathLike[str] | None = None,
) -> str:
    if explicit is not None:
        return str(explicit)
    env_path = os.environ.get(env_name)
    if env_path:
        path = Path(env_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{env_name} points to missing library: {path}")
        return str(path)
    candidates = [
        install_prefix / libdir / name
        for name in names
        for libdir in ("lib", "lib64")
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not find {' or '.join(names)}; checked {checked}")


class TileXRMoonEPRuntime:
    """Owns a TileXR communicator and calls the exact V1 ABI from tilexr_moonep.h."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        install_prefix: str | os.PathLike[str] | None = None,
        *,
        library_paths: Mapping[str, str | os.PathLike[str]] | None = None,
        cdll_loader: Callable[..., object] = ctypes.CDLL,
    ):
        if world_size <= 0 or world_size > 128:
            raise ValueError(f"world_size must be in [1, 128], got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.install_prefix = _resolve_install_prefix(install_prefix)
        self._closed = False
        self._comm = ctypes.c_void_p()
        paths = dict(library_paths or {})
        comm_path = _resolve_library(
            ("libtile-comm.so",), "TILEXR_COMM_LIB", self.install_prefix, paths.get("comm")
        )
        planner_path = _resolve_library(
            ("libtilexr-moonep-planner.so.2", "libtilexr-moonep-planner.so"),
            "TILEXR_MOONEP_PLANNER_LIB",
            self.install_prefix,
            paths.get("planner"),
        )
        moonep_path = _resolve_library(
            ("libtilexr-moonep.so.1", "libtilexr-moonep.so"),
            "TILEXR_MOONEP_LIB",
            self.install_prefix,
            paths.get("moonep"),
        )
        self._comm_lib = cdll_loader(comm_path, mode=ctypes.RTLD_GLOBAL)
        self._planner_lib = cdll_loader(planner_path, mode=ctypes.RTLD_GLOBAL)
        self._moonep_lib = cdll_loader(moonep_path, mode=ctypes.RTLD_GLOBAL)
        self._configure_symbols()
        ret = self._comm_lib.TileXRCommInitRankLocal(
            ctypes.c_int(self.world_size), ctypes.c_int(self.rank), ctypes.byref(self._comm)
        )
        self._check(
            "TileXRCommInitRankLocal", ret, f"rank={self.rank} world_size={self.world_size}"
        )
        self.capabilities = self._query_capabilities()

    def _configure_symbols(self) -> None:
        self._comm_lib.TileXRCommInitRankLocal.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._comm_lib.TileXRCommInitRankLocal.restype = ctypes.c_int
        self._comm_lib.TileXRCommDestroy.argtypes = [ctypes.c_void_p]
        self._comm_lib.TileXRCommDestroy.restype = ctypes.c_int
        self._comm_lib.TileXRUDMARegister.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._comm_lib.TileXRUDMARegister.restype = ctypes.c_int
        self._comm_lib.TileXRUDMAUnregister.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        self._comm_lib.TileXRUDMAUnregister.restype = ctypes.c_int
        self._moonep_lib.TileXRMoonEpGetAbiVersion.argtypes = []
        self._moonep_lib.TileXRMoonEpGetAbiVersion.restype = ctypes.c_uint32
        self._moonep_lib.TileXRMoonEpGetCapabilitiesV1.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._moonep_lib.TileXRMoonEpGetCapabilitiesV1.restype = ctypes.c_int
        self._moonep_lib.TileXRMoonEpPlanningGetWorkspaceSizeV1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self._moonep_lib.TileXRMoonEpPlanningGetWorkspaceSizeV1.restype = ctypes.c_int
        symbols = (
            ("TileXRMoonEpPlanningV1", TileXRMoonEPPlanningArgsV1),
            ("TileXRMoonEpDispatchV1", TileXRMoonEPDispatchArgsV1),
            ("TileXRMoonEpPrefetchWeightV1", TileXRMoonEPPrefetchWeightArgsV1),
            ("TileXRMoonEpCombineV1", TileXRMoonEPCombineArgsV1),
            ("TileXRMoonEpReduceGradV1", TileXRMoonEPReduceGradArgsV1),
        )
        for name, args_type in symbols:
            function = getattr(self._moonep_lib, name)
            function.argtypes = [ctypes.POINTER(args_type), ctypes.c_void_p]
            function.restype = ctypes.c_int

    def _check(self, operation: str, ret: int, detail: str = "") -> None:
        if int(ret) != TILEXR_SUCCESS:
            raise TileXRMoonEPError(operation, int(ret), detail)

    def _query_capabilities(self) -> NativeCapabilities:
        abi_version = int(self._moonep_lib.TileXRMoonEpGetAbiVersion())
        if abi_version != TILEXR_MOONEP_ABI_VERSION:
            raise TileXRMoonEPError(
                "TileXRMoonEpGetAbiVersion",
                -1,
                f"expected ABI {TILEXR_MOONEP_ABI_VERSION}, got {abi_version}",
            )
        native_stages = ctypes.c_uint64()
        stub_stages = ctypes.c_uint64()
        ret = self._moonep_lib.TileXRMoonEpGetCapabilitiesV1(
            ctypes.byref(native_stages), ctypes.byref(stub_stages)
        )
        self._check("TileXRMoonEpGetCapabilitiesV1", ret)
        return NativeCapabilities(abi_version, int(native_stages.value), int(stub_stages.value))

    @property
    def comm_ptr(self) -> int:
        if self._closed or not self._comm.value:
            raise TileXRMoonEPError("comm_ptr", -1, "communicator is not initialized")
        return int(self._comm.value)

    @staticmethod
    def _plan_v1(context, plan) -> TileXRMoonEPPlanV1:
        value = initialize_struct(TileXRMoonEPPlanV1())
        value.n = int(plan.n)
        value.r = int(plan.rank_size)
        value.e = int(plan.expert_count)
        value.b = int(plan.prefetch_slots)
        value.nvS = int(plan.nv_s)
        value.k = int(plan.topk)
        value.dst = tensor_ptr(plan.dst)
        value.expertsToCopy = tensor_ptr(plan.experts_to_copy)
        value.zeroFillRanges = tensor_ptr(plan.zero_fill_ranges)
        value.remoteStats = tensor_ptr(plan.remote_stats)
        value.dupGroups = tensor_ptr(plan.dup_groups)
        value.dupLoffs = tensor_ptr(plan.dup_loffs)
        value.dupCounts = tensor_ptr(plan.dup_counts)
        value.status = tensor_ptr(plan.status)
        return value

    def planning_workspace_size(self, context) -> int:
        workspace_bytes = ctypes.c_uint64()
        capacity = ctypes.c_int64()
        ret = self._moonep_lib.TileXRMoonEpPlanningGetWorkspaceSizeV1(
            void_p(self.comm_ptr),
            ctypes.c_int64(context.tokens_per_rank),
            ctypes.c_int64(context.topk),
            ctypes.c_int64(context.expert_count),
            ctypes.c_int64(context.prefetch_slots),
            ctypes.c_int64(context.token_padding),
            ctypes.byref(workspace_bytes),
            ctypes.byref(capacity),
        )
        self._check("TileXRMoonEpPlanningGetWorkspaceSizeV1", ret)
        if int(capacity.value) != int(context.nv_s):
            raise TileXRMoonEPError(
                "TileXRMoonEpPlanningGetWorkspaceSizeV1",
                -1,
                f"expected NvS={context.nv_s}, got {capacity.value}",
            )
        return int(workspace_bytes.value)

    def planning(
        self,
        context,
        topk_experts,
        tokens_per_expert,
        plan,
        cu_seqlens,
        stream_ptr: int,
        wait_iterations: int,
    ) -> None:
        topk = make_tensor_v1(topk_experts)
        tpe = make_tensor_v1(tokens_per_expert)
        plan_v1 = self._plan_v1(context, plan)
        args = initialize_struct(TileXRMoonEPPlanningArgsV1())
        args.comm = void_p(self.comm_ptr)
        args.topkExperts = ctypes.pointer(topk)
        args.tokensPerExpert = ctypes.pointer(tpe)
        args.workspace = tensor_ptr(plan.workspace)
        args.workspaceBytes = tensor_nbytes(plan.workspace)
        cu_v1 = make_tensor_v1(cu_seqlens)
        args.cuSeqlens = ctypes.pointer(cu_v1)
        args.plan = ctypes.pointer(plan_v1)
        args.waitIterations = int(wait_iterations)
        args.flags = TILEXR_MOONEP_FLAG_NONE
        ret = self._moonep_lib.TileXRMoonEpPlanningV1(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpPlanningV1", ret)

    def dispatch(
        self,
        context,
        plan,
        input_tensor,
        output_tensor,
        stream_ptr: int,
        route_weights=None,
        output_route_weights=None,
        *,
        build_dedup: bool,
        inter_rank_sync: bool,
    ) -> None:
        if not inter_rank_sync:
            raise NotImplementedError(
                "TileXR MoonEP does not support inter_rank_sync=False; "
                "peer protocol synchronization is required"
            )
        if (route_weights is None) != (output_route_weights is None):
            raise ValueError(
                "route_weights and output_route_weights must both be provided or both be None"
            )
        plan_v1 = self._plan_v1(context, plan)
        hidden_sh = make_tensor_v1(input_tensor)
        hidden_nvsh = make_tensor_v1(output_tensor)
        weights_sk = make_tensor_v1(route_weights) if route_weights is not None else None
        weights_nvs = (
            make_tensor_v1(output_route_weights) if output_route_weights is not None else None
        )
        args = initialize_struct(TileXRMoonEPDispatchArgsV1())
        args.comm = void_p(self.comm_ptr)
        args.plan = ctypes.pointer(plan_v1)
        args.hiddenSh = ctypes.pointer(hidden_sh)
        args.routeWeightsSk = ctypes.pointer(weights_sk) if weights_sk is not None else None
        args.hiddenNvsh = ctypes.pointer(hidden_nvsh)
        args.routeWeightsNvs = (
            ctypes.pointer(weights_nvs) if weights_nvs is not None else None
        )
        args.flags = (
            TILEXR_MOONEP_FLAG_BUILD_DEDUP
            if build_dedup
            else TILEXR_MOONEP_FLAG_NONE
        )
        ret = self._moonep_lib.TileXRMoonEpDispatchV1(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpDispatchV1", ret)

    def prefetch_weight(self, context, plan, projections, stream_ptr: int) -> None:
        plan_v1 = self._plan_v1(context, plan)
        gate = make_tensor_v1(projections.gate)
        up = make_tensor_v1(projections.up)
        down = make_tensor_v1(projections.down)
        args = initialize_struct(TileXRMoonEPPrefetchWeightArgsV1())
        args.comm = void_p(self.comm_ptr)
        args.plan = ctypes.pointer(plan_v1)
        args.gate = ctypes.pointer(gate)
        args.up = ctypes.pointer(up)
        args.down = ctypes.pointer(down)
        args.flags = TILEXR_MOONEP_FLAG_NONE
        ret = self._moonep_lib.TileXRMoonEpPrefetchWeightV1(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpPrefetchWeightV1", ret)

    def udma_register(self, tensor) -> int:
        handle = ctypes.c_uint32()
        ret = self._comm_lib.TileXRUDMARegister(
            void_p(self.comm_ptr), tensor_ptr(tensor), tensor_nbytes(tensor),
            ctypes.byref(handle)
        )
        self._check("TileXRUDMARegister", ret, f"bytes={tensor_nbytes(tensor)}")
        return int(handle.value)

    def udma_unregister(self, handle: int) -> None:
        ret = self._comm_lib.TileXRUDMAUnregister(
            void_p(self.comm_ptr), ctypes.c_uint32(int(handle))
        )
        self._check("TileXRUDMAUnregister", ret, f"handle={int(handle)}")

    def combine(
        self,
        context,
        plan,
        input_tensor,
        output_tensor,
        stream_ptr: int,
        route_weights=None,
        output_route_weights=None,
        *,
        inter_rank_sync: bool,
    ) -> None:
        if not inter_rank_sync:
            raise NotImplementedError(
                "TileXR MoonEP does not support inter_rank_sync=False; "
                "peer protocol synchronization is required"
            )
        if (route_weights is None) != (output_route_weights is None):
            raise ValueError(
                "route_weights and output_route_weights must both be provided or both be None"
            )
        plan_v1 = self._plan_v1(context, plan)
        hidden_nvsh = make_tensor_v1(input_tensor)
        hidden_sh = make_tensor_v1(output_tensor)
        weights_nvs = make_tensor_v1(route_weights) if route_weights is not None else None
        weights_sk = (
            make_tensor_v1(output_route_weights) if output_route_weights is not None else None
        )
        args = initialize_struct(TileXRMoonEPCombineArgsV1())
        args.comm = void_p(self.comm_ptr)
        args.plan = ctypes.pointer(plan_v1)
        args.hiddenNvsh = ctypes.pointer(hidden_nvsh)
        args.routeWeightsNvs = ctypes.pointer(weights_nvs) if weights_nvs is not None else None
        args.hiddenSh = ctypes.pointer(hidden_sh)
        args.routeWeightsSk = ctypes.pointer(weights_sk) if weights_sk is not None else None
        args.flags = TILEXR_MOONEP_FLAG_NONE
        ret = self._moonep_lib.TileXRMoonEpCombineV1(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpCombineV1", ret)

    def reduce_grad(self, context, plan, gradients, stream_ptr: int) -> None:
        plan_v1 = self._plan_v1(context, plan)
        tensors = {
            name: make_tensor_v1(getattr(gradients, name))
            for name in ("gate", "up", "down", "gate_reduce", "up_reduce", "down_reduce")
        }
        args = initialize_struct(TileXRMoonEPReduceGradArgsV1())
        args.comm = void_p(self.comm_ptr)
        args.plan = ctypes.pointer(plan_v1)
        args.fullGateGrad = ctypes.pointer(tensors["gate"])
        args.fullUpGrad = ctypes.pointer(tensors["up"])
        args.fullDownGrad = ctypes.pointer(tensors["down"])
        args.gateReduceBuffer = ctypes.pointer(tensors["gate_reduce"])
        args.upReduceBuffer = ctypes.pointer(tensors["up_reduce"])
        args.downReduceBuffer = ctypes.pointer(tensors["down_reduce"])
        args.flags = TILEXR_MOONEP_FLAG_NONE
        ret = self._moonep_lib.TileXRMoonEpReduceGradV1(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpReduceGradV1", ret)

    def close(self) -> None:
        if self._closed:
            return
        if self._comm.value:
            ret = self._comm_lib.TileXRCommDestroy(self._comm)
            self._check("TileXRCommDestroy", ret, f"rank={self.rank}")
            self._comm = ctypes.c_void_p()
        self._closed = True

    def __enter__(self) -> "TileXRMoonEPRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
