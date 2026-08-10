from __future__ import annotations

import ctypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .abi import (
    TILEXR_MOONEP_ABI_VERSION,
    TILEXR_MOONEP_FLAG_BUILD_DEDUP,
    TILEXR_MOONEP_FLAG_COMBINE_CONSUME_ONLY,
    TILEXR_MOONEP_FLAG_COMBINE_PUBLISH_ONLY,
    TILEXR_MOONEP_FLAG_NONE,
    TILEXR_SUCCESS,
    TileXRMoonEPCombineArgsV1,
    TileXRMoonEPDispatchArgsV1,
    TileXRMoonEPPlanV1,
    TileXRMoonEPPlanningArgsV1,
    TileXRMoonEPPrefetchWeightArgsV1,
    TileXRMoonEPReduceGradArgsV2,
    TileXRMoonEPReduceGradTransport,
    TileXRMoonEPReduceGradWorkspaceInfoV2,
    TileXRMoonEPReduceGradWorkspaceQueryV2,
    TileXRMoonEPStage,
    TileXRMoonEPTensorV1,
    dtype_code,
    initialize_struct,
    make_tensor_v1,
    tensor_nbytes,
    tensor_ptr,
    void_p,
)


_TILEXR_ERROR_NOT_SUPPORT = -6


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


@dataclass(frozen=True)
class ReduceGradWorkspaceInfo:
    workspace_bytes: int
    workspace_alignment: int
    udma_chunk_bytes: int
    peer_window_bytes: int
    peer_half_bytes: int
    peer_slot_stride_bytes: int
    row_bytes: tuple[int, int, int]
    transports: tuple[str, str, str]
    block_dim: int

    @property
    def uses_udma(self) -> bool:
        return "udma" in self.transports

    def as_dict(self) -> dict[str, object]:
        return {
            "workspace_bytes": self.workspace_bytes,
            "workspace_alignment": self.workspace_alignment,
            "udma_chunk_bytes": self.udma_chunk_bytes,
            "peer_window_bytes": self.peer_window_bytes,
            "peer_half_bytes": self.peer_half_bytes,
            "peer_slot_stride_bytes": self.peer_slot_stride_bytes,
            "row_bytes": dict(zip(("gate", "up", "down"), self.row_bytes)),
            "transports": dict(zip(("gate", "up", "down"), self.transports)),
            "block_dim": self.block_dim,
            "registration_in_timed_path": False,
        }


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
    """Owns a TileXR communicator and calls the versioned MoonEP C ABI."""

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
        self._udma_qp_count = 0
        self._udma_handle: ctypes.c_uint32 | None = None
        self._udma_workspace = None
        self._udma_workspace_ptr = 0
        self._udma_workspace_bytes = 0
        self._active_udma_owner: str | None = None
        self._active_udma_pointer = 0
        self._active_udma_bytes = 0
        self._active_udma_handle: int | None = None
        self._reduce_grad_lock = threading.RLock()
        self._reduce_grad_owner_token = None
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
        self._udma_qp_count = self._query_udma_qp_count()
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
        self._comm_lib.TileXRUDMAGetQpCount.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
        ]
        self._comm_lib.TileXRUDMAGetQpCount.restype = ctypes.c_int
        self._moonep_lib.TileXRMoonEpGetAbiVersion.argtypes = []
        self._moonep_lib.TileXRMoonEpGetAbiVersion.restype = ctypes.c_uint32
        self._moonep_lib.TileXRMoonEpGetCapabilitiesV1.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._moonep_lib.TileXRMoonEpGetCapabilitiesV1.restype = ctypes.c_int
        self._moonep_lib.TileXRMoonEpGetCapabilitiesV2.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._moonep_lib.TileXRMoonEpGetCapabilitiesV2.restype = ctypes.c_int
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
        self._moonep_lib.TileXRMoonEpDispatchGetWorkspaceSizeV1.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._moonep_lib.TileXRMoonEpDispatchGetWorkspaceSizeV1.restype = ctypes.c_int
        symbols = (
            ("TileXRMoonEpPlanningV1", TileXRMoonEPPlanningArgsV1),
            ("TileXRMoonEpDispatchV1", TileXRMoonEPDispatchArgsV1),
            ("TileXRMoonEpPrefetchWeightV1", TileXRMoonEPPrefetchWeightArgsV1),
            ("TileXRMoonEpCombineV1", TileXRMoonEPCombineArgsV1),
        )
        for name, args_type in symbols:
            function = getattr(self._moonep_lib, name)
            function.argtypes = [ctypes.POINTER(args_type), ctypes.c_void_p]
            function.restype = ctypes.c_int
        self._moonep_lib.TileXRMoonEpReduceGradGetWorkspaceSizeV2.argtypes = [
            ctypes.POINTER(TileXRMoonEPReduceGradWorkspaceQueryV2),
            ctypes.POINTER(TileXRMoonEPReduceGradWorkspaceInfoV2),
        ]
        self._moonep_lib.TileXRMoonEpReduceGradGetWorkspaceSizeV2.restype = ctypes.c_int
        self._moonep_lib.TileXRMoonEpReduceGradV2.argtypes = [
            ctypes.POINTER(TileXRMoonEPReduceGradArgsV2),
            ctypes.c_void_p,
        ]
        self._moonep_lib.TileXRMoonEpReduceGradV2.restype = ctypes.c_int

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
        ret = self._moonep_lib.TileXRMoonEpGetCapabilitiesV2(
            ctypes.byref(native_stages), ctypes.byref(stub_stages)
        )
        self._check("TileXRMoonEpGetCapabilitiesV2", ret)
        return NativeCapabilities(abi_version, int(native_stages.value), int(stub_stages.value))

    def _query_udma_qp_count(self) -> int:
        qp_count = ctypes.c_uint32()
        ret = self._comm_lib.TileXRUDMAGetQpCount(
            void_p(self.comm_ptr), ctypes.byref(qp_count)
        )
        if int(ret) == _TILEXR_ERROR_NOT_SUPPORT:
            return 0
        self._check("TileXRUDMAGetQpCount", ret)
        if qp_count.value == 0:
            raise TileXRMoonEPError(
                "TileXRUDMAGetQpCount", -1,
                "successful UDMA query returned zero QPs",
            )
        return int(qp_count.value)

    @property
    def comm_ptr(self) -> int:
        if self._closed or not self._comm.value:
            raise TileXRMoonEPError("comm_ptr", -1, "communicator is not initialized")
        return int(self._comm.value)

    @property
    def udma_qp_count(self) -> int:
        return self._udma_qp_count

    def _acquire_reduce_grad(self, owner_token: object) -> None:
        with self._reduce_grad_lock:
            if self._closed or not self._comm.value:
                raise RuntimeError("cannot launch ReduceGrad on a closed TileXR MoonEP runtime")
            if self._reduce_grad_owner_token is not None:
                raise RuntimeError(
                    "ReduceGrad is already in flight on this TileXR MoonEP runtime; "
                    "synchronize the owning buffer before launching another ReduceGrad"
                )
            self._reduce_grad_owner_token = owner_token

    def _release_reduce_grad(self, owner_token: object) -> None:
        with self._reduce_grad_lock:
            if self._reduce_grad_owner_token is not owner_token:
                raise RuntimeError("ReduceGrad owner token mismatch")
            self._reduce_grad_owner_token = None

    def _require_reduce_grad_workspace_owner(
        self, owner_token: object | None, operation: str
    ) -> None:
        if (
            self._reduce_grad_owner_token is not None
            and self._reduce_grad_owner_token is not owner_token
        ):
            raise RuntimeError(
                f"{operation} cannot modify the ReduceGrad workspace while another "
                "ReduceGrad is in flight"
            )

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

    def dispatch_workspace_size(self, context) -> tuple[int, int]:
        workspace_bytes = ctypes.c_uint64()
        workspace_alignment = ctypes.c_uint64()
        # The V1 query has no NvS argument. Rounding NvS to complete top-k rows
        # produces a conservative layout when token padding makes NvS > S*K.
        query_s = max(
            int(context.tokens_per_rank),
            (int(context.nv_s) + int(context.topk) - 1) // int(context.topk),
        )
        ret = self._moonep_lib.TileXRMoonEpDispatchGetWorkspaceSizeV1(
            void_p(self.comm_ptr),
            ctypes.c_int64(query_s),
            ctypes.c_int64(context.topk),
            ctypes.c_int64(context.hidden_size),
            ctypes.c_uint32(dtype_code(context.dtype)),
            ctypes.byref(workspace_bytes),
            ctypes.byref(workspace_alignment),
        )
        self._check("TileXRMoonEpDispatchGetWorkspaceSizeV1", ret)
        return int(workspace_bytes.value), int(workspace_alignment.value)

    def register_dispatch_workspace(self, pointer: int, size: int) -> int | None:
        return self._activate_udma_region(
            pointer, size, "dispatch", f"workspace_bytes={int(size)}"
        )

    def _activate_udma_region(
        self, pointer: int, size: int, owner: str, detail: str
    ) -> int | None:
        pointer = int(pointer)
        size = int(size)
        if self.world_size == 1:
            return None
        if pointer <= 0 or size <= 0:
            raise ValueError("UDMA registration requires a valid pointer and positive size")
        if self._active_udma_pointer == pointer and self._active_udma_bytes == size:
            self._active_udma_owner = owner
            return self._active_udma_handle
        handle = ctypes.c_uint32()
        ret = self._comm_lib.TileXRUDMARegister(
            void_p(self.comm_ptr), void_p(pointer), ctypes.c_size_t(size),
            ctypes.byref(handle)
        )
        self._check("TileXRUDMARegister", ret, detail)
        self._active_udma_owner = owner
        self._active_udma_pointer = pointer
        self._active_udma_bytes = size
        self._active_udma_handle = int(handle.value)
        return self._active_udma_handle

    def _deactivate_udma_region(self, owner: str | None = None) -> None:
        if self._active_udma_handle is None:
            return
        if owner is not None and self._active_udma_owner != owner:
            return
        ret = self._comm_lib.TileXRUDMAUnregister(
            void_p(self.comm_ptr), ctypes.c_uint32(self._active_udma_handle)
        )
        self._check("TileXRUDMAUnregister", ret)
        self._active_udma_owner = None
        self._active_udma_pointer = 0
        self._active_udma_bytes = 0
        self._active_udma_handle = None

    def unregister_dispatch_workspace(self, handle: int | None) -> None:
        if handle is None:
            return
        self._deactivate_udma_region("dispatch")

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
        build_dedup: bool = False,
        inter_rank_sync: bool = True,
        registered_workspace: int | None = None,
        registered_workspace_bytes: int = 0,
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
        if (registered_workspace is None) != (registered_workspace_bytes == 0):
            raise ValueError(
                "registered_workspace and registered_workspace_bytes must be provided together"
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
            if build_dedup and registered_workspace is None
            else TILEXR_MOONEP_FLAG_NONE
        )
        args.registeredWorkspace = void_p(registered_workspace)
        args.registeredWorkspaceBytes = int(registered_workspace_bytes)
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
        size = tensor_nbytes(tensor)
        handle = self._activate_udma_region(
            int(tensor.data_ptr()), size, "projection", f"bytes={size}"
        )
        return 0 if handle is None else int(handle)

    def udma_unregister(self, handle: int) -> None:
        self._deactivate_udma_region("projection")

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
        flags: int = TILEXR_MOONEP_FLAG_NONE,
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
        allowed_flags = (
            TILEXR_MOONEP_FLAG_COMBINE_PUBLISH_ONLY |
            TILEXR_MOONEP_FLAG_COMBINE_CONSUME_ONLY
        )
        if int(flags) & ~allowed_flags:
            raise ValueError(f"unsupported TileXR MoonEP Combine flags: {int(flags)}")
        args.flags = int(flags)
        ret = self._moonep_lib.TileXRMoonEpCombineV1(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpCombineV1", ret)

    def reduce_grad_workspace_info(
        self,
        context,
        plan,
        gradients,
        *,
        requested_udma_chunk_bytes: int = 0,
    ) -> ReduceGradWorkspaceInfo:
        plan_v1 = self._plan_v1(context, plan)
        descriptors = [
            make_tensor_v1(getattr(gradients, name)) for name in ("gate", "up", "down")
        ]
        query = initialize_struct(TileXRMoonEPReduceGradWorkspaceQueryV2())
        query.comm = void_p(self.comm_ptr)
        query.plan = ctypes.pointer(plan_v1)
        query.gate = ctypes.pointer(descriptors[0])
        query.up = ctypes.pointer(descriptors[1])
        query.down = ctypes.pointer(descriptors[2])
        query.requestedUdmaChunkBytes = int(requested_udma_chunk_bytes)
        query.flags = TILEXR_MOONEP_FLAG_NONE
        info = initialize_struct(TileXRMoonEPReduceGradWorkspaceInfoV2())
        ret = self._moonep_lib.TileXRMoonEpReduceGradGetWorkspaceSizeV2(
            ctypes.byref(query), ctypes.byref(info)
        )
        self._check("TileXRMoonEpReduceGradGetWorkspaceSizeV2", ret)
        transport_names = {
            int(TileXRMoonEPReduceGradTransport.NONE): "none",
            int(TileXRMoonEPReduceGradTransport.PEER): "peer",
            int(TileXRMoonEPReduceGradTransport.UDMA): "udma",
        }
        try:
            transports = tuple(transport_names[int(value)] for value in info.transports)
        except KeyError as exc:
            raise TileXRMoonEPError(
                "TileXRMoonEpReduceGradGetWorkspaceSizeV2", -1,
                f"unknown transport code {int(exc.args[0])}",
            ) from exc
        return ReduceGradWorkspaceInfo(
            workspace_bytes=int(info.workspaceBytes),
            workspace_alignment=int(info.workspaceAlignment),
            udma_chunk_bytes=int(info.udmaChunkBytes),
            peer_window_bytes=int(info.peerWindowBytes),
            peer_half_bytes=int(info.peerHalfBytes),
            peer_slot_stride_bytes=int(info.peerSlotStrideBytes),
            row_bytes=tuple(int(value) for value in info.rowBytes),
            transports=transports,
            block_dim=int(info.blockDim),
        )

    def register_reduce_grad_workspace(
        self, workspace, required_bytes: int, *, owner_token: object | None = None
    ) -> None:
        with self._reduce_grad_lock:
            self._require_reduce_grad_workspace_owner(
                owner_token, "register_reduce_grad_workspace"
            )
            pointer = int(workspace.data_ptr())
            available = tensor_nbytes(workspace)
            if required_bytes <= 0 or available < required_bytes:
                raise ValueError(
                    f"ReduceGrad workspace has {available} bytes, requires {required_bytes}"
                )
            if self._udma_handle is not None and (
                pointer != self._udma_workspace_ptr or
                available != self._udma_workspace_bytes
            ):
                raise RuntimeError("a different TileXR UDMA workspace is already registered")
            handle = self._activate_udma_region(
                pointer, available, "reduce_grad", f"workspace_bytes={available}"
            )
            self._udma_handle = None if handle is None else ctypes.c_uint32(handle)
            self._udma_workspace = workspace
            self._udma_workspace_ptr = pointer
            self._udma_workspace_bytes = available

    def unregister_reduce_grad_workspace(
        self, *, owner_token: object | None = None
    ) -> None:
        with self._reduce_grad_lock:
            self._require_reduce_grad_workspace_owner(
                owner_token, "unregister_reduce_grad_workspace"
            )
            if self._udma_handle is None:
                return
            self._deactivate_udma_region("reduce_grad")
            self._udma_handle = None
            self._udma_workspace = None
            self._udma_workspace_ptr = 0
            self._udma_workspace_bytes = 0

    def reduce_grad(
        self,
        context,
        plan,
        gradients,
        workspace,
        stream_ptr: int,
        wait_iterations: int,
        *,
        requested_udma_chunk_bytes: int = 0,
    ) -> None:
        plan_v1 = self._plan_v1(context, plan)
        descriptors = [
            make_tensor_v1(getattr(gradients, name)) for name in ("gate", "up", "down")
        ]
        status = make_tensor_v1(plan.reduce_grad_status)
        args = initialize_struct(TileXRMoonEPReduceGradArgsV2())
        args.comm = void_p(self.comm_ptr)
        args.plan = ctypes.pointer(plan_v1)
        args.gate = ctypes.pointer(descriptors[0])
        args.up = ctypes.pointer(descriptors[1])
        args.down = ctypes.pointer(descriptors[2])
        args.workspace = void_p(None if workspace is None else int(workspace.data_ptr()))
        args.workspaceBytes = 0 if workspace is None else tensor_nbytes(workspace)
        args.status = ctypes.pointer(status)
        args.waitIterations = int(wait_iterations)
        args.requestedUdmaChunkBytes = int(requested_udma_chunk_bytes)
        args.flags = TILEXR_MOONEP_FLAG_NONE
        ret = self._moonep_lib.TileXRMoonEpReduceGradV2(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpReduceGradV2", ret)

    def close(self) -> None:
        with self._reduce_grad_lock:
            if self._reduce_grad_owner_token is not None:
                raise RuntimeError(
                    "cannot close TileXR MoonEP runtime while ReduceGrad is in flight; "
                    "synchronize the owning buffer first"
                )
            if self._closed:
                return
            if self._comm.value:
                self.unregister_reduce_grad_workspace()
                self._deactivate_udma_region()
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
