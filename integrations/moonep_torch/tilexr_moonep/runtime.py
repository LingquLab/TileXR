from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .abi import (
    TILEXR_MOONEP_ABI_VERSION,
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
            "transport_performance_valid": self.transport_performance_valid,
        }

    @property
    def transport_performance_valid(self) -> bool:
        required = int(
            TileXRMoonEPStage.PLANNING
            | TileXRMoonEPStage.DISPATCH
            | TileXRMoonEPStage.PREFETCH_WEIGHT
            | TileXRMoonEPStage.COMBINE
            | TileXRMoonEPStage.REDUCE_GRAD
        )
        return (self.stage_mask & required) == required and self.stub_mask == 0


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
        value.s = int(context.tokens_per_rank)
        value.k = int(context.topk)
        value.e = int(context.expert_count)
        value.b = int(context.experts_per_rank)
        value.rank = int(context.planner_group_rank)
        value.world = int(context.planner_group_size)
        value.dispatchedCapacity = int(context.dispatched_capacity)
        value.dst = tensor_ptr(plan.dst)
        value.cu = tensor_ptr(plan.cu_seqlens)
        value.expertsToCopy = tensor_ptr(plan.experts_to_copy)
        value.remoteStats = tensor_ptr(plan.remote_stats)
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
            ctypes.byref(workspace_bytes),
            ctypes.byref(capacity),
        )
        self._check("TileXRMoonEpPlanningGetWorkspaceSizeV1", ret)
        if int(capacity.value) != int(context.dispatched_capacity):
            raise TileXRMoonEPError(
                "TileXRMoonEpPlanningGetWorkspaceSizeV1",
                -1,
                f"expected NvS={context.dispatched_capacity}, got {capacity.value}",
            )
        return int(workspace_bytes.value)

    def planning(
        self,
        context,
        topk_experts,
        tokens_per_expert,
        plan,
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
        args.plan = ctypes.pointer(plan_v1)
        args.waitIterations = int(wait_iterations)
        args.flags = TILEXR_MOONEP_FLAG_NONE
        ret = self._moonep_lib.TileXRMoonEpPlanningV1(
            ctypes.byref(args), void_p(stream_ptr)
        )
        self._check("TileXRMoonEpPlanningV1", ret)

    def _run_stage(self, symbol: str, args_type, context, plan, input_tensor, output_tensor, stream_ptr: int) -> None:
        plan_v1 = self._plan_v1(context, plan)
        input_v1 = make_tensor_v1(input_tensor)
        output_v1 = make_tensor_v1(output_tensor)
        args = initialize_struct(args_type())
        args.comm = void_p(self.comm_ptr)
        args.plan = ctypes.pointer(plan_v1)
        args.input = ctypes.pointer(input_v1)
        args.output = ctypes.pointer(output_v1)
        args.flags = TILEXR_MOONEP_FLAG_NONE
        ret = getattr(self._moonep_lib, symbol)(ctypes.byref(args), void_p(stream_ptr))
        self._check(symbol, ret)

    def dispatch(
        self,
        context,
        plan,
        input_tensor,
        output_tensor,
        stream_ptr: int,
        route_weights=None,
        output_route_weights=None,
    ) -> None:
        if (route_weights is None) != (output_route_weights is None):
            raise ValueError(
                "route_weights and output_route_weights must both be provided or both be None"
            )
        self._run_stage(
            "TileXRMoonEpDispatchV1",
            TileXRMoonEPDispatchArgsV1,
            context,
            plan,
            input_tensor,
            output_tensor,
            stream_ptr,
        )
        if route_weights is not None:
            self._run_stage(
                "TileXRMoonEpDispatchV1",
                TileXRMoonEPDispatchArgsV1,
                context,
                plan,
                route_weights,
                output_route_weights,
                stream_ptr,
            )

    def prefetch_weight(self, context, plan, inputs, outputs, stream_ptr: int) -> None:
        for name in ("gate", "up", "down"):
            self._run_stage(
                "TileXRMoonEpPrefetchWeightV1",
                TileXRMoonEPPrefetchWeightArgsV1,
                context,
                plan,
                getattr(inputs, name),
                getattr(outputs, name),
                stream_ptr,
            )

    def combine(
        self,
        context,
        plan,
        input_tensor,
        output_tensor,
        stream_ptr: int,
        route_weights=None,
        output_route_weights=None,
    ) -> None:
        if (route_weights is None) != (output_route_weights is None):
            raise ValueError(
                "route_weights and output_route_weights must both be provided or both be None"
            )
        self._run_stage(
            "TileXRMoonEpCombineV1",
            TileXRMoonEPCombineArgsV1,
            context,
            plan,
            input_tensor,
            output_tensor,
            stream_ptr,
        )
        if route_weights is not None:
            self._run_stage(
                "TileXRMoonEpCombineV1",
                TileXRMoonEPCombineArgsV1,
                context,
                plan,
                route_weights,
                output_route_weights,
                stream_ptr,
            )

    def reduce_grad(self, context, plan, gradients, stream_ptr: int) -> None:
        for input_name, output_name in (
            ("gate", "gate_reduce"),
            ("up", "up_reduce"),
            ("down", "down_reduce"),
        ):
            self._run_stage(
                "TileXRMoonEpReduceGradV1",
                TileXRMoonEPReduceGradArgsV1,
                context,
                plan,
                getattr(gradients, input_name),
                getattr(gradients, output_name),
                stream_ptr,
            )

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
