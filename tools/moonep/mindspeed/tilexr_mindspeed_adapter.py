from __future__ import annotations

import base64
import hashlib
import json
import os
import zlib

import moonep as upstream_moonep
from tilexr_moonep import Buffer as TileXRBuffer
from tilexr_moonep import ProjectionBuffers

from .moonep_backend import (
    MOONEP_NATIVE_NPU_CAPABILITY,
    MoonEPBufferFlexBackend,
)
from .mindspeed_stage_barrier import optional_stage_barrier


_UDMA_COMPAT_REGISTRATION_BYTES = 2 * 1024 * 1024
_FRAMEWORK_OPS_PREWARMED = False


def _prewarm_framework_ops():
    global _FRAMEWORK_OPS_PREWARMED
    if (
        _FRAMEWORK_OPS_PREWARMED
        or os.environ.get("TILEXR_MINDSPEED_PREWARM_FRAMEWORK_OPS", "0") != "1"
    ):
        return

    import torch

    # Load framework kernels before TileXR reserves its registered communication arena.
    probe = torch.empty((1,), dtype=torch.float32, device="npu")
    difference = probe - probe
    equal = difference == probe
    torch.all(equal)
    torch.npu.synchronize()
    _FRAMEWORK_OPS_PREWARMED = True
    print("TILEXR_MINDSPEED_FRAMEWORK_OPS_PREWARM=complete", flush=True)


def _retained_route_inputs(native_buffer, plan):
    native_plan = plan._require_native()
    for refs in reversed(native_buffer._pending_refs):
        if len(refs) >= 3 and refs[0] is native_plan:
            return refs[1], refs[2]
    return None, None


def _tensor_descriptor(tensor):
    if tensor is None:
        return None
    return {
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "contiguous": bool(tensor.is_contiguous()),
        "storage_offset": int(tensor.storage_offset()),
    }


def _reject_upstream_buffer_init(*args, **kwargs):
    del args, kwargs
    raise RuntimeError(
        "TileXR MindSpeed mode forbids construction of the SHMEM communication Buffer"
    )


if not getattr(upstream_moonep.Buffer, "_tilexr_mindspeed_guard", False):
    upstream_moonep.Buffer.__init__ = _reject_upstream_buffer_init
    upstream_moonep.Buffer._tilexr_mindspeed_guard = True


class MindSpeedTileXRBuffer(TileXRBuffer):
    """MR3832 Buffer surface backed by TileXR communication resources."""

    __mindspeed_external_communication_owner__ = True

    def __init__(self, *args, token_buffer_count=1, **kwargs):
        if isinstance(token_buffer_count, bool) or int(token_buffer_count) <= 0:
            raise ValueError("token_buffer_count must be a positive integer")
        super().__init__(*args, **kwargs)
        self.token_buffer_count = int(token_buffer_count)
        device = f"npu:{self._context.device_index}"
        self.token_buffers = tuple(
            self._torch.empty(
                (self._context.nv_s, self.H),
                dtype=self._torch.bfloat16,
                device=device,
            )
            for _ in range(self.token_buffer_count)
        )
        self.route_buffers = tuple(
            self._torch.zeros(
                (self._context.nv_s,),
                dtype=self._torch.float32,
                device=device,
            )
            for _ in range(self.token_buffer_count)
        )
        self._route_by_token_ptr = {
            int(token.data_ptr()): route
            for token, route in zip(self.token_buffers, self.route_buffers)
        }
        self._packed_projections = None
        self._packed_projection_signature = None
        self._reduce_dummy = None
        self._reduce_dummy_buffer = None
        self._reduce_dummy_buffer_allocation = None
        self._tilexr_remote_prefetches = 0
        self._plan_owner_token = object()
        self._dispatch_generation = 0
        self._finite_check_sequence = 0
        self._route_capture_count = 0
        self._route_capture_seen = 0
        self._ctx = self._require_ctx()
        if os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1":
            print(
                f"[TileXR MindSpeed rank {self._context.planner_group_rank}] "
                f"buffer_owner=TileXRComm boundaries={self.token_buffer_count}",
                flush=True,
            )

    def _require_ctx(self):
        context = super()._require_ctx()
        context.update(
            rank=int(self._context.planner_group_rank),
            num_sms=int(self.num_sms),
            sparse_epoch_prefetch_batch2=False,
            direct_local_reduce_batch3=False,
        )
        return context

    def _validate_boundary(self, hidden_buffer):
        if hidden_buffer is None:
            return None
        if int(hidden_buffer.data_ptr()) not in self._route_by_token_ptr:
            raise RuntimeError("hidden_buffer is not owned by this TileXR Buffer")
        if (
            tuple(hidden_buffer.shape) != (self._context.nv_s, self.H)
            or hidden_buffer.dtype != self._torch.bfloat16
            or not hidden_buffer.is_contiguous()
        ):
            raise RuntimeError("hidden_buffer has an incompatible TileXR boundary layout")
        return self._route_by_token_ptr[int(hidden_buffer.data_ptr())]

    def _bind_zero_copy_views(self, plan, hidden, route_weights):
        self._dispatch_generation += 1
        hidden_alias = hidden.view_as(hidden)
        hidden_alias._moonep_buffer_owner = self._plan_owner_token
        hidden_alias._moonep_dispatch_generation = self._dispatch_generation
        hidden_alias._moonep_dispatch_plan = plan
        route_alias = (
            route_weights.view_as(route_weights)
            if route_weights is not None
            else None
        )
        return hidden_alias, route_alias

    def _dump_native_plan_once(self, plan):
        dump_dir = os.environ.get("TILEXR_MINDSPEED_PLAN_DUMP_DIR")
        if not dump_dir:
            return
        native = plan._require_native()
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(
            dump_dir,
            f"rank{self._context.planner_group_rank}_epoch{native.epoch}.pt",
        )
        if os.path.exists(path):
            return
        names = (
            "dst",
            "experts_to_copy",
            "zero_fill_ranges",
            "remote_stats",
            "dup_groups",
            "dup_loffs",
            "dup_counts",
            "status",
            "reduce_grad_status",
            "workspace",
        )
        self._torch.save(
            {
                "epoch": int(native.epoch),
                "tensors": {
                    name: getattr(native, name).detach().cpu()
                    for name in names
                },
            },
            path,
        )

    def _require_finite(self, stage, plan=None, **tensors):
        if os.environ.get("TILEXR_MINDSPEED_FINITE_CHECK", "0") != "1":
            return
        self._finite_check_sequence += 1
        epoch = "none" if plan is None else str(plan._require_native().epoch)
        for name, tensor in tensors.items():
            if tensor is None:
                continue
            checksum = tensor.sum(dtype=self._torch.float32)
            if bool(self._torch.isfinite(checksum).item()):
                continue
            print(
                "TILEXR_MINDSPEED_FIRST_NONFINITE "
                f"rank={self._context.planner_group_rank} "
                f"sequence={self._finite_check_sequence} stage={stage} "
                f"epoch={epoch} tensor={name} checksum={float(checksum.item())} "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype}",
                flush=True,
            )
            raise RuntimeError(
                f"TileXR MindSpeed detected a non-finite checksum at {stage}.{name}"
            )

    def _reserve_route_capture(self):
        capture_dir = os.environ.get("TILEXR_MINDSPEED_ROUTE_CAPTURE_DIR")
        if not capture_dir:
            return None
        try:
            capture_skip = int(
                os.environ.get("TILEXR_MINDSPEED_ROUTE_CAPTURE_SKIP_CALLS", "0")
            )
            capture_calls = int(
                os.environ.get("TILEXR_MINDSPEED_ROUTE_CAPTURE_CALLS", "10")
            )
        except ValueError as exc:
            raise ValueError("route capture skip/call counts must be integers") from exc
        if capture_skip < 0 or capture_calls <= 0:
            raise ValueError(
                "route capture skip must be non-negative and calls must be positive"
            )
        source_call = self._route_capture_seen
        self._route_capture_seen += 1
        if source_call < capture_skip or self._route_capture_count >= capture_calls:
            return None
        call = self._route_capture_count
        self._route_capture_count += 1
        return capture_dir, call, source_call

    def _write_route_capture(
        self,
        capture,
        plan,
        topk_experts,
        tokens_per_expert,
        hidden_input,
    ):
        if capture is None:
            return
        capture_dir, call, source_call = capture
        rank = int(self._context.planner_group_rank)
        topk_cpu = topk_experts.detach().to("cpu").contiguous()
        tpe_cpu = tokens_per_expert.detach().to("cpu").contiguous()
        topk_raw = topk_cpu.numpy().astype("<i4", copy=False).tobytes()
        native_plan = plan._require_native() if plan is not None else None
        capture_id = os.environ.get(
            "TILEXR_MINDSPEED_ROUTE_CAPTURE_ID", "unknown"
        )
        seed_material = f"{capture_id}:{rank}:{call}:payload-v1".encode("utf-8")
        payload = {
            "schema_version": 1,
            "complete": native_plan is not None,
            "rank": rank,
            "call": call,
            "source_call": source_call,
            "topk_shape": [int(value) for value in topk_cpu.shape],
            "topk_dtype": str(topk_cpu.dtype),
            "topk_encoding": "int32-le-zlib-base64",
            "topk_zlib_base64": base64.b64encode(
                zlib.compress(topk_raw)
            ).decode("ascii"),
            "topk_sha256": hashlib.sha256(topk_raw).hexdigest(),
            "tokens_per_expert": [int(value) for value in tpe_cpu.tolist()],
            "capture_id": capture_id,
            "payload_seed": int.from_bytes(
                hashlib.sha256(seed_material).digest()[:8], "little"
            ),
            "tensor_descriptors": {
                "hidden": _tensor_descriptor(hidden_input),
                "topk": _tensor_descriptor(topk_experts),
                "tokens_per_expert": _tensor_descriptor(tokens_per_expert),
            },
        }
        if native_plan is not None:
            payload["remote_stats"] = [
                int(value)
                for value in native_plan.remote_stats.detach().cpu().tolist()
            ]
            payload["experts_to_copy"] = [
                [int(value) for value in row]
                for row in native_plan.experts_to_copy.detach().cpu().tolist()
            ]
        os.makedirs(capture_dir, exist_ok=True)
        target = os.path.join(capture_dir, f"rank{rank}_call{call:02d}.json")
        temporary = f"{target}.tmp.{os.getpid()}"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)

    def dispatch(self, *args, hidden_buffer=None, **kwargs):
        self._optional_stage_barrier()
        hidden_input = args[0] if args else kwargs.get("hidden_sh")
        self._require_finite("dispatch.input", kwargs.get("plan"), hidden=hidden_input)
        async_finish = bool(kwargs.pop("async_finish", False))
        zero_copy = bool(kwargs.pop("zero_copy", False))
        input_plan = kwargs.get("plan")
        if input_plan is None and len(args) > 4:
            input_plan = args[4]
        topk_experts = kwargs.get("topk_experts_sk")
        tokens_per_expert = kwargs.get("tokens_per_expert")
        if topk_experts is None and len(args) > 2:
            topk_experts = args[2]
        if tokens_per_expert is None and len(args) > 3:
            tokens_per_expert = args[3]
        capture = (
            self._reserve_route_capture()
            if input_plan is None
            and topk_experts is not None
            and tokens_per_expert is not None
            else None
        )
        self._write_route_capture(
            capture, None, topk_experts, tokens_per_expert, hidden_input
        )
        result = super().dispatch(
            *args,
            async_finish=False,
            zero_copy=False,
            **kwargs,
        )
        hidden, route_weights, cu_seqlens, plan = result
        if (
            input_plan is None
            and os.environ.get("TILEXR_MINDSPEED_ROUTE_CAPTURE_DIR")
            and (topk_experts is None or tokens_per_expert is None)
        ):
            retained_topk, retained_tokens_per_expert = _retained_route_inputs(
                self._native_buffer, plan
            )
            if topk_experts is None:
                topk_experts = retained_topk
            if tokens_per_expert is None:
                tokens_per_expert = retained_tokens_per_expert
        if (
            input_plan is None
            and capture is None
            and topk_experts is not None
            and tokens_per_expert is not None
        ):
            capture = self._reserve_route_capture()
        self._write_route_capture(
            capture, plan, topk_experts, tokens_per_expert, hidden_input
        )
        self._require_finite(
            "dispatch.output", plan, hidden=hidden, route_weights=route_weights
        )
        if os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1":
            native_plan = plan._require_native()
            print(
                f"[TileXR MindSpeed rank "
                f"{self._context.planner_group_rank}] "
                f"dispatch_sync_begin epoch={native_plan.epoch}",
                flush=True,
            )
            try:
                self._native_buffer.synchronize()
            except BaseException as exc:
                print(
                    f"[TileXR MindSpeed rank "
                    f"{self._context.planner_group_rank}] "
                    f"dispatch_device_diag epoch={native_plan.epoch} "
                    f"exception={type(exc).__name__} message={exc!s}",
                    flush=True,
                )
                raise
            print(
                f"[TileXR MindSpeed rank "
                f"{self._context.planner_group_rank}] "
                f"dispatch_sync_end epoch={native_plan.epoch}",
                flush=True,
            )
        self._dump_native_plan_once(plan)
        route_boundary = self._validate_boundary(hidden_buffer)
        if hidden_buffer is not None:
            hidden_buffer.copy_(hidden)
            hidden = hidden_buffer
            if route_weights is not None:
                route_boundary.copy_(route_weights)
                route_weights = route_boundary
        if zero_copy:
            hidden, route_weights = self._bind_zero_copy_views(
                plan, hidden, route_weights
            )
            self._zero_copy_aliases = (
                plan._require_native(),
                hidden,
                route_weights,
            )
        event = self._record_event() if async_finish else None
        public_result = (hidden, route_weights, cu_seqlens, plan)
        return (*public_result, event) if async_finish else public_result

    def _stage_route_weights(self, route_weights, *, hidden_buffer):
        route_boundary = self._validate_boundary(hidden_buffer)
        if route_weights is None:
            return None
        flattened = route_weights.reshape(-1)
        if int(flattened.numel()) != int(route_boundary.numel()):
            raise RuntimeError("route weight staging does not match the TileXR boundary")
        route_boundary.copy_(flattened)
        return route_boundary

    def combine(self, *args, hidden_buffer=None, **kwargs):
        self._optional_stage_barrier()
        plan = kwargs.get("plan") if kwargs.get("plan") is not None else (
            args[0] if args else None
        )
        hidden_input = kwargs.get("hidden_nvsh")
        route_input = kwargs.get("route_weights_nvs")
        if hidden_input is None and len(args) > 1:
            hidden_input = args[1]
        if route_input is None and len(args) > 2:
            route_input = args[2]
        self._require_finite(
            "combine.input", plan, hidden=hidden_input, route_weights=route_input
        )
        zero_copy = bool(kwargs.pop("zero_copy", False))
        if hidden_buffer is not None:
            self._validate_boundary(hidden_buffer)
        if os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1":
            plan = kwargs.get("plan")
            if plan is None and args:
                plan = args[0]
            native_plan = plan._require_native()
            limit = int(plan.R) * int(plan.NvS)
            dst_min = int(plan.dst.min().item())
            dst_max = int(plan.dst.max().item())
            invalid = int(
                ((plan.dst >= limit) | (plan.dst < -limit)).sum().item()
            )
            dup_counts = plan.dup_counts.cpu().tolist()
            print(
                f"[TileXR MindSpeed rank {self._context.planner_group_rank}] "
                f"combine_plan epoch={native_plan.epoch} dst_min={dst_min} "
                f"dst_max={dst_max} limit={limit} invalid={invalid} "
                f"dup_counts={dup_counts}",
                flush=True,
            )
            if invalid:
                raise RuntimeError(
                    f"TileXR Combine plan has {invalid} out-of-range routes"
                )
        try:
            result = super().combine(*args, zero_copy=False, **kwargs)
            self._require_finite(
                "combine.output",
                plan,
                hidden=result[0],
                route_weights=result[1],
            )
            if os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1":
                try:
                    self._native_buffer.synchronize()
                except RuntimeError:
                    print(
                        f"[TileXR MindSpeed rank "
                        f"{self._context.planner_group_rank}] "
                        f"combine_device_diag epoch={native_plan.epoch} "
                        f"status={int(native_plan.status.item())} "
                        f"encoded_and_limit={plan.dup_counts.cpu().tolist()}",
                        flush=True,
                    )
                    raise
            return result
        finally:
            if zero_copy:
                self._zero_copy_aliases = None

    def _ensure_packed_projections(self, local_fc1, local_fc2):
        signature = (
            tuple(local_fc1.shape),
            local_fc1.dtype,
            tuple(local_fc2.shape),
            local_fc2.dtype,
        )
        if self._packed_projections is None:
            local = self._context.experts_per_rank
            dummy = self._torch.zeros(
                (local, 32),
                dtype=self._torch.bfloat16,
                device=local_fc1.device,
            )
            projections = ProjectionBuffers.from_local_weights(
                self._context,
                local_fc1,
                dummy,
                local_fc2,
                registration_backing_factory=lambda required_bytes, dtype: (
                    self._context.promote_projection_arena(
                        self._torch, dtype, required_bytes
                    )
                ),
                torch_module=self._torch,
            )
            self._native_buffer.register_projection_buffers(projections)
            self._packed_projections = projections
            self._packed_projection_signature = signature
        elif signature != self._packed_projection_signature:
            raise RuntimeError("packed projection shape changed after TileXR registration")
        local = self._context.experts_per_rank
        self._packed_projections.gate.narrow(0, 0, local).copy_(local_fc1)
        self._packed_projections.up.narrow(0, 0, local).zero_()
        self._packed_projections.down.narrow(0, 0, local).copy_(local_fc2)
        return self._packed_projections

    def _prefetch_weight_packed_batch2(
        self,
        plan,
        *,
        full_weights,
        local_experts,
        local_slots,
        source_vas,
        async_finish=False,
    ):
        del local_slots, source_vas
        full_fc1, full_fc2 = full_weights
        local_fc1, local_fc2 = local_experts
        self._require_finite(
            "prefetch.input", plan, local_fc1=local_fc1, local_fc2=local_fc2
        )
        projections = self._ensure_packed_projections(local_fc1, local_fc2)
        native_plan = plan._require_native()
        trace = os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1"
        if trace:
            print(
                f"[TileXR MindSpeed rank {self._context.planner_group_rank}] "
                f"packed_prefetch_begin epoch={native_plan.epoch}",
                flush=True,
            )
        try:
            self._native_buffer.prefetch_weight(
                native_plan, projections, async_finish=False
            )
        except BaseException as exc:
            if trace:
                print(
                    f"[TileXR MindSpeed rank "
                    f"{self._context.planner_group_rank}] "
                    f"packed_prefetch_exception epoch={native_plan.epoch} "
                    f"exception={type(exc).__name__} message={exc!s}",
                    flush=True,
                )
            raise
        if trace:
            print(
                f"[TileXR MindSpeed rank {self._context.planner_group_rank}] "
                f"packed_prefetch_end epoch={native_plan.epoch}",
                flush=True,
            )
        local = self._context.experts_per_rank
        rank = self._context.planner_group_rank
        active_remote = 0
        experts = plan.experts_to_copy[rank]
        for slot in range(self.B):
            expert = int(experts[slot].item())
            if expert < 0:
                continue
            full_fc1[self.E + slot].copy_(projections.gate[local + slot])
            full_fc2[self.E + slot].copy_(projections.down[local + slot])
            if expert // local != rank:
                active_remote += 1
        self._require_finite(
            "prefetch.output", plan, full_fc1=full_fc1, full_fc2=full_fc2
        )
        self._tilexr_remote_prefetches += active_remote
        if os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1":
            print(
                f"[TileXR MindSpeed rank {rank}] packed_prefetch_remote="
                f"{active_remote} total={self._tilexr_remote_prefetches}",
                flush=True,
            )
        return self._record_event() if async_finish else None

    def _optional_stage_barrier(self):
        optional_stage_barrier(self._torch)

    def _ensure_reduce_dummy(self, full_fc1, reduce_fc1):
        dummy_width = 16
        full_shape = (self.E + self.B, dummy_width)
        reduce_shape = (self.R, self.B, dummy_width)
        if self._reduce_dummy is None:
            self._reduce_dummy = self._torch.zeros(
                full_shape, dtype=self._torch.float32, device=full_fc1.device
            )
            reduce_elements = self.R * self.B * dummy_width
            allocation_elements = max(
                reduce_elements,
                _UDMA_COMPAT_REGISTRATION_BYTES // 4,
            )
            self._reduce_dummy_buffer_allocation = self._torch.zeros(
                (allocation_elements,),
                dtype=self._torch.float32,
                device=reduce_fc1.device,
            )
            self._reduce_dummy_buffer = self._reduce_dummy_buffer_allocation.narrow(
                0, 0, reduce_elements
            ).reshape(reduce_shape)
            self._reduce_dummy_buffer._tilexr_registration_backing = (
                self._reduce_dummy_buffer_allocation
            )
        else:
            self._reduce_dummy.zero_()
            self._reduce_dummy_buffer.zero_()
        return self._reduce_dummy, self._reduce_dummy_buffer

    def _reduce_grad_packed_batch2(
        self,
        plan,
        *,
        full_grads,
        reduce_buffers,
        local_grads,
        local_slots,
        local_staging,
        source_vas,
    ):
        del local_slots, local_staging, source_vas
        full_fc1, full_fc2 = full_grads
        reduce_fc1, reduce_fc2 = reduce_buffers
        local_fc1, local_fc2 = local_grads
        self._require_finite(
            "reduce_grad.input",
            plan,
            full_fc1=full_fc1,
            full_fc2=full_fc2,
            reduce_fc1=reduce_fc1,
            reduce_fc2=reduce_fc2,
        )
        dummy, dummy_reduce = self._ensure_reduce_dummy(full_fc1, reduce_fc1)
        trace = os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1"
        native_plan = plan._require_native()
        if trace:
            print(
                f"[TileXR MindSpeed rank {self._context.planner_group_rank}] "
                f"packed_reduce_grad_begin epoch={native_plan.epoch} "
                f"full_shapes={[tuple(value.shape) for value in (full_fc1, dummy, full_fc2)]} "
                f"reduce_shapes={[tuple(value.shape) for value in (reduce_fc1, dummy_reduce, reduce_fc2)]} "
                f"experts_to_copy={native_plan.experts_to_copy.cpu().tolist()}",
                flush=True,
            )
        try:
            super().reduce_grad(
                plan=plan,
                full_gate_grad=full_fc1,
                full_up_grad=dummy,
                full_down_grad=full_fc2,
                gate_reduce_buffer=reduce_fc1,
                up_reduce_buffer=dummy_reduce,
                down_reduce_buffer=reduce_fc2,
            )
        except BaseException as exc:
            if trace:
                info = self._native_buffer.reduce_grad_info
                print(
                    f"[TileXR MindSpeed rank {self._context.planner_group_rank}] "
                    f"packed_reduce_grad_exception epoch={native_plan.epoch} "
                    f"exception={type(exc).__name__} message={exc!s} "
                    f"status={int(native_plan.reduce_grad_status.item())} "
                    f"info={info}",
                    flush=True,
                )
            raise
        local = self._context.experts_per_rank
        begin = self._context.planner_group_rank * local
        local_fc1.copy_(full_fc1.narrow(0, begin, local))
        local_fc2.copy_(full_fc2.narrow(0, begin, local))
        self._require_finite(
            "reduce_grad.output",
            plan,
            full_fc1=full_fc1,
            full_fc2=full_fc2,
            local_fc1=local_fc1,
            local_fc2=local_fc2,
        )
        if os.environ.get("TILEXR_MINDSPEED_TRACE", "0") == "1":
            print(
                f"[TileXR MindSpeed rank {self._context.planner_group_rank}] "
                f"packed_reduce_grad=1 info={self._native_buffer.reduce_grad_info}",
                flush=True,
            )
        return local_fc1, local_fc2

    def destroy(self):
        self._packed_projections = None
        self._reduce_dummy = None
        self._reduce_dummy_buffer = None
        self._reduce_dummy_buffer_allocation = None
        self.token_buffers = ()
        self.route_buffers = ()
        self._route_by_token_ptr = {}
        super().destroy()


def create_tilexr_moonep_backend(**kwargs):
    _prewarm_framework_ops()
    kwargs["buffer_cls"] = MindSpeedTileXRBuffer
    return MoonEPBufferFlexBackend(**kwargs)


create_tilexr_moonep_backend.__mindspeed_capabilities__ = {
    MOONEP_NATIVE_NPU_CAPABILITY
}
create_tilexr_moonep_backend.__mindspeed_external_communication_owner__ = True
