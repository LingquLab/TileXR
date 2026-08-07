from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = ROOT / "integrations" / "moonep_torch"
for path in (ROOT, INTEGRATION, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fakes import FakeRuntime, FakeTensor, FakeTorch
from tilexr_moonep import TileXRMoonEPBuffer, TileXRMoonEPContext
from tilexr_moonep.runtime import NativeCapabilities
from tools.moonep.contracts import (
    BackendUnavailableError,
    ContractError,
    MoonEPBackend,
    MoonEPDimensions,
    ProjectionTensors,
)
import tools.moonep.tilexr_backend as tilexr_backend


def dimensions() -> MoonEPDimensions:
    return MoonEPDimensions(
        rank=0,
        world_size=2,
        tokens_per_rank=4,
        topk=2,
        expert_count=4,
        prefetch_slots=2,
        token_padding=4,
        hidden_size=4,
        intermediate_size=8,
    )


def tensor(shape, dtype, *, device="npu:0"):
    return FakeTensor(shape, dtype, device)


def make_context(torch, runtime, *, node_count=1, device_index=0):
    d = dimensions()
    return TileXRMoonEPContext(
        runtime=runtime,
        global_rank=d.rank,
        global_world_size=d.world_size,
        node_rank=0,
        node_count=node_count,
        local_rank=device_index,
        local_world_size=d.world_size,
        planner_group_rank=d.rank,
        planner_group_size=d.world_size,
        lane_group_rank=0,
        lane_group_size=1,
        device_index=device_index,
        tokens_per_rank=d.tokens_per_rank,
        hidden_size=d.hidden_size,
        topk=d.topk,
        expert_count=d.expert_count,
        dtype=torch.bfloat16,
        token_padding=d.token_padding,
        prefetch_slots=d.prefetch_slots,
    )


def make_backend(*, write_status_markers=True):
    torch = FakeTorch()
    runtime = FakeRuntime(write_status_markers=write_status_markers)
    context = make_context(torch, runtime)
    buffer = TileXRMoonEPBuffer(context, torch_module=torch)
    backend = tilexr_backend.TileXRMoonEPBackend(
        torch_module=torch,
        dimensions=dimensions(),
        buffer=buffer,
    )
    return torch, runtime, context, backend


def projection_tensors(torch, *, reduce=False):
    d = dimensions()
    prefix = (d.world_size, d.prefetch_slots) if reduce else (
        d.expert_count + d.prefetch_slots,
    )
    dtype = torch.float32 if reduce else torch.bfloat16
    return ProjectionTensors(
        gate=tensor(prefix + (d.hidden_size, d.intermediate_size), dtype),
        up=tensor(prefix + (d.hidden_size, d.intermediate_size), dtype),
        down=tensor(prefix + (d.intermediate_size, d.hidden_size), dtype),
    )


class TileXRCorrectnessAdapterTests(unittest.TestCase):
    def _planned(self, backend, torch):
        return backend.planning(
            tensor((4, 2), torch.int32), tensor((4,), torch.int32)
        ).plan

    def test_planning_and_dispatch_preserve_native_tensor_identity(self):
        torch, runtime, _, backend = make_backend()
        plan = self._planned(backend, torch)
        native = next(call[1] for call in runtime.calls if call[0] == "planning")
        self.assertIs(plan.dst, native.dst)
        self.assertIs(plan.experts_to_copy, native.experts_to_copy)
        self.assertIs(plan.zero_fill_ranges, native.zero_fill_ranges)
        self.assertIs(plan.remote_stats, native.remote_stats)
        self.assertIsNone(plan.dedup)
        backend.synchronize()

        hidden = tensor((4, 4), torch.bfloat16)
        weights = tensor((4, 2), torch.float32)
        first = backend.dispatch(plan, hidden, weights)
        call = [item for item in runtime.calls if item[0] == "dispatch"][-1]
        self.assertIs(call[1], native)
        self.assertIs(call[2], hidden)
        self.assertIs(call[3], first.hidden)
        self.assertIs(call[4], weights)
        self.assertIs(call[5], first.route_weights)
        self.assertEqual(call[-2:], (False, True))
        self.assertIs(first.plan, plan)
        self.assertIs(plan.dedup.groups, native.dup_groups)
        self.assertIs(plan.dedup.loffs, native.dup_loffs)
        self.assertIs(plan.dedup.counts, native.dup_counts)
        backend.synchronize()

        dedup = (plan.dedup.groups, plan.dedup.loffs, plan.dedup.counts)
        second = backend.dispatch(plan, hidden, weights)
        call = [item for item in runtime.calls if item[0] == "dispatch"][-1]
        self.assertEqual(call[-2:], (False, True))
        self.assertEqual(
            (plan.dedup.groups, plan.dedup.loffs, plan.dedup.counts), dedup
        )
        self.assertIs(second.plan, plan)
        backend.synchronize()
        backend.close()

    def test_all_downstream_stages_map_in_place_and_optional_weights(self):
        torch, runtime, _, backend = make_backend()
        plan = self._planned(backend, torch)
        backend.synchronize()
        dispatched = backend.dispatch(
            plan, tensor((4, 4), torch.bfloat16), route_weights_sk=None
        )
        self.assertIsNone(dispatched.route_weights)
        backend.synchronize()

        projections = projection_tensors(torch)
        prefetched = backend.prefetch_weight(plan, projections)
        self.assertIs(prefetched.projections, projections)
        native_prefetch = [
            call for call in runtime.calls if call[0] == "prefetch_weight"
        ][-1][2]
        self.assertIsNot(native_prefetch.gate, projections.gate)
        self.assertIsNot(native_prefetch.up, projections.up)
        self.assertIsNot(native_prefetch.down, projections.down)
        self.assertEqual(native_prefetch.gate.shape, (4, 4, 8))
        self.assertEqual(native_prefetch.down.shape, (4, 8, 4))
        self.assertIsNotNone(native_prefetch.backing)
        self.assertEqual(native_prefetch.udma_handle, 1)
        backend.synchronize()

        expert_output = tensor((dimensions().nvsh, 4), torch.bfloat16)
        combined = backend.combine(plan, expert_output, route_weights_nvs=None)
        self.assertIsNone(combined.route_weights)
        native_combine = [call for call in runtime.calls if call[0] == "combine"][-1]
        self.assertIs(native_combine[2], expert_output)
        self.assertIs(native_combine[3], combined.hidden)
        backend.synchronize()

        full_grads = ProjectionTensors(
            tensor((6, 4, 8), torch.float32),
            tensor((6, 4, 8), torch.float32),
            tensor((6, 8, 4), torch.float32),
        )
        reduce_buffers = projection_tensors(torch, reduce=True)
        reduced = backend.reduce_grad(plan, full_grads, reduce_buffers)
        self.assertIs(reduced.full_grads, full_grads)
        self.assertIs(reduced.reduce_buffers, reduce_buffers)
        native_reduce = [call for call in runtime.calls if call[0] == "reduce_grad"][-1][2]
        self.assertIs(native_reduce.gate, full_grads.gate)
        self.assertIs(native_reduce.gate_reduce, reduce_buffers.gate)
        self.assertIs(native_reduce.down_reduce, reduce_buffers.down)
        for _, full_tensor in full_grads.items():
            self.assertEqual(len(full_tensor.copy_calls), 1)
        for _, reduce_tensor in reduce_buffers.items():
            self.assertEqual(reduce_tensor.masked_fill_calls, [])
            self.assertEqual(len(reduce_tensor.zero_calls), 1)
        native_plan = backend._native_plans[id(plan)].native
        self.assertEqual(native_plan.reduce_grad_status.item(), 0)
        backend.synchronize()
        backend.close()
        lifecycle = [call[0] for call in runtime.calls]
        self.assertLess(lifecycle.index("udma_unregister"), lifecycle.index("close"))

    def test_unknown_or_cloned_plan_is_rejected(self):
        torch, runtime, _, backend = make_backend()
        plan = self._planned(backend, torch)
        backend.synchronize()
        native_call_count = len(runtime.calls)
        cloned = copy.copy(plan)
        with self.assertRaisesRegex(ContractError, "not owned"):
            backend.dispatch(cloned, tensor((4, 4), torch.bfloat16))
        self.assertEqual(len(runtime.calls), native_call_count)
        backend.close()

    def test_replaced_normalized_plan_tensor_is_rejected(self):
        torch, runtime, _, backend = make_backend()
        plan = self._planned(backend, torch)
        backend.synchronize()
        native_call_count = len(runtime.calls)
        plan.cu_seqlens = tensor((6,), torch.int32)
        with self.assertRaisesRegex(ContractError, "cu_seqlens.*native storage"):
            backend.dispatch(plan, tensor((4, 4), torch.bfloat16))
        self.assertEqual(len(runtime.calls), native_call_count)
        backend.close()

    def test_synchronize_propagates_native_stage_status_failure(self):
        torch, _, _, backend = make_backend()
        plan = self._planned(backend, torch)
        backend.synchronize()
        backend.dispatch(plan, tensor((4, 4), torch.bfloat16))
        native = backend._native_plans[id(plan)].native
        native.status._item = 3000
        with self.assertRaisesRegex(RuntimeError, "expected 0"):
            backend.synchronize()
        backend.close()

    def test_close_is_idempotent_and_releases_plan_registry(self):
        torch, runtime, _, backend = make_backend()
        self._planned(backend, torch)
        backend.close()
        backend.close()
        self.assertEqual([call[0] for call in runtime.calls].count("close"), 1)
        self.assertEqual(backend._native_plans, {})

    def test_backend_satisfies_normalized_protocol(self):
        _, _, _, backend = make_backend()
        self.assertIsInstance(backend, MoonEPBackend)
        self.assertEqual(backend.name, "tilexr_native")
        self.assertFalse(backend.supports_duplicate_destinations)
        backend.close()


class TileXRCorrectnessFactoryTests(unittest.TestCase):
    def setUp(self):
        self.torch = FakeTorch()
        self.runtime = FakeRuntime()
        self.context = make_context(self.torch, self.runtime)
        self.args = SimpleNamespace(install_prefix="/tmp/tilexr", wait_iterations=77)

    def _create(self):
        with patch.object(
            tilexr_backend.TileXRMoonEPContext,
            "from_env",
            return_value=self.context,
        ) as factory:
            backend = tilexr_backend.create_backend(
                torch_module=self.torch,
                dimensions=dimensions(),
                case=SimpleNamespace(case_id="adapter"),
                args=self.args,
            )
        return backend, factory

    def test_factory_constructs_context_with_exact_dimensions(self):
        backend, factory = self._create()
        factory.assert_called_once_with(
            tokens_per_rank=4,
            hidden_size=4,
            topk=2,
            expert_count=4,
            dtype=self.torch.bfloat16,
            token_padding=4,
            prefetch_slots=2,
            install_prefix="/tmp/tilexr",
            torch_module=self.torch,
        )
        self.assertEqual(backend.buffer.wait_iterations, 77)
        backend.close()

    def test_factory_rejects_missing_native_stage_and_closes_context(self):
        self.runtime.capabilities = NativeCapabilities(2, 31, 1)
        with self.assertRaisesRegex(BackendUnavailableError, "planning.*stub"):
            self._create()
        self.assertTrue(self.runtime.closed)

    def test_factory_rejects_cross_node_and_dimension_mismatch(self):
        self.context.node_count = 2
        with self.assertRaisesRegex(BackendUnavailableError, "same-node"):
            self._create()
        self.assertTrue(self.runtime.closed)

        self.runtime = FakeRuntime()
        self.context = make_context(self.torch, self.runtime)
        self.context.tokens_per_rank += 1
        with self.assertRaisesRegex(BackendUnavailableError, "tokens_per_rank"):
            self._create()
        self.assertTrue(self.runtime.closed)

    def test_factory_closes_context_when_buffer_construction_fails(self):
        with patch.object(
            tilexr_backend.TileXRMoonEPContext,
            "from_env",
            return_value=self.context,
        ), patch.object(
            tilexr_backend,
            "TileXRMoonEPBuffer",
            side_effect=RuntimeError("buffer failed"),
        ):
            with self.assertRaisesRegex(BackendUnavailableError, "buffer failed"):
                tilexr_backend.create_backend(
                    torch_module=self.torch,
                    dimensions=dimensions(),
                    case=SimpleNamespace(case_id="adapter"),
                    args=self.args,
                )
        self.assertTrue(self.runtime.closed)


if __name__ == "__main__":
    unittest.main()
