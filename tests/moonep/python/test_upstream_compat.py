from __future__ import annotations

import inspect
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = ROOT / "integrations" / "moonep_torch"
for path in (ROOT, INTEGRATION, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fakes import FakeRuntime, FakeTensor, FakeTorch
import tilexr_moonep as package
import tilexr_moonep.compat as compat
from tilexr_moonep.compat import Buffer, MoonEPCommPlan
from tilexr_moonep.torch_api import MoonEPPlan, TileXRMoonEPContext
from tools.moonep.expert_forward import run_expert_forward


def tensor(shape, dtype="int32"):
    return FakeTensor(shape, dtype, "npu:0")


def context(*, prefetch_slots=2, token_padding=128, node_count=1):
    torch = FakeTorch()
    runtime = FakeRuntime(rank=0, world_size=2)
    value = TileXRMoonEPContext(
        runtime=runtime,
        global_rank=0,
        global_world_size=2,
        node_rank=0,
        node_count=node_count,
        local_rank=0,
        local_world_size=2,
        planner_group_rank=0,
        planner_group_size=2,
        lane_group_rank=0,
        lane_group_size=1,
        device_index=0,
        tokens_per_rank=4,
        hidden_size=8,
        topk=2,
        expert_count=8,
        dtype=torch.bfloat16,
        token_padding=token_padding,
        prefetch_slots=prefetch_slots,
    )
    return torch, runtime, value


def native_plan(runtime):
    return MoonEPPlan(
        dst=tensor((8,)),
        experts_to_copy=tensor((2, 2)),
        zero_fill_ranges=tensor((10, 2)),
        remote_stats=tensor((2,)),
        dup_groups=tensor((1024, 3)),
        dup_loffs=tensor((1024,)),
        dup_counts=tensor((2,)),
        status=tensor((1,)),
        reduce_grad_status=tensor((1,)),
        workspace=tensor((64,), "uint8"),
        n=8,
        tokens_per_rank=4,
        topk=2,
        expert_count=8,
        rank_size=2,
        prefetch_slots=2,
        nv_s=1024,
        token_padding=128,
        epoch=3,
        backend="native",
        runtime=runtime,
    )


def compat_buffer(*, node_count=1):
    torch, runtime, native_context = context(
        prefetch_slots=2, node_count=node_count
    )
    runtime.write_status_markers = True
    with patch.object(
        TileXRMoonEPContext, "from_env", return_value=native_context
    ), patch.object(compat, "_torch", return_value=torch):
        buffer = Buffer(4, 8, 2, 8, 2, B=2, token_padding=128)
    return torch, runtime, buffer


def planned_buffer():
    torch, runtime, buffer = compat_buffer()
    _, _, _, plan = buffer.dispatch(
        tensor((4, 8), torch.bfloat16),
        tensor((4, 2), torch.float32),
        tensor((4, 2), torch.int32),
        tensor((8,), torch.int32),
    )
    return torch, runtime, buffer, plan


class PlanningCompatibilityTests(unittest.TestCase):
    def test_context_accepts_compact_prefetch_slots(self):
        _, _, value = context(prefetch_slots=2)
        self.assertEqual(value.experts_per_rank, 4)
        self.assertEqual(value.prefetch_slots, 2)
        self.assertEqual(value.nv_s, 8 + (128 - 1) * 2 * 4)

    def test_context_rejects_prefetch_slots_outside_upstream_range(self):
        with self.assertRaisesRegex(ValueError, r"prefetch_slots must be in \[1, E/R\]"):
            context(prefetch_slots=0)
        with self.assertRaisesRegex(ValueError, r"prefetch_slots must be in \[1, E/R\]"):
            context(prefetch_slots=5)

    def test_plan_exposes_upstream_fields_and_clone_owns_private_state(self):
        runtime = FakeRuntime(rank=0, world_size=2)
        native = native_plan(runtime)
        plan = MoonEPCommPlan._from_native(native)

        self.assertEqual(
            (plan.N, plan.R, plan.E, plan.B, plan.NvS, plan.K),
            (8, 2, 8, 2, 1024, 2),
        )
        self.assertIs(plan.dst, native.dst)
        self.assertIs(plan.dup_groups, native.dup_groups)
        with self.assertRaises(FrozenInstanceError):
            plan.N = 9

        cloned = plan.clone()
        self.assertIsNot(cloned, plan)
        self.assertEqual(
            (cloned.N, cloned.R, cloned.E, cloned.B, cloned.NvS, cloned.K),
            (plan.N, plan.R, plan.E, plan.B, plan.NvS, plan.K),
        )
        for name in (
            "dst",
            "experts_to_copy",
            "zero_fill_ranges",
            "remote_stats",
            "dup_groups",
            "dup_loffs",
            "dup_counts",
        ):
            self.assertIsNot(getattr(cloned, name), getattr(plan, name))
        self.assertIs(cloned._native_plan.runtime, runtime)
        self.assertIsNot(cloned._native_plan.status, native.status)
        self.assertIsNot(cloned._native_plan.reduce_grad_status, native.reduce_grad_status)
        self.assertIsNot(cloned._native_plan.workspace, native.workspace)

    def test_buffer_constructor_matches_upstream_and_maps_dimensions(self):
        expected = (
            "S",
            "H",
            "K",
            "E",
            "num_ep_ranks",
            "num_sms",
            "token_padding",
            "B",
            "group",
            "comm_stream_priority",
            "enable_pdl",
            "explicitly_destroy",
        )
        parameters = tuple(inspect.signature(Buffer.__init__).parameters)[1:]
        self.assertEqual(parameters, expected)

        torch, _, native_context = context(prefetch_slots=2)
        with patch.object(
            TileXRMoonEPContext, "from_env", return_value=native_context
        ) as factory, patch.object(compat, "_torch", return_value=torch):
            buffer = Buffer(4, 8, 2, 8, 2, B=2, token_padding=128)
        factory.assert_called_once_with(
            tokens_per_rank=4,
            hidden_size=8,
            topk=2,
            expert_count=8,
            dtype=torch.bfloat16,
            token_padding=128,
            prefetch_slots=2,
            torch_module=torch,
        )
        self.assertEqual(buffer.S, 4)
        self.assertEqual(buffer.B, 2)
        buffer.destroy()
        self.assertTrue(buffer.destroyed)

    def test_buffer_validates_process_group_against_tilexr_context(self):
        torch, _, native_context = context(prefetch_slots=2)

        class Distributed:
            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_rank(*, group):
                return 1

            @staticmethod
            def get_world_size(*, group):
                return 2

        torch.distributed = Distributed()
        with patch.object(
            TileXRMoonEPContext, "from_env", return_value=native_context
        ), patch.object(compat, "_torch", return_value=torch):
            with self.assertRaisesRegex(AssertionError, "does not match"):
                Buffer(4, 8, 2, 8, 2, B=2, group=object())

    def test_buffer_rejects_group_world_size_mismatch_before_native_init(self):
        torch = FakeTorch()

        class Distributed:
            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def get_rank(*, group):
                return 0

            @staticmethod
            def get_world_size(*, group):
                return 4

        torch.distributed = Distributed()
        with patch.object(compat, "_torch", return_value=torch), patch.object(
            TileXRMoonEPContext, "from_env"
        ) as factory:
            with self.assertRaisesRegex(AssertionError, "group world size"):
                Buffer(4, 8, 2, 8, 2, B=2, group=object())
        factory.assert_not_called()


class DispatchCompatibilityTests(unittest.TestCase):
    def test_dispatch_signature_matches_upstream(self):
        expected = (
            "self",
            "hidden_sh",
            "route_weights_sk",
            "topk_experts_sk",
            "tokens_per_expert",
            "plan",
            "async_finish",
            "inter_rank_sync",
            "zero_copy",
        )
        self.assertEqual(tuple(inspect.signature(Buffer.dispatch).parameters), expected)

    def test_fresh_saved_and_optional_weight_dispatch_contract(self):
        torch, runtime, buffer = compat_buffer()
        zero_allocations = []
        original_zeros = torch.zeros

        def tracked_zeros(shape, *, dtype, device):
            zero_allocations.append((tuple(shape), dtype, device))
            return original_zeros(shape, dtype=dtype, device=device)

        torch.zeros = tracked_zeros
        hidden = tensor((4, 8), torch.bfloat16)
        weights = tensor((4, 2), torch.float32)
        topk = tensor((4, 2), torch.int32)
        tpe = tensor((8,), torch.int32)

        dispatched, dispatched_weights, cu_seqlens, plan = buffer.dispatch(
            hidden, weights, topk, tpe
        )
        self.assertEqual(dispatched.shape, (1024, 8))
        self.assertEqual(dispatched_weights.shape, (1024,))
        self.assertIn(((1024,), torch.float32, "npu:0"), zero_allocations)
        self.assertEqual(cu_seqlens.shape, (10,))
        self.assertIsInstance(plan, MoonEPCommPlan)
        native_call = [call for call in runtime.calls if call[0] == "dispatch"][-1]
        self.assertIs(native_call[1], plan._native_plan)

        reused, no_weights, reused_cu, echoed = buffer.dispatch(hidden, plan=plan)
        self.assertEqual(reused.shape, (1024, 8))
        self.assertIsNone(no_weights)
        self.assertIsNone(reused_cu)
        self.assertIs(echoed, plan)

        cloned = plan.clone()
        _, _, _, echoed_clone = buffer.dispatch(hidden, plan=cloned)
        self.assertIs(echoed_clone, cloned)
        buffer.destroy()

    def test_async_and_zero_copy_dispatch_surface(self):
        torch, _, buffer = compat_buffer()
        hidden = tensor((4, 8), torch.bfloat16)
        weights = tensor((4, 2), torch.float32)
        topk = tensor((4, 2), torch.int32)
        tpe = tensor((8,), torch.int32)

        hidden_nvsh, weights_nvs, _, plan, event = buffer.dispatch(
            hidden,
            weights,
            topk,
            tpe,
            async_finish=True,
            zero_copy=True,
        )
        self.assertIsNotNone(event)
        self.assertEqual(
            buffer._zero_copy_aliases,
            (plan._native_plan, hidden_nvsh, weights_nvs),
        )

        buffer.dispatch(hidden, plan=plan, zero_copy=False)
        self.assertIsNone(buffer._zero_copy_aliases)
        buffer.destroy()

    def test_dispatch_rejects_foreign_public_plan(self):
        torch, _, buffer = compat_buffer()
        hidden = tensor((4, 8), torch.bfloat16)
        public_only = MoonEPCommPlan(
            dst=tensor((8,)),
            experts_to_copy=tensor((2, 2)),
            zero_fill_ranges=tensor((10, 2)),
            remote_stats=tensor((2,)),
            N=8,
            R=2,
            E=8,
            B=2,
            NvS=1024,
            K=2,
            dup_groups=tensor((1024, 3)),
            dup_loffs=tensor((1024,)),
            dup_counts=tensor((2,)),
        )
        with self.assertRaisesRegex(ValueError, "not created by this TileXR Buffer"):
            buffer.dispatch(hidden, plan=public_only)
        buffer.destroy()


class PrefetchWeightCompatibilityTests(unittest.TestCase):
    def test_prefetch_weight_signature_matches_upstream(self):
        expected = (
            "self",
            "plan",
            "async_finish",
            "full_gate_weight",
            "full_up_weight",
            "full_down_weight",
        )
        self.assertEqual(
            tuple(inspect.signature(Buffer.prefetch_weight).parameters), expected
        )

    def test_prefetch_weight_gathers_from_each_callers_full_tensor(self):
        torch, runtime, buffer, plan = planned_buffer()
        full = [tensor((10, 8, 8), torch.bfloat16) for _ in range(3)]

        result = buffer.prefetch_weight(
            plan=plan,
            full_gate_weight=full[0],
            full_up_weight=full[1],
            full_down_weight=full[2],
        )
        self.assertIsNone(result)
        for value in full:
            self.assertTrue(
                any(call[2] == (2, 8, 8) for call in value.copy_calls),
                "the [E:E+B] slots were not copied back",
            )
        self.assertNotIn("prefetch_weight", [call[0] for call in runtime.calls])
        self.assertNotIn("udma_register", [call[0] for call in runtime.calls])

        event = buffer.prefetch_weight(
            plan=plan,
            async_finish=True,
            full_gate_weight=full[0],
            full_up_weight=full[1],
            full_down_weight=full[2],
        )
        self.assertIsNotNone(event)
        self.assertNotIn("prefetch_weight", [call[0] for call in runtime.calls])
        buffer.destroy()

    def test_prefetch_weight_requires_plan_and_upstream_tensor_layout(self):
        torch, runtime, buffer, plan = planned_buffer()
        good = tensor((10, 8, 8), torch.bfloat16)
        call_count = len(runtime.calls)
        with self.assertRaisesRegex(AssertionError, "plan is required"):
            buffer.prefetch_weight(
                full_gate_weight=good,
                full_up_weight=good,
                full_down_weight=good,
            )
        with self.assertRaisesRegex(AssertionError, r"\[E\+B, H, H'\]"):
            buffer.prefetch_weight(
                plan=plan,
                full_gate_weight=tensor((10, 64), torch.bfloat16),
                full_up_weight=good,
                full_down_weight=good,
            )
        self.assertEqual(len(runtime.calls), call_count)
        buffer.destroy()


class ExpertForwardCompatibilityBoundaryTests(unittest.TestCase):
    def test_expert_forward_remains_caller_owned(self):
        self.assertFalse(hasattr(Buffer, "expert_forward"))
        self.assertNotIn("run_expert_forward", compat.__dict__)
        self.assertTrue(callable(run_expert_forward))


class CombineCompatibilityTests(unittest.TestCase):
    def test_combine_signature_matches_upstream(self):
        expected = (
            "self",
            "plan",
            "hidden_nvsh",
            "route_weights_nvs",
            "async_finish",
            "inter_rank_sync",
            "zero_copy",
        )
        self.assertEqual(tuple(inspect.signature(Buffer.combine).parameters), expected)

    def test_combine_sync_optional_weights_and_async_surface(self):
        torch, runtime, buffer = compat_buffer()
        hidden_nvsh, weights_nvs, _, plan = buffer.dispatch(
            tensor((4, 8), torch.bfloat16),
            tensor((4, 2), torch.float32),
            tensor((4, 2), torch.int32),
            tensor((8,), torch.int32),
        )

        hidden, no_weights, event = buffer.combine(
            plan=plan, hidden_nvsh=hidden_nvsh
        )
        self.assertEqual(hidden.shape, (4, 8))
        self.assertIsNone(no_weights)
        self.assertIsNone(event)

        hidden, weights, event = buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_nvsh,
            route_weights_nvs=weights_nvs,
            async_finish=True,
        )
        self.assertEqual(hidden.shape, (4, 8))
        self.assertEqual(weights.shape, (4, 2))
        self.assertIsNotNone(event)
        self.assertEqual([call[0] for call in runtime.calls].count("combine"), 2)
        buffer.destroy()

    def test_combine_zero_copy_requires_latest_dispatch_alias(self):
        torch, _, buffer = compat_buffer()
        hidden_nvsh, weights_nvs, _, plan = buffer.dispatch(
            tensor((4, 8), torch.bfloat16),
            tensor((4, 2), torch.float32),
            tensor((4, 2), torch.int32),
            tensor((8,), torch.int32),
            zero_copy=True,
        )
        with self.assertRaisesRegex(AssertionError, "alias"):
            buffer.combine(
                plan=plan,
                hidden_nvsh=tensor((1024, 8), torch.bfloat16),
                zero_copy=True,
            )

        hidden, weights, event = buffer.combine(
            plan=plan,
            hidden_nvsh=hidden_nvsh,
            route_weights_nvs=weights_nvs,
            zero_copy=True,
        )
        self.assertEqual(hidden.shape, (4, 8))
        self.assertEqual(weights.shape, (4, 2))
        self.assertIsNone(event)
        self.assertIsNone(buffer._zero_copy_aliases)
        buffer.destroy()

    def test_cross_node_combine_uses_v2_internal_synchronization(self):
        torch, runtime, buffer = compat_buffer(node_count=2)
        hidden_nvsh, _, _, plan = buffer.dispatch(
            tensor((4, 8), torch.bfloat16),
            topk_experts_sk=tensor((4, 2), torch.int32),
            tokens_per_expert=tensor((8,), torch.int32),
        )
        phases = []
        buffer._host_phase_barrier = lambda phase: phases.append(phase)

        buffer.combine(plan=plan, hidden_nvsh=hidden_nvsh)

        self.assertEqual(phases, [])
        combine_calls = [call for call in runtime.calls if call[0] == "combine"]
        self.assertEqual([call[-1] for call in combine_calls], [0])
        buffer.destroy()

    def test_host_phase_barrier_uses_upstream_process_group(self):
        _, _, buffer = compat_buffer()
        calls = []

        class Distributed:
            @staticmethod
            def is_initialized():
                return True

            @staticmethod
            def barrier(*, group):
                calls.append(group)

        group = object()
        buffer.group = group
        buffer._torch.distributed = Distributed()
        buffer._host_phase_barrier("published")
        self.assertEqual(calls, [group])
        buffer.destroy()


class ReduceGradAndPublicApiTests(unittest.TestCase):
    def test_reduce_grad_signature_matches_upstream(self):
        expected = (
            "self",
            "plan",
            "async_finish",
            "full_gate_grad",
            "full_up_grad",
            "full_down_grad",
            "gate_reduce_buffer",
            "up_reduce_buffer",
            "down_reduce_buffer",
        )
        self.assertEqual(
            tuple(inspect.signature(Buffer.reduce_grad).parameters), expected
        )

    def test_reduce_grad_maps_in_place_and_returns_optional_event(self):
        torch, runtime, buffer, plan = planned_buffer()
        full = [tensor((10, 8, 8), torch.float32) for _ in range(3)]
        reduced = [tensor((2, 2, 8, 8), torch.float32) for _ in range(3)]

        result = buffer.reduce_grad(
            plan=plan,
            full_gate_grad=full[0],
            full_up_grad=full[1],
            full_down_grad=full[2],
            gate_reduce_buffer=reduced[0],
            up_reduce_buffer=reduced[1],
            down_reduce_buffer=reduced[2],
        )
        self.assertIsNone(result)
        for value in full:
            self.assertGreaterEqual(
                sum(call[2] == (2, 8, 8) for call in value.copy_calls), 2
            )
        for value in reduced:
            self.assertEqual(len(value.zero_calls), 1)
            self.assertEqual(len(value.copy_calls), 1)
            self.assertEqual(len(value.masked_fill_calls), 1)

        event = buffer.reduce_grad(
            plan=plan,
            async_finish=True,
            full_gate_grad=full[0],
            full_up_grad=full[1],
            full_down_grad=full[2],
            gate_reduce_buffer=reduced[0],
            up_reduce_buffer=reduced[1],
            down_reduce_buffer=reduced[2],
        )
        self.assertIsNotNone(event)
        event.wait(torch.npu.current_stream())
        next_event = buffer.reduce_grad(
            plan=plan,
            async_finish=True,
            full_gate_grad=full[0],
            full_up_grad=full[1],
            full_down_grad=full[2],
            gate_reduce_buffer=reduced[0],
            up_reduce_buffer=reduced[1],
            down_reduce_buffer=reduced[2],
        )
        next_event.wait(torch.npu.current_stream())
        self.assertEqual(
            [call[0] for call in runtime.calls].count("reduce_grad"), 3
        )
        buffer.destroy()

    def test_reduce_grad_requires_plan_and_all_tensors(self):
        torch, runtime, buffer, plan = planned_buffer()
        full = tensor((10, 8, 8), torch.float32)
        reduced = tensor((2, 2, 8, 8), torch.float32)
        call_count = len(runtime.calls)
        with self.assertRaisesRegex(AssertionError, "plan is required"):
            buffer.reduce_grad(
                full_gate_grad=full,
                full_up_grad=full,
                full_down_grad=full,
                gate_reduce_buffer=reduced,
                up_reduce_buffer=reduced,
                down_reduce_buffer=reduced,
            )
        with self.assertRaisesRegex(AssertionError, "provided together"):
            buffer.reduce_grad(
                plan=plan,
                full_gate_grad=full,
                full_up_grad=full,
                full_down_grad=full,
                gate_reduce_buffer=reduced,
                up_reduce_buffer=reduced,
            )
        self.assertEqual(len(runtime.calls), call_count)
        buffer.destroy()

    def test_async_reduce_event_wait_surfaces_device_status(self):
        torch, _, buffer, plan = planned_buffer()
        full = [tensor((10, 8, 8), torch.float32) for _ in range(3)]
        reduced = [tensor((2, 2, 8, 8), torch.float32) for _ in range(3)]
        event = buffer.reduce_grad(
            plan=plan,
            async_finish=True,
            full_gate_grad=full[0],
            full_up_grad=full[1],
            full_down_grad=full[2],
            gate_reduce_buffer=reduced[0],
            up_reduce_buffer=reduced[1],
            down_reduce_buffer=reduced[2],
        )
        plan._native_plan.reduce_grad_status._item = 9
        with self.assertRaisesRegex(RuntimeError, "actual 9"):
            event.wait(torch.npu.current_stream())
        buffer.destroy()

    def test_package_exports_and_lifecycle_match_upstream(self):
        self.assertIs(package.Buffer, Buffer)
        self.assertIs(package.MoonEPCommPlan, MoonEPCommPlan)
        self.assertIn("Buffer", package.__all__)
        self.assertIn("MoonEPCommPlan", package.__all__)

        torch, _, native_context = context(prefetch_slots=2)
        with patch.object(
            TileXRMoonEPContext, "from_env", return_value=native_context
        ), patch.object(compat, "_torch", return_value=torch):
            with Buffer(4, 8, 2, 8, 2, B=2) as buffer:
                self.assertFalse(buffer.destroyed)
            self.assertTrue(buffer.destroyed)
            buffer.destroy()

    def test_zero_copy_context_view_matches_dispatch_output(self):
        torch, _, buffer = compat_buffer()
        hidden, weights, _, _ = buffer.dispatch(
            tensor((4, 8), torch.bfloat16),
            tensor((4, 2), torch.float32),
            tensor((4, 2), torch.int32),
            tensor((8,), torch.int32),
            zero_copy=True,
        )
        visible = buffer._require_ctx()
        self.assertEqual(visible["hidden_buf_local"].data_ptr(), hidden.data_ptr())
        self.assertEqual(visible["weights_buf_local"].data_ptr(), weights.data_ptr())
        buffer.destroy()

    def test_six_stage_mock_flow_uses_upstream_surface(self):
        torch, runtime, buffer = compat_buffer()
        hidden_nvsh, weights_nvs, cu_seqlens, plan = buffer.dispatch(
            tensor((4, 8), torch.bfloat16),
            tensor((4, 2), torch.float32),
            tensor((4, 2), torch.int32),
            tensor((8,), torch.int32),
            zero_copy=True,
        )
        full_weights = [tensor((10, 8, 8), torch.bfloat16) for _ in range(3)]
        buffer.prefetch_weight(
            plan=plan.clone(),
            full_gate_weight=full_weights[0],
            full_up_weight=full_weights[1],
            full_down_weight=full_weights[2],
        )

        expert_output = hidden_nvsh
        self.assertEqual(expert_output.data_ptr(), hidden_nvsh.data_ptr())
        combined, gathered_weights, _ = buffer.combine(
            plan=plan,
            hidden_nvsh=expert_output,
            route_weights_nvs=weights_nvs,
            zero_copy=True,
        )
        self.assertEqual(combined.shape, (4, 8))
        self.assertEqual(gathered_weights.shape, (4, 2))
        self.assertEqual(cu_seqlens.shape, (10,))

        full_grads = [tensor((10, 8, 8), torch.float32) for _ in range(3)]
        reduce_buffers = [
            tensor((2, 2, 8, 8), torch.float32) for _ in range(3)
        ]
        buffer.reduce_grad(
            plan=plan,
            full_gate_grad=full_grads[0],
            full_up_grad=full_grads[1],
            full_down_grad=full_grads[2],
            gate_reduce_buffer=reduce_buffers[0],
            up_reduce_buffer=reduce_buffers[1],
            down_reduce_buffer=reduce_buffers[2],
        )

        calls = [call[0] for call in runtime.calls]
        positions = [
            calls.index(stage)
            for stage in (
                "planning",
                "dispatch",
                "combine",
                "reduce_grad",
            )
        ]
        self.assertEqual(positions, sorted(positions))
        buffer.destroy()


class NpuE2ESourceContractTests(unittest.TestCase):
    def test_npu_e2e_preserves_upstream_public_contract(self):
        source = (
            ROOT / "tools" / "moonep" / "test_npu_e2e.py"
        ).read_text(encoding="utf-8")
        for required in (
            "from tilexr_moonep import Buffer, MoonEPCommPlan",
            'backend="hccl"',
            'REQUIRED_UDMA_QP_ROUTE_SPEC = "port_count:6,port_count:2"',
            "plan_snapshot = plan_sync.clone()",
            "async_finish=True",
            "zero_copy=True",
            "prefetch_event.wait(torch.npu.current_stream())",
            "reduce_event.wait(torch.npu.current_stream())",
            "buffer.destroy()",
            "dist.destroy_process_group()",
            "def run_npu_e2e()",
            "def main(",
            '"torch.distributed.run"',
            "subprocess.run",
        ):
            self.assertIn(required, source)
        self.assertNotIn("import pytest", source)
        self.assertNotIn("torch.cuda", source)
        self.assertNotIn('backend="nccl"', source)
        self.assertNotIn("from moonep", source)


if __name__ == "__main__":
    unittest.main()
