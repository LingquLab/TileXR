from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
INTEGRATION = ROOT / "integrations" / "moonep_torch"
for path in (INTEGRATION, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fakes import FakeTensor, FakeTorch
from tilexr_moonep import ProjectionBuffers, TileXRMoonEPBuffer, TileXRMoonEPContext
from tilexr_moonep.abi import (
    TileXRMoonEPCombineArgsV1,
    TileXRMoonEPDispatchArgsV1,
    TileXRMoonEPPlanV1,
    TileXRMoonEPPlanningArgsV1,
    TileXRMoonEPPrefetchWeightArgsV1,
    TileXRMoonEPReduceGradArgsV1,
    TileXRMoonEPTensorV1,
)
from tilexr_moonep.runtime import TileXRMoonEPRuntime, _resolve_library


class FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class FakeLibrary:
    pass


class FakeCDLLLoader:
    def __init__(self):
        self.loads = []
        self.destroy_calls = 0
        self.planning_records = []
        self.stage_records = []
        self.comm = self._comm_library()
        self.planner = FakeLibrary()
        self.moonep = self._moonep_library()

    def _comm_library(self):
        library = FakeLibrary()

        def init_rank_local(world, rank, output):
            self.assert_scalar(world, 2)
            self.assert_scalar(rank, 0)
            ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0x1234
            return 0

        def destroy(comm):
            self.assert_scalar(comm, 0x1234)
            self.destroy_calls += 1
            return 0

        library.TileXRCommInitRankLocal = FakeFunction(init_rank_local)
        library.TileXRCommDestroy = FakeFunction(destroy)
        return library

    @staticmethod
    def assert_scalar(value, expected):
        actual = value.value if hasattr(value, "value") else value
        if int(actual) != expected:
            raise AssertionError(f"expected {expected}, got {actual}")

    def _moonep_library(self):
        library = FakeLibrary()
        library.TileXRMoonEpGetAbiVersion = FakeFunction(lambda: 1)

        def capabilities(native, stub):
            ctypes.cast(native, ctypes.POINTER(ctypes.c_uint64)).contents.value = 31
            ctypes.cast(stub, ctypes.POINTER(ctypes.c_uint64)).contents.value = 0
            return 0

        def workspace(comm, s, k, e, b, token_padding, workspace_bytes, capacity):
            self.assert_scalar(comm, 0x1234)
            self.assert_scalar(s, 4)
            self.assert_scalar(k, 2)
            self.assert_scalar(e, 4)
            self.assert_scalar(b, 1)
            self.assert_scalar(token_padding, 2)
            ctypes.cast(workspace_bytes, ctypes.POINTER(ctypes.c_uint64)).contents.value = 256
            ctypes.cast(capacity, ctypes.POINTER(ctypes.c_int64)).contents.value = 12
            return 0

        def planning(args_ptr, stream):
            args = ctypes.cast(
                args_ptr, ctypes.POINTER(TileXRMoonEPPlanningArgsV1)
            ).contents
            topk = args.topkExperts.contents
            tpe = args.tokensPerExpert.contents
            cu = args.cuSeqlens.contents
            plan = args.plan.contents
            self.planning_records.append(
                {
                    "stream": stream.value,
                    "workspace_bytes": args.workspaceBytes,
                    "wait_iterations": args.waitIterations,
                    "flags": args.flags,
                    "topk_shape": tuple(topk.shape[: topk.rank]),
                    "tpe_shape": tuple(tpe.shape[: tpe.rank]),
                    "cu_shape": tuple(cu.shape[: cu.rank]),
                    "plan": (plan.n, plan.r, plan.e, plan.b, plan.nvS, plan.k),
                }
            )
            return 0

        library.TileXRMoonEpGetCapabilitiesV1 = FakeFunction(capabilities)
        library.TileXRMoonEpPlanningGetWorkspaceSizeV1 = FakeFunction(workspace)
        library.TileXRMoonEpPlanningV1 = FakeFunction(planning)
        for name, args_type in (
            ("dispatch", TileXRMoonEPDispatchArgsV1),
            ("prefetch_weight", TileXRMoonEPPrefetchWeightArgsV1),
            ("combine", TileXRMoonEPCombineArgsV1),
            ("reduce_grad", TileXRMoonEPReduceGradArgsV1),
        ):
            setattr(
                library,
                {
                    "dispatch": "TileXRMoonEpDispatchV1",
                    "prefetch_weight": "TileXRMoonEpPrefetchWeightV1",
                    "combine": "TileXRMoonEpCombineV1",
                    "reduce_grad": "TileXRMoonEpReduceGradV1",
                }[name],
                FakeFunction(self._stage_callback(name, args_type)),
            )
        return library

    def _stage_callback(self, name, args_type):
        def callback(args_ptr, stream):
            args = ctypes.cast(args_ptr, ctypes.POINTER(args_type)).contents
            plan = args.plan.contents
            fields = {
                "dispatch": ("hiddenSh", "routeWeightsSk", "hiddenNvsh", "routeWeightsNvs"),
                "prefetch_weight": ("fullGateWeight", "fullUpWeight", "fullDownWeight"),
                "combine": ("hiddenNvsh", "routeWeightsNvs", "hiddenSh", "routeWeightsSk"),
                "reduce_grad": (
                    "fullGateGrad", "fullUpGrad", "fullDownGrad",
                    "gateReduceBuffer", "upReduceBuffer", "downReduceBuffer",
                ),
            }[name]
            shapes = {}
            for field in fields:
                pointer = getattr(args, field)
                shapes[field] = None if not pointer else tuple(
                    pointer.contents.shape[: pointer.contents.rank]
                )
            self.stage_records.append({
                "name": name,
                "stream": stream.value,
                "flags": args.flags,
                "plan_capacity": plan.nvS,
                "shapes": shapes,
            })
            return 0

        return callback

    def __call__(self, path, mode):
        self.loads.append((str(path), mode))
        return (self.comm, self.planner, self.moonep)[len(self.loads) - 1]


def tensor(shape, dtype):
    return FakeTensor(shape, dtype, "npu:0")


class FfiAbiTests(unittest.TestCase):
    def test_ctypes_layout_matches_tilexr_moonep_header(self):
        self.assertEqual(ctypes.sizeof(TileXRMoonEPTensorV1), 64)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPPlanV1), 120)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPPlanningArgsV1), 80)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPDispatchArgsV1), 64)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPPrefetchWeightArgsV1), 56)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPCombineArgsV1), 64)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPReduceGradArgsV1), 80)
        self.assertEqual(TileXRMoonEPTensorV1.shape.offset, 32)
        self.assertEqual(TileXRMoonEPPlanV1.dst.offset, 56)
        self.assertEqual(TileXRMoonEPPlanV1.status.offset, 112)
        self.assertEqual(TileXRMoonEPPlanningArgsV1.cuSeqlens.offset, 48)
        self.assertEqual(TileXRMoonEPPlanningArgsV1.flags.offset, 72)
        self.assertEqual(TileXRMoonEPDispatchArgsV1.flags.offset, 56)

    def test_fake_cdll_receives_exact_v1_descriptors_and_current_stream(self):
        loader = FakeCDLLLoader()
        runtime = TileXRMoonEPRuntime(
            rank=0,
            world_size=2,
            library_paths={
                "comm": "libtile-comm.so",
                "planner": "libtilexr-moonep-planner.so",
                "moonep": "libtilexr-moonep.so.1",
            },
            cdll_loader=loader,
        )
        torch = FakeTorch()
        context = TileXRMoonEPContext(
            runtime=runtime,
            global_rank=0,
            global_world_size=2,
            node_rank=0,
            node_count=1,
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
            expert_count=4,
            dtype=torch.bfloat16,
            token_padding=2,
            prefetch_slots=1,
        )
        buffer = TileXRMoonEPBuffer(context, wait_iterations=1234, torch_module=torch)
        hidden_nvsh, route_weights_nvs, cu_seqlens, plan = buffer.dispatch(
            tensor((4, 8), torch.bfloat16),
            tensor((4, 2), torch.float32),
            tensor((4, 2), torch.int32),
            tensor((4,), torch.int32),
        )
        projections = ProjectionBuffers(
            tensor((5, 2, 4), torch.bfloat16),
            tensor((5, 4, 3), torch.bfloat16),
            tensor((5, 3, 2), torch.bfloat16),
        )
        buffer.prefetch_weight(
            plan,
            full_gate_weight=projections.gate,
            full_up_weight=projections.up,
            full_down_weight=projections.down,
        )
        buffer.combine(plan, hidden_nvsh, route_weights_nvs)
        gradients = ProjectionBuffers(
            tensor((5, 2, 4), torch.float32),
            tensor((5, 4, 3), torch.float32),
            tensor((5, 3, 2), torch.float32),
            tensor((2, 1, 2, 4), torch.float32),
            tensor((2, 1, 4, 3), torch.float32),
            tensor((2, 1, 3, 2), torch.float32),
        )
        buffer.reduce_grad(
            plan,
            full_gate_grad=gradients.gate,
            full_up_grad=gradients.up,
            full_down_grad=gradients.down,
            gate_reduce_buffer=gradients.gate_reduce,
            up_reduce_buffer=gradients.up_reduce,
            down_reduce_buffer=gradients.down_reduce,
        )
        self.assertEqual(tuple(cu_seqlens.shape), (5,))
        plan.status._item = 5000
        buffer.close()

        self.assertEqual(
            [Path(path).name for path, _ in loader.loads],
            ["libtile-comm.so", "libtilexr-moonep-planner.so", "libtilexr-moonep.so.1"],
        )
        self.assertEqual(runtime.capabilities.stage_mask, 31)
        self.assertEqual(runtime.capabilities.stub_mask, 0)
        self.assertTrue(runtime.capabilities.transport_correctness_valid)
        self.assertFalse(runtime.capabilities.transport_performance_valid)
        self.assertEqual(loader.destroy_calls, 1)
        self.assertEqual(
            loader.planning_records,
            [{
                "stream": 0xCAFE,
                "workspace_bytes": 256,
                "wait_iterations": 1234,
                "flags": 0,
                "topk_shape": (4, 2),
                "tpe_shape": (4,),
                "cu_shape": (5,),
                "plan": (8, 2, 4, 1, 12, 2),
            }],
        )
        self.assertEqual(
            [record["name"] for record in loader.stage_records],
            ["dispatch", "prefetch_weight", "combine", "reduce_grad"],
        )
        self.assertTrue(all(record["stream"] == 0xCAFE for record in loader.stage_records))
        self.assertEqual(loader.stage_records[0]["flags"], 1)
        self.assertTrue(all(record["flags"] == 0 for record in loader.stage_records[1:]))
        self.assertEqual(loader.stage_records[0]["shapes"]["hiddenNvsh"], (12, 8))
        self.assertEqual(loader.stage_records[0]["shapes"]["routeWeightsNvs"], (12,))
        self.assertEqual(loader.stage_records[1]["shapes"]["fullUpWeight"], (5, 4, 3))
        self.assertEqual(loader.stage_records[2]["shapes"]["routeWeightsSk"], (4, 2))
        self.assertEqual(
            loader.stage_records[3]["shapes"]["gateReduceBuffer"], (2, 1, 2, 4)
        )

    def test_versioned_planner_library_is_preferred(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            library_dir = prefix / "lib"
            library_dir.mkdir()
            (library_dir / "libtilexr-moonep-planner.so").touch()
            versioned_dir = prefix / "lib64"
            versioned_dir.mkdir()
            versioned = versioned_dir / "libtilexr-moonep-planner.so.2"
            versioned.touch()
            resolved = _resolve_library(
                ("libtilexr-moonep-planner.so.2", "libtilexr-moonep-planner.so"),
                "TILEXR_TEST_UNUSED_LIBRARY",
                prefix,
            )
            self.assertEqual(Path(resolved), versioned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
