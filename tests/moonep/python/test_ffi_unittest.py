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
            ctypes.cast(native, ctypes.POINTER(ctypes.c_uint64)).contents.value = 1
            ctypes.cast(stub, ctypes.POINTER(ctypes.c_uint64)).contents.value = 30
            return 0

        def workspace(comm, s, k, e, workspace_bytes, capacity):
            self.assert_scalar(comm, 0x1234)
            self.assert_scalar(s, 4)
            self.assert_scalar(k, 2)
            self.assert_scalar(e, 4)
            ctypes.cast(workspace_bytes, ctypes.POINTER(ctypes.c_uint64)).contents.value = 256
            ctypes.cast(capacity, ctypes.POINTER(ctypes.c_int64)).contents.value = 8
            return 0

        def planning(args_ptr, stream):
            args = ctypes.cast(
                args_ptr, ctypes.POINTER(TileXRMoonEPPlanningArgsV1)
            ).contents
            topk = args.topkExperts.contents
            tpe = args.tokensPerExpert.contents
            plan = args.plan.contents
            self.planning_records.append(
                {
                    "stream": stream.value,
                    "workspace_bytes": args.workspaceBytes,
                    "wait_iterations": args.waitIterations,
                    "flags": args.flags,
                    "topk_shape": tuple(topk.shape[: topk.rank]),
                    "tpe_shape": tuple(tpe.shape[: tpe.rank]),
                    "plan": (plan.s, plan.k, plan.e, plan.b, plan.rank, plan.world,
                             plan.dispatchedCapacity),
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
            input_tensor = args.input.contents
            output_tensor = args.output.contents
            self.stage_records.append(
                {
                    "name": name,
                    "stream": stream.value,
                    "flags": args.flags,
                    "plan_capacity": plan.dispatchedCapacity,
                    "input_shape": tuple(input_tensor.shape[: input_tensor.rank]),
                    "output_shape": tuple(output_tensor.shape[: output_tensor.rank]),
                }
            )
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
        self.assertEqual(ctypes.sizeof(TileXRMoonEPPlanV1), 104)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPPlanningArgsV1), 72)
        self.assertEqual(ctypes.sizeof(TileXRMoonEPDispatchArgsV1), 48)
        self.assertEqual(TileXRMoonEPTensorV1.shape.offset, 32)
        self.assertEqual(TileXRMoonEPPlanV1.dst.offset, 64)
        self.assertEqual(TileXRMoonEPPlanningArgsV1.flags.offset, 64)
        self.assertEqual(TileXRMoonEPDispatchArgsV1.flags.offset, 40)

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
        )
        buffer = TileXRMoonEPBuffer(context, wait_iterations=1234, torch_module=torch)
        plan = buffer.planning(tensor((4, 2), torch.int32), tensor((4,), torch.int32))
        dispatched = buffer.dispatch(
            tensor((4, 8), torch.bfloat16), plan, tensor((4, 2), torch.float32)
        )
        projections = ProjectionBuffers(
            tensor((6, 8), torch.bfloat16),
            tensor((6, 8), torch.bfloat16),
            tensor((6, 8), torch.bfloat16),
        )
        prefetched = buffer.prefetch_weight(plan, projections)
        buffer.combine(dispatched.hidden, plan, dispatched.route_weights)
        gradients = ProjectionBuffers(
            tensor((6, 8), torch.float32),
            tensor((6, 8), torch.float32),
            tensor((6, 8), torch.float32),
            tensor((6, 8), torch.float32),
            tensor((6, 8), torch.float32),
            tensor((6, 8), torch.float32),
        )
        buffer.reduce_grad(plan, gradients)
        self.assertEqual(tuple(prefetched.gate.shape), (6, 8))
        buffer.close()

        self.assertEqual(
            [Path(path).name for path, _ in loader.loads],
            ["libtile-comm.so", "libtilexr-moonep-planner.so", "libtilexr-moonep.so.1"],
        )
        self.assertEqual(runtime.capabilities.stage_mask, 1)
        self.assertEqual(runtime.capabilities.stub_mask, 30)
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
                "plan": (4, 2, 4, 2, 0, 2, 8),
            }],
        )
        self.assertEqual(
            [record["name"] for record in loader.stage_records],
            [
                "dispatch",
                "dispatch",
                "prefetch_weight",
                "prefetch_weight",
                "prefetch_weight",
                "combine",
                "combine",
                "reduce_grad",
                "reduce_grad",
                "reduce_grad",
            ],
        )
        self.assertTrue(all(record["stream"] == 0xCAFE for record in loader.stage_records))
        self.assertTrue(all(record["flags"] == 0 for record in loader.stage_records))
        self.assertEqual(loader.stage_records[1]["input_shape"], (4, 2))
        self.assertEqual(loader.stage_records[1]["output_shape"], (8,))
        self.assertEqual(loader.stage_records[6]["input_shape"], (8,))
        self.assertEqual(loader.stage_records[6]["output_shape"], (4, 2))

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
