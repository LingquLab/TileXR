from __future__ import annotations

from types import SimpleNamespace

import torch

from tools.moonep.benchmark import _reference_process_group_backend
from tools.moonep.contracts import MoonEPDimensions
from tools.moonep.torch_npu_backend import TorchDistributedCollective


class FakeDistributed:
    ReduceOp = SimpleNamespace(MIN="min")

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.gather_input_device = None

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def get_world_size(self, _group) -> int:
        return 2

    def get_rank(self, _group) -> int:
        return 0

    def get_backend(self, _group) -> str:
        return self.backend

    def all_gather(self, outputs, tensor, *, group) -> None:
        del group
        self.gather_input_device = tensor.device.type
        outputs[0].copy_(tensor)
        outputs[1].copy_(tensor + 1)

    def all_reduce(self, value, *, op, group) -> None:
        del value, group
        assert op == self.ReduceOp.MIN


class FakeNpuTensor:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.device = SimpleNamespace(type="npu")
        self.cpu_calls = 0

    def cpu(self) -> torch.Tensor:
        self.cpu_calls += 1
        return self.value


class RestoredStack:
    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.device = SimpleNamespace(type="cpu")
        self.restored_device = None

    def to(self, *, device):
        self.restored_device = device
        return self


class FakeTorch:
    int32 = torch.int32

    def __init__(self, backend: str) -> None:
        self.distributed = FakeDistributed(backend)
        self.tensor_devices = []

    @staticmethod
    def empty_like(tensor):
        return torch.empty_like(tensor)

    @staticmethod
    def stack(tensors, *, dim):
        return RestoredStack(torch.stack(tensors, dim=dim))

    def tensor(self, data, *, dtype, device):
        self.tensor_devices.append(device)
        return torch.tensor(data, dtype=dtype, device=device)


def dimensions() -> MoonEPDimensions:
    return MoonEPDimensions(0, 2, 2, 1, 4, 2, 1, 4, 2)


def test_reference_collective_backend_uses_gloo_only_when_oversubscribed() -> None:
    assert _reference_process_group_backend({"TILEXR_OVERSUBSCRIBED": "0"}) == "hccl"
    assert _reference_process_group_backend({"TILEXR_OVERSUBSCRIBED": "1"}) == "gloo"
    assert _reference_process_group_backend({}) == "hccl"


def test_gloo_collective_stages_npu_tensors_through_cpu() -> None:
    torch_module = FakeTorch("gloo")
    tensor = FakeNpuTensor(torch.tensor([3, 5], dtype=torch.int32))
    collective = TorchDistributedCollective(torch_module, dimensions())

    gathered = collective.all_gather(tensor)

    assert tensor.cpu_calls == 1
    assert torch_module.distributed.gather_input_device == "cpu"
    assert torch.equal(gathered.value, torch.tensor([[3, 5], [4, 6]]))
    assert gathered.restored_device is tensor.device


def test_gloo_rank_agreement_uses_cpu() -> None:
    torch_module = FakeTorch("gloo")
    collective = TorchDistributedCollective(torch_module, dimensions())

    assert collective.all_agree(True, device=SimpleNamespace(type="npu"))
    assert torch_module.tensor_devices == ["cpu"]
