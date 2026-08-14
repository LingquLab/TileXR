from __future__ import annotations

import os

import moonep as upstream_moonep

from .moonep_backend import MOONEP_NATIVE_NPU_CAPABILITY, MoonEPBufferFlexBackend


_STAGE_BARRIER_ENV = "MOONEP_MINDSPEED_STAGE_BARRIER"


def optional_stage_barrier(torch_module) -> None:
    if os.environ.get(_STAGE_BARRIER_ENV, "0") != "1":
        return
    distributed = torch_module.distributed
    if not distributed.is_available() or not distributed.is_initialized():
        raise RuntimeError(
            "MoonEP stage barrier requires an initialized process group"
        )
    distributed.barrier()


class MindSpeedBarrierMoonEPBuffer(upstream_moonep.Buffer):
    """Native MoonEP Buffer with opt-in pre-Dispatch/Combine synchronization."""

    def dispatch(self, *args, **kwargs):
        import torch

        optional_stage_barrier(torch)
        return super().dispatch(*args, **kwargs)

    def combine(self, *args, **kwargs):
        import torch

        optional_stage_barrier(torch)
        return super().combine(*args, **kwargs)


def create_native_barrier_backend(**kwargs):
    kwargs["buffer_cls"] = MindSpeedBarrierMoonEPBuffer
    return MoonEPBufferFlexBackend(**kwargs)


create_native_barrier_backend.__mindspeed_capabilities__ = {
    MOONEP_NATIVE_NPU_CAPABILITY
}
