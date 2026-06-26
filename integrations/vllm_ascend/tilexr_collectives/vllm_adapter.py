from __future__ import annotations

import os
from dataclasses import dataclass, field

from .torch_collectives import all_gather, all_reduce, all_to_all, broadcast, reduce_scatter


_ENABLED_VALUES = {"1", "true", "yes", "on"}


def _flag_enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in _ENABLED_VALUES


def enabled() -> bool:
    return _flag_enabled(os.environ.get("VLLM_ASCEND_TILEXR_COLLECTIVES"))


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise ValueError(f"invalid dim={dim} for tensor with ndim={ndim}")
    return dim


def _same_normalized_dim(tensor, scatter_dim: int, gather_dim: int) -> bool:
    ndim = int(tensor.dim())
    return _normalize_dim(scatter_dim, ndim) == _normalize_dim(gather_dim, ndim)


@dataclass
class TileXRVllmCollectivesAdapter:
    rank: int
    world_size: int
    install_prefix: str
    last_error: BaseException | None = field(default=None, init=False, repr=False)

    def should_fallback(self, tensor=None, *, scatter_sizes=None, gather_sizes=None) -> bool:
        self.last_error = None
        if not enabled():
            return True
        if tensor is None:
            return True
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) != "npu":
            return True
        is_contiguous = getattr(tensor, "is_contiguous", None)
        if not callable(is_contiguous) or not is_contiguous():
            return True
        return scatter_sizes is not None or gather_sizes is not None

    def _call_or_fallback(self, callback):
        try:
            return callback()
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
            self.last_error = exc
            return None

    def all_reduce(self, input_):
        if self.should_fallback(input_):
            return None
        return self._call_or_fallback(
            lambda: all_reduce(input_, self.rank, self.world_size, self.install_prefix)
        )

    def all_gather(self, input_, dim: int = -1):
        if self.should_fallback(input_):
            return None
        return self._call_or_fallback(
            lambda: all_gather(input_, self.rank, self.world_size, self.install_prefix, dim=dim)
        )

    def reduce_scatter(self, input_, dim: int = -1):
        if self.should_fallback(input_):
            return None
        return self._call_or_fallback(
            lambda: reduce_scatter(input_, self.rank, self.world_size, self.install_prefix, dim=dim)
        )

    def broadcast(self, tensor, src: int = 0):
        if self.should_fallback(tensor):
            return None
        return self._call_or_fallback(
            lambda: broadcast(tensor, self.rank, self.world_size, self.install_prefix, root=src)
        )

    def all_to_all(
        self,
        input_,
        scatter_dim: int = 0,
        gather_dim: int = -1,
        scatter_sizes: list[int] | None = None,
        gather_sizes: list[int] | None = None,
    ):
        if self.should_fallback(input_, scatter_sizes=scatter_sizes, gather_sizes=gather_sizes):
            return None
        if not self._same_normalized_dim(input_, scatter_dim, gather_dim):
            return None
        return self._call_or_fallback(
            lambda: all_to_all(
                input_,
                self.rank,
                self.world_size,
                self.install_prefix,
                scatter_dim=scatter_dim,
                gather_dim=gather_dim,
            )
        )

    def _same_normalized_dim(self, tensor, scatter_dim: int, gather_dim: int) -> bool:
        try:
            return _same_normalized_dim(tensor, scatter_dim, gather_dim)
        except (AttributeError, TypeError, ValueError) as exc:
            self.last_error = exc
            return False
