from __future__ import annotations

import os
import sys


def _enabled() -> bool:
    value = os.environ.get("VLLM_ASCEND_TILEXR_COLLECTIVES")
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _trace_enabled() -> bool:
    value = os.environ.get("TILEXR_VLLM_TRACE")
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


if _enabled():
    try:
        from tilexr_collectives.vllm_patch import patch_npu_communicator

        patch_npu_communicator()
    except Exception as exc:
        if _trace_enabled():
            print(f"[tilexr-vllm] sitecustomize fallback: {type(exc).__name__}: {exc}", file=sys.stderr)
