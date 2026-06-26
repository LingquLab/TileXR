# vllm-ascend TileXR Collectives Phase 3 Design

## Goal

Connect TileXR collectives to the vllm/vllm-ascend collective path from the rewritten `main` branch, keep the integration opt-in, validate it on `ssh blue`, and document the supported path well enough for a PR review.

## Current State

`main` already contains the standalone TileXR collectives library and public C API for `TileXRAllGather`, `TileXRAllToAll`, `TileXRAllReduce`, `TileXRReduceScatter`, and `TileXRBroadcast`. The Phase 2 vllm shim has been ported to the `main` baseline as a separate commit, but it currently exposes only `all_gather` and equal-split `all_to_all` to Python.

vllm routes group collectives through `GroupCoordinator`, and the out-of-tree Ascend platform uses `NPUCommunicator`. In the inspected vllm-ascend code, `NPUCommunicator` customizes only `all_to_all`; `all_reduce`, `all_gather`, and `reduce_scatter` inherit the base implementation that calls `torch.distributed`. Phase 3 will add a TileXR-backed communicator adapter that can be used by vllm-ascend when explicitly enabled.

## Approach

Use an opt-in Python integration layer rather than changing TileXR host/kernels again. TileXR remains the collective provider under `integrations/vllm_ascend/tilexr_collectives`; vllm-ascend can import this module and delegate compatible group collectives to it. Unsupported operations, dtypes, dimensions, uneven collectives, missing libraries, or disabled feature flags fall back to existing vllm-ascend behavior.

The integration is controlled by environment variables:

- `VLLM_ASCEND_TILEXR_COLLECTIVES=1` enables the TileXR path.
- `TILEXR_INSTALL_PREFIX`, `TILEXR_COMM_LIB`, and `TILEXR_COLLECTIVES_LIB` keep the existing library discovery behavior.

## Runtime API

Extend `TileXRCollectivesRuntime` to configure and call:

- `TileXRAllGather`
- `TileXRAllToAll`
- `TileXRAllReduce` with SUM only
- `TileXRReduceScatter` with SUM only
- `TileXRBroadcast`

The torch helper layer will expose:

- `all_gather(tensor, rank, world_size, install_prefix, dim=-1, runtime=None)`
- `all_to_all(tensor, rank, world_size, install_prefix, scatter_dim=0, gather_dim=-1, runtime=None)`
- `all_reduce(tensor, rank, world_size, install_prefix, runtime=None)`
- `reduce_scatter(tensor, rank, world_size, install_prefix, dim=-1, runtime=None)`
- `broadcast(tensor, rank, world_size, install_prefix, root=0, runtime=None)`

For vllm compatibility, all-gather and reduce-scatter accept an arbitrary dimension by moving that dimension to the first contiguous dimension before calling TileXR, then moving it back. The first implementation supports equal-size all-to-all only. `all_gatherv` and `reduce_scatterv` stay unsupported and should fall back.

## vllm-ascend Adapter

Add a small adapter module under `integrations/vllm_ascend/tilexr_collectives` that mirrors the methods used by vllm `DeviceCommunicatorBase`:

- `enabled()`
- `all_reduce(input_)`
- `all_gather(input_, dim=-1)`
- `reduce_scatter(input_, dim=-1)`
- `all_to_all(input_, scatter_dim=0, gather_dim=-1, scatter_sizes=None, gather_sizes=None)`

The adapter takes rank, world size, current device, and install prefix from the caller. It never imports vllm at module import time. A downstream vllm-ascend patch can instantiate it inside `NPUCommunicator` and call it before falling back to the existing HCCL/torch path.

The TileXR path is selected only when:

- the feature flag is enabled;
- the tensor is on NPU and contiguous after any required dimension move;
- the dtype maps to a TileXR-supported dtype;
- the operation shape is equal-split where required;
- the TileXR libraries can be loaded and the communicator initializes successfully.

## Smoke and Remote Validation

`smoke_collectives.py` and `run_tilexr_collectives_smoke.sh` will be extended beyond AllGather:

- `allgather int32/fp16`
- `allreduce int32/fp16`
- `reducescatter int32/fp16`
- `broadcast int32/fp16`
- `alltoall int32/fp16` when the remote topology supports it

The remote deployment script will run the standalone collectives checks first, then the Python shim smoke. It will also perform a vllm-ascend environment probe. If `vllm_ascend` is importable, it should run a communicator-level import/adapter probe; otherwise it records the missing package and continues with the TileXR shim validation. Full vllm inference validation is required before the final PR is considered complete, but the script must make package/environment gaps explicit rather than silently treating them as success.

## Documentation

Update `tests/collectives/README.md` with:

- the new Phase 3 feature flag;
- the operation coverage and fallback boundaries;
- remote validation commands;
- the distinction between TileXR shim validation and full vllm-ascend inference validation.

Add a concise root or integration README only if needed to point reviewers to the vllm integration entrypoint; keep operational detail in `tests/collectives/README.md`.

## Testing

Local tests:

- Source guards proving all 5 TileXR C APIs are configured and exposed.
- Source guards proving the adapter is opt-in and avoids vllm import-time coupling.
- `pytest` for Python source tests.
- `bash -n` for launch/deploy scripts.
- `py_compile` for Python integration files.
- `git diff --check`.

Remote tests on `blue`:

- Build/install TileXR collectives from the Phase 3 branch.
- CTest source/unit checks.
- 2-rank Python shim smoke for the covered operations.
- vllm/vllm-ascend import and communicator/inference probes when the environment supports them.

## PR Completion Criteria

The PR can be submitted only after:

- Phase 3 branch is based on `origin/main`.
- Phase 2 migration remains isolated from old collectives history.
- The opt-in TileXR adapter is implemented with fallback boundaries documented.
- Local checks pass.
- Remote `blue` checks pass or expose a separately documented environment blocker that has been resolved before final PR.
- Docs are updated.
