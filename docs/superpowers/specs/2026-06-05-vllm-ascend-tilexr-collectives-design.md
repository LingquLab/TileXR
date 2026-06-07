# vllm-ascend TileXR Collectives Integration Design

## Context

TileXR provides Ascend C/C++ communication libraries:

- `libtile-comm.so` owns rank initialization, peer memory, `CommArgs`, and optional UDMA/SDMA metadata.
- `libtilexr-collectives.so` exposes `TileXRAllGather` and equal-size `TileXRAllToAll` on top of `tile-comm`.
- `libtilexr-ep.so` exposes a standalone MoE EP dispatch API, but this design intentionally starts with collectives.

vllm-ascend is a vLLM hardware plugin. Its current communication paths include `torch.distributed`, HCCL-backed process groups, a Python HCCL wrapper, and MoE-specific `ALLGATHER`, `ALLTOALL`, `MC2`, and `FUSED_MC2` strategies. The first integration should not replace vllm-ascend wholesale.

Remote installation and hardware validation must run on `ssh blue`. The host currently has eight 910B3 NPUs and driver tooling available, but no active CANN environment variables and no installed `torch`, `torch-npu`, `vllm`, or `vllm-ascend` in the default Python environment. The validation environment must therefore be isolated under a project scratch directory.

## Goals

1. Build an isolated remote validation environment on `blue`.
2. Build and validate TileXR collectives in that environment.
3. Prove that a vllm-ascend Python process can call TileXR collectives with torch-npu NPU tensors through a minimal shim.
4. After the shim is stable, integrate the shim as an optional vllm-ascend collective backend for one inference-path collective.

## Non-Goals

- Do not modify `blue` system Python or global shell startup files.
- Do not replace the full vllm-ascend platform plugin.
- Do not start with MoE EP dispatch replacement.
- Do not mix different CANN runtimes inside one process.
- Do not treat unsupported TileXR `AllToAll` topology as a correctness failure.

## Approach

Use a two-phase integration.

### Phase 1: External Shim Validation

Create an isolated scratch directory on `blue` selected by `TILEXR_VLLM_REMOTE_BASE`, for example:

```text
/path/to/remote/tilexr_vllm_collectives_YYYYMMDD_HHMMSS/
  TileXR/
  logs/
  artifacts/
```

The deployment script syncs the current TileXR commit and initialized local submodule worktrees, probes CANN/Python/NPU state, builds `tile-comm` and `tilexr-collectives`, and writes all logs under `logs/`. When the remote login shell does not expose CANN Toolkit, driver headers, or CCE compiler paths, the script accepts explicit remote path overrides.

Add a minimal TileXR collectives shim that can run inside the vllm-ascend Python environment. The shim accepts torch-npu tensor pointers, dtype, count, rank, world size, and stream information, initializes or reuses a `TileXRCommPtr`, and calls `TileXRAllGather` or `TileXRAllToAll`.

The smoke test launches 2-rank and 8-rank Python processes. Each rank creates NPU tensors with deterministic values, calls the shim, synchronizes, copies results back, and checks correctness.

### Phase 2: Optional vllm-ascend Backend

After Phase 1 passes, wrap the shim behind an opt-in vllm-ascend backend selected by an environment variable such as:

```text
VLLM_ASCEND_TILEXR_COLLECTIVES=1
```

Default behavior remains the upstream vllm-ascend path. The first candidate integration points are:

- `NPUCommunicator.all_to_all` for an isolated all-to-all replacement.
- The all-gather helper path in `vllm_ascend.distributed.utils` for an isolated all-gather replacement.

Only one path should be replaced initially, with a baseline run using `VLLM_ASCEND_TILEXR_COLLECTIVES=0`.

## Components

### Remote Deployment Script

Responsibilities:

- Create and clean a scratch directory on `blue`.
- Sync the current TileXR commit without copying build outputs.
- Sync initialized local submodule worktrees so remote validation does not depend on outbound GitHub access.
- Dump `npu-smi info`, compiler versions, Python version, CANN env, and relevant package versions.
- Build TileXR collectives, using explicit CANN Toolkit, driver, and CCE compiler paths when provided.
- Run TileXR standalone collectives checks.
- Run shim smoke tests.
- Preserve rank logs and summary status.

The script must not install global system packages or mutate global Python state.

### TileXR Collectives Shim

Responsibilities:

- Convert Python inputs into the C ABI expected by TileXR.
- Map torch dtypes to `TileXR::TileXRDataType`.
- Obtain rank, world size, and device from the test launcher or vllm process context.
- Initialize and cache `TileXRCommPtr` per process/group.
- Call `TileXRAllGather` and `TileXRAllToAll`.
- Return success or raise a Python exception containing the TileXR error code and operation metadata.

The shim does not own tensor memory. Tensor lifetimes remain owned by torch-npu.

### vllm-ascend Smoke Tests

Responsibilities:

- Launch multi-rank Python tests in the same environment where vllm-ascend is installed.
- Allocate NPU tensors using torch-npu.
- Call the TileXR shim with tensor `data_ptr()` and operation metadata.
- Validate deterministic outputs.
- Emit per-rank logs.

These tests do not modify vllm-ascend source during Phase 1.

## Data Flow

```text
torch-npu tensor
  -> tensor.data_ptr(), dtype, count, stream, rank metadata
  -> TileXR shim
  -> TileXRCommPtr
  -> TileXRAllGather / TileXRAllToAll
  -> stream synchronization
  -> torch-npu tensor correctness check
```

## Error Handling

- Missing NPU, CANN, Python, torch-npu, vLLM, or vllm-ascend dependencies fail early with an environment report.
- CANN version is recorded before build and test. One process must use one CANN runtime stack.
- `TileXRCommInit*` failures raise Python exceptions with rank, world size, device id, and TileXR error code.
- `TileXRAllGather` and `TileXRAllToAll` nonzero returns raise operation-specific exceptions.
- Stream extraction failures fall back to a correctness-only mode using explicit synchronization, and the report marks performance numbers invalid.
- Unsupported `AllToAll` topology is reported as a known skip, not as a failed integration.

## Test Matrix

Phase 1:

- Build `tile-comm` and `tilexr-collectives`.
- Run TileXR standalone collectives correctness for 2 ranks.
- Run TileXR standalone collectives correctness for 8 ranks when all devices are free.
- Run `AllGather` shim smoke for `int32` and `fp16`.
- Run conditional `AllToAll` shim smoke for `int32`.
- Run a basic TileXR collectives perf command and save bandwidth/latency output.

Phase 2:

- Run baseline vllm-ascend small-model inference with `VLLM_ASCEND_TILEXR_COLLECTIVES=0`.
- Enable one TileXR collective path with `VLLM_ASCEND_TILEXR_COLLECTIVES=1`.
- Run TP=2 small-model inference correctness.
- Only after the single collective path is stable, consider MoE-specific communication integration.

## Risks

- vllm-ascend documentation currently targets CANN 9.0.0, while TileXR is configured for CANN 9.1.0. The remote environment must prove one compatible runtime stack before shim testing.
- `blue` default Python is 3.9, while vllm-ascend requires Python >= 3.10. The validation environment needs an isolated Python runtime.
- TileXR `AllToAll` currently has topology restrictions. `AllGather` is the required first success target.
- Passing the correct `aclrtStream` from torch-npu to TileXR may require a C++ extension instead of pure `ctypes`.
- Full vLLM inference integration may expose shape, graph capture, or stream-ordering assumptions that the external shim smoke cannot cover.

## Success Criteria

Phase 1 is complete when:

- The remote isolated environment can be recreated on `ssh blue`.
- TileXR collectives build successfully in that environment.
- vllm-ascend Python can call TileXR `AllGather` on NPU tensors for at least 2 ranks with deterministic correctness.
- Logs and artifacts are saved under the scratch directory.

Phase 2 is complete when:

- vllm-ascend can run one small-model inference path with one collective routed through TileXR under an opt-in flag.
- The default vllm-ascend path remains unchanged when the flag is disabled.
