# vllm-ascend TileXR Collectives Phase 2 Design

## Context

Phase 1 added a TileXR Python shim under `integrations/vllm_ascend` and a remote validation script for
`ssh blue`. The remote TileXR build now succeeds with explicit CANN Toolkit, CCE compiler, and driver shim paths.
CTest and the standalone TileXR 2-rank collectives correctness runner pass on `blue`.

The remaining Phase 1 blocker is Python environment selection. The default remote Python is
`/home/miniconda3/bin/python3` with Python 3.7.10 and `torch 1.13.1`, but it does not provide `torch_npu`,
`vllm`, or `vllm_ascend`. A usable candidate exists in the `tt4` conda environment:

- Python 3.10.19.
- `torch 2.6.0+cpu`.
- `torch_npu 2.6.0`.
- 8 visible 910B3 NPUs.
- `torch.npu.current_stream().npu_stream` exposes an integer stream handle.

The `blue` host also has vllm/vllm-ascend source directories, but those directories are not git checkouts. The
available vllm-ascend source expects newer `torch==2.9.0` and `torch-npu==2.9.0`, so Phase 2 should not start by
modifying that source tree or by claiming inference-path support.

## Goals

1. Make remote Python selection explicit and reproducible for the vllm collectives validation script.
2. Run the existing TileXR Python shim smoke in the selected `torch_npu` environment on `blue`.
3. Record enough environment and rank-log detail to diagnose CANN/Python/stream/ABI failures.
4. Define the handoff criteria for later vllm-ascend inference-path backend work.

## Non-Goals

- Do not modify remote system Python, conda base, or shell startup files.
- Do not install or upgrade vLLM/vllm-ascend packages during this phase.
- Do not edit `/home/d00520898/vllm-ascend` or `/home/d00520898/vllm/vllm-ascend`.
- Do not replace vllm-ascend inference-path collectives yet.
- Do not treat vllm-ascend import failure as a Phase 2 failure. It is recorded as environment context.

## Approach

Extend `tests/collectives/deploy_and_run_vllm_remote.sh` so all Python package probes and Python smoke commands run
through one selected remote Python environment. The script accepts either a direct Python executable path or a conda
environment name:

```text
TILEXR_VLLM_REMOTE_PYTHON=/path/to/python
TILEXR_VLLM_REMOTE_CONDA_ENV=tt4
```

`TILEXR_VLLM_REMOTE_PYTHON` has precedence. If only `TILEXR_VLLM_REMOTE_CONDA_ENV` is set, the remote script sources
`/home/miniconda3/etc/profile.d/conda.sh`, activates the requested environment, and uses the resulting `python` and
`python3` commands for probes and smoke execution. If neither variable is set, the script keeps the current default
behavior and uses the remote login shell's `python3`.

The smoke launcher also accepts a Python override so the selected interpreter is used consistently:

```text
TILEXR_VLLM_SMOKE_PYTHON=/path/to/python
```

The deployment script exports `TILEXR_VLLM_SMOKE_PYTHON` before invoking
`integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`.

## Components

### Remote Deployment Script

Responsibilities:

- Validate that at most one selected Python path is used for all package probes and smoke tests.
- Activate `TILEXR_VLLM_REMOTE_CONDA_ENV` when provided without relying on shell startup files.
- Print the selected Python executable, Python version, `sys.path` prefix, and package locations for `torch`,
  `torch_npu`, `vllm`, and `vllm_ascend`.
- Run a preflight that imports `torch` and `torch_npu`, checks NPU availability, checks device count, and checks the
  current stream handle fields.
- Keep TileXR CMake/CANN build behavior unchanged.
- Run `allgather int32` and `allgather fp16` shim smoke using the selected Python.
- Preserve each rank log under the remote scratch directory.

### Smoke Launcher

Responsibilities:

- Use `${TILEXR_VLLM_SMOKE_PYTHON:-python3}` for all rank processes.
- Log the interpreter path at launch time.
- Keep existing rank timeout and log tail behavior.
- Return nonzero when any rank fails, including import failures and TileXR ABI errors.

### Source-Level Tests

Responsibilities:

- Guard that the remote script exposes explicit Python/conda selection variables.
- Guard that package probes use the selected Python command instead of hard-coded `python3`.
- Guard that the smoke launcher uses `TILEXR_VLLM_SMOKE_PYTHON`.
- Guard that no remote script defaults to a specific remote host, scratch path, or mutates global shell startup files.

## Data Flow

```text
local worktree commit
  -> staging clone with local submodules
  -> rsync to TILEXR_VLLM_REMOTE_BASE/TileXR
  -> remote CANN env setup
  -> TileXR build/install
  -> selected Python preflight
  -> selected Python rank processes
  -> torch_npu NPU tensors
  -> tilexr_collectives ctypes shim
  -> libtile-comm.so + libtilexr-collectives.so
  -> per-rank correctness checks and logs
```

## Error Handling

- Missing conda activation script fails with a clear message when `TILEXR_VLLM_REMOTE_CONDA_ENV` is set.
- Missing selected Python fails before TileXR build starts.
- Missing `torch` or `torch_npu` fails before the multi-rank smoke.
- Missing `vllm` or `vllm_ascend` is recorded but does not fail this phase.
- Stream handle preflight fails if neither `npu_stream` nor `stream` is exposed by `torch.npu.current_stream()`.
- TileXR build, standalone collectives correctness, and Python smoke failures remain hard failures.
- The script prints the remote log path on both success and failure when possible.

## Test Matrix

Local:

- `python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py`
- `python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py`
- `bash -n tests/collectives/deploy_and_run_vllm_remote.sh integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
- `python3 -m py_compile integrations/vllm_ascend/smoke_collectives.py integrations/vllm_ascend/tilexr_collectives/*.py`
- `git diff --check`

Remote on `blue`:

```bash
TILEXR_VLLM_REMOTE=<ssh-target> \
TILEXR_VLLM_REMOTE_BASE=<remote-scratch-dir>/tilexr_vllm_collectives_$(date +%Y%m%d_%H%M%S) \
TILEXR_VLLM_REMOTE_ASCEND_HOME_PATH=<remote-cann-9.1.0-toolkit> \
TILEXR_VLLM_REMOTE_ASCEND_DRIVER_PATH=<remote-driver-shim-or-driver-path> \
TILEXR_VLLM_REMOTE_CMAKE_CCE_COMPILER=<remote-cann-9.1.0-toolkit>/bin/ccec \
TILEXR_VLLM_REMOTE_CONDA_ENV=tt4 \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

For the current `blue` host, fill those variables with the verified CANN Toolkit, driver shim, CCE compiler, and
writable scratch paths for that host. They must remain explicit run-time inputs, not script defaults.

Expected remote success lines:

```text
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=fp16
```

## Handoff To Inference-Path Work

Later vllm-ascend backend work may start only after Phase 2 proves:

- The selected remote Python environment can be recreated through documented variables.
- `torch_npu` imports and sees at least two NPUs.
- `torch.npu.current_stream()` exposes a usable stream handle.
- The TileXR shim passes `allgather int32` and `allgather fp16` on 2 ranks.
- Rank logs show no TileXR communicator lifecycle leaks or destroy failures.

After that, the next design should target a single opt-in vllm-ascend path, preferably `all_gather_async` or an
isolated `NPUCommunicator.all_to_all` path, while keeping the default vllm-ascend behavior unchanged.

## Risks

- The `tt4` environment uses `torch_npu 2.6.0`, while the local vllm-ascend source expects 2.9.0. This phase avoids
  inference-path claims to keep that version mismatch contained.
- `torch 2.6.0+cpu` paired with `torch_npu 2.6.0` can still run NPU operations, but the exact package provenance should
  be captured in logs for later compatibility decisions.
- Passing `aclrtStream` through `ctypes` may expose ordering issues that require a small C++ extension in a later phase.
- A successful 2-rank AllGather smoke does not prove MoE EP, all-to-all topology coverage, or full vLLM inference
  correctness.

## Success Criteria

Phase 2 is complete when:

- The updated script can select `tt4` on `blue` without modifying global remote state.
- TileXR builds and existing CTest/standalone collectives checks still pass remotely.
- `allgather int32` and `allgather fp16` shim smoke pass in the selected `torch_npu` environment.
- The command, selected Python, package versions, NPU summary, preflight output, and per-rank logs are recorded under
  the remote scratch directory.
