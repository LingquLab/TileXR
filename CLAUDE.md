# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TileXR** (eXtreme Rendezvous for Asynchronous Tile Communication) is a data-centric asynchronous communication runtime for Huawei Ascend NPUs, built on the CANN stack. It moves communication control from coarse BSP-style kernel phases toward tile-level, AICore-driven rendezvous: data readiness, transport choice, and synchronization become explicit runtime state instead of a fixed all-ranks barrier.

- **CANN version:** 9.1.0 (build scripts and CMake are aligned to this)
- **Language:** C++14
- **Core supported chips:** Ascend 910B, 910A5
- **UDMA data-plane validation target:** A5 / Ascend950 / 950 only (smoke tests on 910B are *not* valid UDMA validation)
- **NPU driver requirement:** >= 25.1.rc1.xxx (`npu-smi info` to check)
- **User:** root or Ascend driver-group membership is typically required for CANN install and device access

`README.md` is the authoritative, most detailed reference (full validation recipes, profiling, diagrams). This file is the condensed agent-facing version — when they disagree, trust `README.md` and the code.

## Environment Setup

**Always source before building or running anything:**

```bash
source scripts/common_env.sh
```

Sets `TILEXR_HOME`, `TILEXR_CANN_HOME`, `TILEXR_TEMP_HOME`, detects CPU arch / device count / SOC name, and configures CANN paths. For non-root builds it can substitute readable driver headers from the repo-managed CANN install while still linking the system driver libraries.

If a readable CANN 9.1.0 install already exists, point at it instead of downloading:

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
source scripts/common_env.sh
```

See [scripts/README.md](scripts/README.md) for complete script documentation.

## Build

### Build options (CMake)

The top-level `CMakeLists.txt` always builds `src/comm` (`libtile-comm.so`). Everything else is opt-in:

| Option | Default | Adds |
| --- | --- | --- |
| `TILEXR_BUILD_COLLECTIVES` | OFF | `src/collectives` → `libtilexr-collectives.so` + `tilexr_collectives.h` |
| `TILEXR_BUILD_EP` | OFF | `src/ep` → `libtilexr-ep.so`, `libtilexr_ep_dispatch_kernel.so` |
| `TILEXR_BUILD_TESTS` | OFF | test targets (also enables CTest; `BUILD_TESTING` works too) |
| `TILEXR_BUILD_CHECKER` | OFF | `tools/checker` no-NPU checker |

### Core runtime (default)

```bash
source scripts/common_env.sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j"$(nproc)"
cmake --install build
# Output: install/lib*/libtile-comm.so
```

Building only `libtile-comm.so` does **not** require hcomm or ops-transformer.

### Optional collectives library

```bash
source scripts/common_env.sh
cmake -S . -B build-collectives \
  -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build-collectives -j"$(nproc)"
cmake --install build-collectives
```

### First-time dependency setup

```bash
bash scripts/cann_download_install.sh   # repo-managed CANN 9.1.0 toolkit
bash scripts/prepare.sh                 # local utilities + optional operator deps
# Optional MC2/operator stack (only if working on examples/mc2):
bash scripts/hcomm_build_install.sh
bash scripts/ops_build_run.sh
```

### MC2 operator examples & simulator

```bash
bash scripts/ops_build_run.sh           # build ops-transformer and run operators
bash scripts/ops_only_run.sh            # run operators without rebuilding
cd op-simulator && bash compile_and_run.sh
```

## Running Tests

CTest is the entry point for hardware-free checks; physical multi-NPU runs are manual.

```bash
# Collectives source/CLI smoke checks (no NPU)
ctest --test-dir build-collectives --output-on-failure

# Standalone EP checks (no NPU)
bash tests/ep/build.sh source-only
ctest --test-dir tests/ep/build --output-on-failure

# SDMA unit tests (hardware-free) against a chosen CANN install
bash tests/sdma/build.sh /path/to/cann
bash tests/sdma/run_tests.sh /path/to/cann

# UDMA layout/registry unit tests
cd tests/udma && bash build.sh
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry

# HCCL suite (legacy)
bash scripts/test_build.sh
bash scripts/test_allreduce.sh   # AllReduce via mpirun, multiple ranks
```

Hardware data-plane demos (run only on the validation target hardware) live under each component's `demo/` dir — e.g. `tests/udma/demo/`, `tests/sdma/demo/`, `tests/ep/demo/`. Manual multi-NPU collectives correctness/perf tools are under `tests/collectives/` (see its README for the full perf + profiling workflow).

Log filtering: `bash scripts/plog_grep.sh ERROR` (also `WARNING`, or any search term).

## Repository Structure

```
src/
  comm/           # Core runtime -> libtile-comm.so (always built)
    udma/         # TileXR-owned HCCP/RA UDMA transport
    sdma/         # On-card PTO SDMA local copy transport
    tools/socket/ # Socket-based rank-to-rank setup sync
  collectives/    # Optional TileXR collectives lib (host/ + kernels/)
  ep/             # Standalone EP dispatch MVP (common/ host/ kernels/)
  include/        # Public C/C++ and device headers
examples/mc2/     # Fused collective operator examples (via ops-transformer)
op-simulator/     # Ascend C kernel simulation (no hardware)
tests/            # checker collectives comm data_as_flag ep memory reference sdma udma
tools/checker/    # No-NPU static/behavioral checker
integrations/     # Downstream integrations (e.g. vllm_ascend)
scripts/          # Build, setup, test, utility scripts
3rdparty/         # spdlog (+ optional hcomm, ops-transformer submodules)
reference/        # Scripts for ignored reference-only checkouts
docs/             # Design, migration, validation notes; diagrams/
```

## Architecture

### Core runtime (`src/comm/`) — deliberately minimal dependencies

`libtile-comm.so` exposes the public API in `src/include/tilexr_api.h`. It is **intentionally independent of hcomm, HCCL, shmem, and ops-transformer** — it uses only CANN runtime/ACL/driver APIs plus TileXR-owned metadata/types. Do not add hcomm/HCCL/shmem includes to `src/comm` unless the architecture is intentionally changed.

Host-side entry points (`tilexr_api.h`), by role:
- **Lifecycle:** `TileXRGetUniqueId`, `TileXRCommInitRankLocal`, `TileXRCommInitRank`, `TileXRCommInitRankWithDomain`, `TileXRCommDestroy`.
- **CommArgs access:** `TileXRGetCommArgsHost` (host view), `TileXRGetCommArgsDev` (device pointer for kernels).
- **Sync rounds:** `TileXRCommNextMagic` hands out a fresh magic value so callers reuse flag memory across rounds.

Init allocates shared IPC buffers, exchanges peer mappings, uploads `CommArgs` to device HBM, and records topology/capability flags in `CommArgs::extraFlag`. Internal helpers (`RegistKernel`, `LoadMTE`, `GetChipName`, `GetCoreNum`) live in `tilexr_internal.*`; `comm_wrap.cpp` is the C wrapper over the C++ `TileXRComm` class.

### Device synchronization (`src/include/tilexr_sync.h`)

`SyncCollectives`: AICore kernel-side flag-based sync. Two flag regions per rank — inner (intra-rank/card) and outer (inter-rank). Flags encode `(magic << 32) | value` so multiple rounds reuse the same flag memory without a reset. The `data_as_flag` path (`tilexr_data_as_flag.h`, tested under `tests/data_as_flag/`) overlays readiness flags onto data blocks.

### Three interchangeable transports

Kernels select a data-plane transport by data size, link state, peer readiness, and capability flags. UDMA and SDMA are brought up **best-effort** — if either is unavailable, init continues without setting its `extraFlag` bit, and existing paths are unaffected.

1. **IPC / MTE** — same-host peer-memory windows via `CommArgs::peerMems[]`. Always available.
2. **UDMA** (`src/comm/udma/`, `tilexr_udma.h`) — registered remote memory on A5 / Ascend950. `TileXRComm::InitUDMA()` runs during multi-rank init; `tilexr_hccp_loader.*` dynamically loads CANN HCCP/RA libs (`libra.so`, `libtsdclient.so`); `tilexr_udma_transport.*` builds contexts/queues and a device-side `TileXR::UDMAInfo`. Register ordinary `aclrtMalloc` memory with `TileXRUDMARegister`; `CommArgs::udmaInfoPtr` / `udmaRegistryPtr` expose metadata to kernels (`ExtraFlag::UDMA`, bit 10). Device API: `UDMAPutNbi`, `UDMAGetNbi`, `UDMAPutSignalNbi`, `UDMAQuiet`. **UDMA targets only `TileXRUDMARegister`-registered memory — IPC addresses in `peerMems[]` are not valid UDMA targets.** Not supported in `InitThread` mode.
3. **SDMA** (`src/comm/sdma/`, `tilexr_sdma.h`) — local on-card GM-to-GM copy within a single device. **Disabled by default; enable with `TILEXR_ENABLE_SDMA=1`.** `TileXRComm::InitSDMA()` creates a PTO `SdmaWorkspaceManager`, stores its address in `CommArgs::sdmaWorkspacePtr`, sets `ExtraFlag::SDMA`. Host queries: `TileXRSDMAAvailable`, `TileXRGetSDMAWorkspaceDev` (workspace owned by `TileXRComm` — do not free). Device API: `SDMACopyNbi`, `SDMAWait` on raw same-device GM pointers (no registration/ownership checks). CANN 9.0.0/9.1.0 header diffs are isolated in `tilexr_sdma_compat.h`.

### Optional collectives (`src/collectives/`)

`libtilexr-collectives.so` (only when `TILEXR_BUILD_COLLECTIVES=ON`) layers standalone collectives on `libtile-comm.so`. The split is intentional: `libtile-comm.so` owns comm setup + infra API (`tilexr_api.h`); the collectives lib owns host validation, launch, embedded CCE kernel registration, and `tilexr_collectives.h`. Installing only the core runtime does *not* install `tilexr_collectives.h`. APIs: `TileXRAllGather` (validated multi-rank), `TileXRAllToAll` (equal per-peer counts; multi-rank only on `TOPO_910_93` topology — others return a param-check error). Single-rank loopback works for both.

### Standalone EP dispatch (`src/ep/`)

First TileXR-native MoE Expert-Parallelism dispatch (`TILEXR_BUILD_EP=ON`). `libtilexr-ep.so` exposes `TileXRMoeEpDispatch` (`tilexr_ep.h`); `libtilexr_ep_dispatch_kernel.so` holds the Ascend C dispatch/combine kernel. The MVP route uses `CommArgs::peerMems[]`, `TileXR::IPC_DATA_OFFSET`, and `SyncCollectives` for peer-memory communication — intentionally independent from `examples/mc2`, ops-transformer, shmem, and UDMA. A future route may add a UDMA backend while keeping peer-memory as fallback.

### MC2 operator examples (`examples/mc2/`)

Fused communication+compute **examples** (not core libraries), built via `scripts/ops_build_run.sh` following the ops-transformer host/tiling/kernel split:
1. Host side: `_def.cpp`, `_tiling.cpp`, `aclnn_*.h/cpp` (and `op_api/` + `op_graph/` for `all_gather_matmul`).
2. Kernel side: AICore impl in `op_kernel/*.cpp`.

- `all_gather_add` — AllGather + element-wise Add. Fixed shapes: input `a(240,256)`, output `b(480,256)`, `rank_size=2`, FLOAT16 only.
- `all_gather_matmul` — AllGather + MatMul with full aclnn API, graph integration, and `tests/{ut,st}/`.
- `common/` — shared MC2 tiling/topology/HCCL/matmul utilities (`new_mc2_mm`).

## Key Notes

- **Git submodules** must be initialized: `git submodule update --init --recursive`.
- `examples/mc2/` and `src/comm/` build independently via their own `CMakeLists.txt`.
- `AGENTS.md` directs agents to the Ascend C skills under `.claude/skills/` (`ascendc-api-best-practices`, `ascendc-code-review`, `ascendc-docs-search`, `ascendc-env-check`, `cann-env-setup`) — use them for Ascend C API usage, code review, and env checks.

### CANN 9.1.0 build layout

```cmake
# Headers live under pkg_inc root (added in 9.1.0):
include_directories(
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
)
# ascend_hal link-time fallback:
target_link_directories(${ASCEND_HOME_PATH}/${ARCH}-linux/devlib)
```

### Ascend C kernel launch rule

- Do not introduce or use Host wrappers that launch kernels with
  `kernel<<<blockDim, nullptr, stream>>>(...)` syntax.
- Host launch code must use `rtKernelLaunchWithFlagV2` with the function signature
  registered for the compiled kernel binary.
- When runtime reports an unknown or null kernel pointer, diagnose binary registration
  and function-signature identity. Do not work around it with a compiler launch wrapper.
- Source guards for new or modified kernel paths must reject `<<<...>>>` Host wrappers.

### RPATH warning (non-obvious, causes silent runtime failures)

`${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` is **link-time fallback only — never put it in runtime RPATH/RUNPATH.** It contains a stub `libascend_hal.so`; if it ends up on the runtime path the process loads the stub instead of the real driver HAL, and `aclInit` fails (e.g. error `500000`, log `init soc version failed`). Runtime must resolve `libascend_hal.so` from the driver path, typically `/usr/local/Ascend/driver/lib64/driver`. Confirm `install/lib/libtile-comm.so` links only the expected CANN runtime/driver libs (not hcomm/HCCL/shmem/ops-transformer).

## Troubleshooting

```bash
bash scripts/driver_fix.sh && npu-smi info   # driver/device issues
bash scripts/plog_grep.sh ERROR              # device log analysis
```

Build failures: re-run `git submodule update --init --recursive`, `source scripts/common_env.sh` before CMake, and check `ASCEND_HOME_PATH` / CANN 9.1.0 include+library layout. See [docs/CANN_VERSION_MIGRATION.md](docs/CANN_VERSION_MIGRATION.md), [docs/BUILD_VERIFICATION.md](docs/BUILD_VERIFICATION.md), [docs/UDMA_INTEGRATION_SUMMARY.md](docs/UDMA_INTEGRATION_SUMMARY.md), and [docs/SDMA_TRANSPORT.md](docs/SDMA_TRANSPORT.md).

<!-- karpathy-guidelines:start -->
# Karpathy Guidelines

## Think Before Coding

- State assumptions and tradeoffs before implementation.
- Stop and clarify genuinely ambiguous requirements.
- Define observable success criteria for every change.

## Simplicity First

- Implement the minimum code required by the request.
- Avoid speculative abstractions, options, and error handling.
- Prefer a smaller direct implementation when it is equally correct.

## Surgical Changes

- Touch only files and lines required by the task.
- Preserve existing style and unrelated user changes.
- Remove only code made obsolete by the current change.

## Goal-Driven Execution

- Convert requirements into verifiable tests or checks.
- For behavior changes, establish a failing check before implementation.
- Keep iterating until the defined checks pass, and report unverified layers explicitly.
<!-- karpathy-guidelines:end -->
