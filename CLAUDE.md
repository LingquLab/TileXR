# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TileXR** (eXtreme Rendezvous for Asynchronous Tile Communication) is a data-centric asynchronous communication runtime for Huawei Ascend NPU chips, built on the CANN stack. It provides tile-level synchronization, MC2 fused collective examples, and a registered-memory UDMA prototype for A5 / Ascend950 hardware.

- **CANN version:** 9.1.0
- **Target OS:** Ubuntu 20.04 LTS (root user required for device access)
- **Supported chips:** Ascend 910B, 910A5, 310P3; UDMA data-plane validation currently targets A5 / Ascend950 / 950 only
- **Language:** C++14
- **NPU driver requirement:** ≥ 25.5.0 (`npu-smi info` to check)

## Repository Structure

```
src/
  comm/           # Core TileXR communication library -> libtile-comm.so
    udma/         # TileXR-owned HCCP/RA UDMA transport
  mc2/            # Fused collective operators (AllGather+Add, AllGather+MatMul)
    all_gather_add/       # Fused AllGather + element-wise Add
    all_gather_matmul/    # Fused AllGather + MatMul (with op_api, tests/)
    common/               # Shared MC2 utilities and new_mc2_mm abstractions
  include/        # Public C/C++ headers
op-simulator/     # Operator simulation and testing without physical hardware
tests/            # Test suites (UDMA, integration tests)
scripts/          # Build and utility scripts (see scripts/README.md)
3rdparty/         # Git submodules: hcomm, ops-transformer, spdlog, mki, shmem
docs/             # Documentation (UDMA, CANN migration, etc.)
```

## Environment Setup

Always source before building or running anything:

```bash
source scripts/common_env.sh
```

Sets `TILEXR_HOME`, `TILEXR_CANN_HOME`, `TILEXR_TEMP_HOME`, detects CPU arch, device count, and SOC name.

See [scripts/README.md](scripts/README.md) for complete script documentation and workflows.

## Build

### Dependencies

TileXR requires the following dependencies:

- **CANN toolkit** (9.1.0): Installed via `scripts/cann_download_install.sh`
- **hcomm**: Git submodule, built via `scripts/hcomm_build_install.sh`
- **ops-transformer**: Git submodule, built via `scripts/ops_build_run.sh`
- **spdlog**: Git submodule (header-only logging)
- **mki**: Git submodule (matrix kernel interface)
- **shmem** (optional/reference): Git submodule at `3rdparty/shmem`, kept for historical UDMA experiments and comparison. Current `src/comm` does not include or link shmem.

### Quick setup:

```bash
bash scripts/prepare.sh  # Automated CANN + dependencies setup
```

### Manual setup (step-by-step):

```bash
bash scripts/cann_download_install.sh       # Install CANN toolkit
bash scripts/hcomm_build_install.sh         # Build and install hcomm submodule
bash scripts/ops_build_run.sh               # Build ops-transformer and run operators
```

### Core tile-comm library:

```bash
source scripts/common_env.sh
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) && make install
# Output: install/lib/libtile-comm.so
```

### Operator simulator:

```bash
cd op-simulator && bash compile_and_run.sh
```

## Running Tests

```bash
bash scripts/test_build.sh        # Build HCCL test suite
bash scripts/test_allreduce.sh    # Run AllReduce test via mpirun (multiple ranks)
bash scripts/ops_only_run.sh      # Run ops-transformer operators without rebuilding
```

Operator simulator:
```bash
cd op-simulator && bash run_test_ca.sh
```

`all_gather_matmul` has its own unit/system tests under `src/mc2/all_gather_matmul/tests/{ut,st}/`.

Logs: `bash scripts/plog_grep.sh ERROR` filters device logs.

## Architecture

### Core Communication (`src/comm/`)

- **`tilexr_comm.h/cpp`** — `TileXRComm` class: comm init, IPC shared memory (100 MB buffer + 2 MB flag space per rank), peer memory access between ranks, device `CommArgs`, and optional TileXR-owned UDMA initialization.
- **`tilexr_internal.h/cpp`** — Internal helpers: `RegistKernel`, `LoadMTE`, `GetChipName`, `GetCoreNum`.
- **`comm_wrap.cpp`** — C wrapper exposing the C++ class via the public C API.
- **`tools/socket/sock_exchange.*`** — Socket-based rank-to-rank synchronization during setup.

### UDMA Integration (`src/comm/udma/`)

TileXR integrates UDMA (UnifiedBus DMA) for registered-memory communication on A5 / Ascend950-class hardware:

- **TileXR-owned transport**: `TileXRUDMATransport` dynamically loads CANN/HCCP runtime libraries and creates RA contexts, queues, and memory registration metadata.
- **Device-side pointer**: `CommArgs::udmaInfoPtr` points to a device-side `TileXR::UDMAInfo` image built by TileXR.
- **Registered memory**: host code registers ordinary `aclrtMalloc` device memory through `TileXRUDMARegister`; `CommArgs::udmaRegistryPtr` exposes per-rank registered regions to kernels.
- **Graceful capability detection**: if UDMA is unavailable, communicator initialization continues without setting `ExtraFlag::UDMA`.
- **No shmem dependency**: current `src/comm` sources must not include or link shmem; `tests/udma/unit/test_tilexr_no_shmem_dependency.cpp` guards this.

### UDMA Transport (`src/comm/` + `src/include/tilexr_udma.h`)

- **UDMA capability**: TileXR-owned HCCP/RA transport provides device-visible UDMA queue metadata on supported A5 / Ascend950 systems.
- **Initialization**: `TileXRComm::InitUDMA()` is invoked during normal comm init for multi-rank process mode. `TileXRUDMARegister` is not supported in `InitThread` mode in the current implementation.
- **Device API**: `include/tilexr_udma.h` provides `UDMAPutNbi`, `UDMAGetNbi`, `UDMAPutSignalNbi`, and `UDMAQuiet`.
- **CommArgs 扩展**：
  - `ExtraFlag::UDMA` (bit 10)：标识 UDMA 已初始化
  - `udmaInfoPtr`：指向设备 HBM 上的 `TileXR::UDMAInfo` 结构体（QP 上下文）
  - `udmaRegistryPtr`：指向设备 HBM 上的 `TileXRUDMARegistry`
- **使用约定**：
  - 目标地址必须属于 `TileXRUDMARegister` 注册的普通 device memory
  - `peerMems[]` 中的 IPC 地址不适用 UDMA 接口
- **降级行为**：UDMA 硬件不可用或 HCCP/RA 初始化失败时，`udmaInfoPtr` 保持 `nullptr`，现有集合通信路径不受影响

### Public API (`src/include/`)

- **`tilexr_api.h`** — C API for comm lifecycle, `CommArgs` queries, DFX logging, and UDMA memory registration.
- **`tilexr_types.h`** — Enums: `ChipName`, `PhysicalLink`, `TileXRType`; constants (max rank size: 128, shared buffer: 204 MB + 4 MB flag buffer).
- **`tilexr_sync.h`** — `SyncCollectives` class: AICore kernel-side flag-based synchronization primitives. Two flag regions per rank: inner (intra-rank/card) and outer (inter-rank). Flags encode `(magic << 32) | value` to allow multi-round reuse without reset.
- **`comm_args.h`** — `CommArgs` struct with send matrices, peer memory pointers, and DFX debug info.

### Collective Operators (`src/mc2/`)

Each operator follows the ops-transformer two-phase calling convention:
1. **Host side:** operator definition (`_def.cpp`), tiling (`_tiling.cpp`), aclnn API (`aclnn_*.h/cpp`), and for `all_gather_matmul` an additional `op_api/` and `op_graph/` layer.
2. **Kernel side:** AICore kernel implementation (`op_kernel/*.cpp`).

Key operators:
- **`all_gather_add`** — Fuses AllGather + element-wise Add. Fixed shapes: input `a(240,256)`, output `b(480,256)`, `rank_size=2`, FLOAT16 only.
- **`all_gather_matmul`** — Fuses AllGather + MatMul. Has full aclnn API (`op_api/`), graph integration (`op_graph/`), and test suite (`tests/`).
- **`common/`** — Shared MC2 infrastructure including `new_mc2_mm` matrix-multiply primitives.

### Operator Simulator (`op-simulator/`)

Functional and performance simulation of AICore kernels without physical hardware. Use `base_test.cpp` and `test_template.cpp` as templates for new operator tests.

## Key Notes

- **Git submodules** must be initialized: `git submodule update --init --recursive`
- `src/mc2/` and `src/comm/` can be built independently via their own `CMakeLists.txt`
- All build and utility scripts are in `scripts/` directory (see `scripts/README.md`)

### CANN Version Compatibility

**Current version**: CANN 9.1.0 (cann-9.1.0)

**Important changes from CANN 9.0.0**:
- Directory structure: headers now in `${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/` (added root `pkg_inc/`)
- Library location: `ascend_hal` moved to `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib/`
- AscendC API changes: Some SIMT-related APIs may have been removed or restructured

**Build requirements**:
```cmake
# Include paths must include pkg_inc root
include_directories(
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/
    ${ASCEND_HOME_PATH}/${ARCH}-linux/pkg_inc/runtime/
)

# Link directories must include devlib
target_link_directories(
    ${ASCEND_HOME_PATH}/${ARCH}-linux/devlib
)
```

**Runtime RPATH warning**:
- `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` is for link-time fallback only. Do not write it into runtime RPATH/RUNPATH.
- If `aarch64-linux/devlib` is present in RPATH, the process may load the stub `libascend_hal.so` from devlib instead of the real driver HAL, causing `aclInit` to fail, for example with `500000` and log message `init soc version failed`.
- Runtime should load `libascend_hal.so` from the driver path, typically `/usr/local/Ascend/driver/lib64/driver`.

### shmem Integration Notes

The old shmem-backed UDMA proposal has been superseded by TileXR-owned UDMA transport under `src/comm/udma/`.

- `3rdparty/shmem` remains as a reference/experimental submodule.
- Current `tile-comm` does not link `libshmem.so` or `libaclshmem.so`.
- Do not add shmem includes to `src/comm` unless the architecture is intentionally changed.
- See [docs/SHMEM_INTEGRATION.md](docs/SHMEM_INTEGRATION.md) for historical context and current guardrails.
