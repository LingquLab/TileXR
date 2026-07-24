# AGENTS.md

This file provides guidance to coding agents working in this repository. Reusable skills are maintained separately in [LingquLab/skills](https://github.com/LingquLab/skills).

## AI Skills

If `ascendc-development` is unavailable, tell the user to run these commands and start a new Codex task. Do not install plugins without permission.

```bash
codex plugin marketplace add https://github.com/LingquLab/skills.git
codex plugin add ascendc-development@lingqulab
```

## Project Overview

**TileXR** (eXtreme Rendezvous for Asynchronous Tile Communication) is a data-centric asynchronous communication runtime for Huawei Ascend NPU chips, built on the CANN stack. It provides tile-level synchronization, standalone collectives and EP communication, and a registered-memory UDMA prototype for A5 / Ascend950 hardware.

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
  include/        # Public C/C++ headers
op-simulator/     # Operator simulation and testing without physical hardware
tests/            # Test suites (UDMA, integration tests)
scripts/          # Build and utility scripts (see scripts/README.md)
3rdparty/         # Git submodule: spdlog
reference/        # Ignored reference-only source downloaded on demand, including hcomm and ops-transformer
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
- **spdlog**: Git submodule (header-only logging)
- **hcomm / ops-transformer / shmem** (reference-only): Download on demand with `reference/download_cann_repos.sh` into ignored directories under `reference/` for upstream comparison. Current TileXR libraries do not include or link them.

### Quick setup:

```bash
bash scripts/prepare.sh  # Automated CANN + dependencies setup
```

### Manual setup (step-by-step):

```bash
bash scripts/cann_download_install.sh       # Install CANN toolkit
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
```

Operator simulator:
```bash
cd op-simulator && bash run_test_ca.sh
```

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
- **No shmem dependency**: current `src/comm` sources must not include or link shmem.

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

### Operator Simulator (`op-simulator/`)

Functional and performance simulation of AICore kernels without physical hardware. Use `base_test.cpp` and `test_template.cpp` as templates for new operator tests.

## Key Notes

- **Git submodules** must be initialized: `git submodule update --init --recursive`
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

- `reference/shmem/` is an ignored reference-only checkout created by `bash reference/download_cann_repos.sh shmem` when needed.
- Current `tile-comm` does not link `libshmem.so` or `libaclshmem.so`.
- Do not add shmem includes to `src/comm` unless the architecture is intentionally changed.
- See [docs/UDMA_INTEGRATION_SUMMARY.md](docs/UDMA_INTEGRATION_SUMMARY.md) for current UDMA architecture notes and historical shmem context.
