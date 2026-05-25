# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TileXR** (eXtreme Rendezvous for Asynchronous Tile Communication) is a distributed communication toolkit for Huawei Ascend NPU chips, built on the CANN stack. It provides tile-level asynchronous collective communication primitives optimized for distributed training.

- **CANN version:** 9.0.0-beta.1
- **Target OS:** Ubuntu 20.04 LTS (root user required for device access)
- **Supported chips:** Ascend 910B, 910A5, 310P3
- **Language:** C++14
- **NPU driver requirement:** ≥ 25.5.0 (`npu-smi info` to check)

## Repository Structure

```
comm/           # Core TileXR communication library → libtile-comm.so
mc2/            # Fused collective operators (AllGather+Add, AllGather+MatMul)
  all_gather_add/       # Fused AllGather + element-wise Add
  all_gather_matmul/    # Fused AllGather + MatMul (with op_api, tests/)
  common/               # Shared MC2 utilities and new_mc2_mm abstractions
op-simulator/   # Operator simulation and testing without physical hardware
include/        # Public C/C++ headers
3rdparty/       # Git submodules: hcomm, ops-transformer, opbase, spdlog, mki
```

## Environment Setup

Always source before building or running anything:

```bash
source common_env.sh
```

Sets `TILEXR_HOME`, `TILEXR_CANN_HOME`, `TILEXR_TEMP_HOME`, detects CPU arch, device count, and SOC name.

## Build

### Dependencies

TileXR requires the following dependencies:

- **CANN toolkit** (9.0.0-beta.1): Installed via `cann_download_install.sh`
- **hcomm**: Git submodule, built via `hcomm_build_install.sh`
- **opbase**: Git submodule, installed via `opbase_build_install.sh`
- **ops-transformer**: Git submodule, built via `ops_build_run.sh`
- **spdlog**: Git submodule (header-only logging)
- **mki**: Git submodule (matrix kernel interface)
- **shmem** (optional): Git submodule at `3rdparty/shmem`, provides UDMA capability for cross-node URMA communication. If unavailable or initialization fails, TileXR silently falls back to IPC-only mode.

### First-time setup (in order):

```bash
bash cann_download_install.sh       # Install CANN toolkit
bash hcomm_build_install.sh         # Build and install hcomm submodule
bash opbase_build_install.sh        # Install opbase submodule
bash ops_build_run.sh               # Build ops-transformer and run operators
```

### Core tile-comm library:

```bash
source common_env.sh
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
bash test_build.sh        # Build HCCL test suite
bash test_allreduce.sh    # Run AllReduce test via mpirun (multiple ranks)
bash ops_only_run.sh      # Run ops-transformer operators without rebuilding
```

Operator simulator:
```bash
cd op-simulator && bash run_test_ca.sh
```

`all_gather_matmul` has its own unit/system tests under `mc2/all_gather_matmul/tests/{ut,st}/`.

Logs: `bash plog_grep.sh ERROR` filters device logs.

## Architecture

### Core Communication (`comm/`)

- **`tilexr_comm.h/cpp`** — `TileXRComm` class: comm init, IPC shared memory (100 MB buffer + 2 MB flag space per rank), peer memory access between ranks.
- **`tilexr_internal.h/cpp`** — Internal helpers: `RegistKernel`, `LoadMTE`, `GetChipName`, `GetCoreNum`.
- **`comm_wrap.cpp`** — C wrapper exposing the C++ class via the public C API.
- **`tools/socket/sock_exchange.*`** — Socket-based rank-to-rank synchronization during setup.

### UDMA Transport (`comm/` + `include/tilexr_udma.h`)

- **UDMA 能力**：通过 shmem 库（`3rdparty/shmem`）提供跨节点 URMA 直驱通信
- **初始化**：`TileXRComm::InitUDMA()` 在 `Init()`/`InitThread()` 中透明调用，失败时静默降级
- **设备侧 API**：`include/tilexr_udma.h` 提供 `UDMAPutNbi`、`UDMAGetNbi`、`UDMAPutSignalNbi`、`UDMAQuiet`、原子操作等封装
- **CommArgs 扩展**：
  - `ExtraFlag::UDMA` (bit 10)：标识 UDMA 已初始化
  - `udmaInfoPtr`：指向设备 HBM 上的 `ACLSHMEMAIVUDMAInfo` 结构体（QP 上下文）
- **使用约定**：
  - 目标地址必须是通过 `aclshmem_malloc` 分配的对称内存
  - UB 暂存区最小 64 字节
  - `peerMems[]` 中的 IPC 地址不适用 UDMA 接口
- **降级行为**：UDMA 硬件不可用或 shmem init 失败时，`udmaInfoPtr` 保持 `nullptr`，现有集合通信路径不受影响

### Public API (`include/`)

- **`tilexr_api.h`** — 9 C functions for comm lifecycle (init, sync, teardown, buffer queries).
- **`tilexr_types.h`** — Enums: `ChipName`, `PhysicalLink`, `TileXRType`; constants (max rank size: 128, shared buffer: 204 MB + 4 MB flag buffer).
- **`tilexr_sync.h`** — `SyncCollectives` class: AICore kernel-side flag-based synchronization primitives. Two flag regions per rank: inner (intra-rank/card) and outer (inter-rank). Flags encode `(magic << 32) | value` to allow multi-round reuse without reset.
- **`comm_args.h`** — `CommArgs` struct with send matrices, peer memory pointers, and DFX debug info.

### Collective Operators (`mc2/`)

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
- `mc2/` and `comm/` can be built independently via their own `CMakeLists.txt`
- `mc2/build.sh` handles the mc2 standalone build
