# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TileXR** (eXtended Rendezvous for Asynchronous Tile Communication) is a distributed communication toolkit for Huawei Ascend NPU chips, built on the CANN (Computer Architecture for Neural Networks) stack. It provides tile-level asynchronous collective communication primitives optimized for distributed training workloads.

- **CANN version:** 9.0.0-beta.1
- **Target OS:** Ubuntu 20.04 LTS (root user context required for device access)
- **Supported chips:** Ascend 910B, 910A5, 310P3, and others
- **Language:** C++14

## Repository Structure

```
comm/           # Core TileXR communication library (tile-comm shared lib)
mc2/            # Fused collective operators (e.g., AllGather+Add)
op-simulator/   # Operator simulation and testing framework
include/        # Public C/C++ headers (tilexr_api.h, tilexr_types.h, comm_args.h)
3rdparty/       # Git submodules: hcomm, ops-transformer, opbase, spdlog, mki
```

## Environment Setup

Always source the environment script before building or running anything:

```bash
source common_env.sh
```

This sets `TILEXR_HOME`, `TILEXR_CANN_HOME`, `TILEXR_TEMP_HOME`, detects CPU arch, device count, and SOC name.

## Build

### Full first-time setup (in order):

```bash
bash cann_download_install.sh       # Install CANN toolkit (9.0.0-beta.1)
bash hcomm_build_install.sh         # Build and install hcomm submodule
bash opbase_build_install.sh        # Install opbase submodule
bash ops_build_run.sh               # Build ops-transformer and run operators
```

### Build the core tile-comm library:

```bash
source common_env.sh
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc)
make install
```

The output is `libtile-comm.so` installed to `install/lib/`.

### Build the operator simulator:

```bash
cd op-simulator
bash compile_and_run.sh
```

## Running Tests

```bash
bash test_build.sh       # Build HCCL test suite
bash test_allreduce.sh   # Run AllReduce test via mpirun (multiple ranks)
bash ops_only_run.sh     # Run ops-transformer operators without rebuilding
```

For operator simulator tests:
```bash
cd op-simulator
bash run_test_ca.sh      # Run with pre-compiled operators
```

## Architecture

### Core Communication (`comm/`)

- **`tilexr_comm.h/cpp`** — `TileXRComm` class managing comm initialization, IPC shared memory (100 MB buffer + 2 MB flag space per rank), and peer memory access between ranks.
- **`comm_wrap.cpp`** — C wrapper exposing the C++ class via the public C API.
- **`tools/socket/sock_exchange.*`** — Socket-based rank-to-rank synchronization during setup.

### Public API (`include/`)

- **`tilexr_api.h`** — 9 C functions for comm lifecycle (init, sync, teardown, buffer queries).
- **`tilexr_types.h`** — Enums: `ChipName`, `PhysicalLink`, `TileXRType`; key constants (max rank size: 128, shared buffer: 204 MB + 4 MB flag buffer).
- **`comm_args.h`** — `CommArgs` struct with send matrices, peer memory pointers, and DFX debug info.

### Collective Operators (`mc2/`)

Each operator follows the ops-transformer two-phase calling convention:
1. **Host side:** operator definition (`_def.cpp`), tiling (`_tiling.cpp`), and aclnn API (`aclnn_*.h/cpp`).
2. **Kernel side:** AICore kernel implementation (`op_kernel/*.cpp`).

The `all_gather_add` operator is the primary example — it fuses AllGather and element-wise Add in a single kernel pass with fixed shape constraints: input `a(240,256)`, output `b(480,256)`, `rank_size=2`, `FLOAT16` only.

### Operator Simulator (`op-simulator/`)

Provides functional and performance simulation of AICore kernels without physical hardware. Uses `base_test.cpp` and `test_template.cpp` as templates for new operator tests.

## Key Notes

- **Git submodules** (`3rdparty/hcomm`, `3rdparty/ops-transformer`, `3rdparty/opbase`) must be initialized: `git submodule update --init --recursive`.
- Builds require NPU driver version ≥ 25.5.0.
- Log filtering: `bash plog_grep.sh` filters error/warning logs from device logs.
- The `mc2/` and `comm/` modules can be built independently via their own `CMakeLists.txt`.
