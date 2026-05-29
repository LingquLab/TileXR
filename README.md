# TileXR

**TileXR** (eXtreme Rendezvous for Asynchronous Tile Communication) is a data-centric asynchronous communication runtime for Huawei Ascend NPUs. It moves communication control from coarse BSP-style kernel phases toward tile-level, AICore-driven rendezvous: data readiness, transport choice, and synchronization become explicit runtime state instead of a fixed all-ranks barrier.

The project currently contains a core communication library, MC2 fused collective operators, a registered-memory UDMA prototype for A5 / Ascend950 hardware, and simulator/test infrastructure for Ascend C kernels.

## Design Direction

TileXR is designed around three ideas from the current architecture deck:

- **Tile as the unit of progress**: split large BSP communication phases into smaller data tiles that can be produced, transferred, synchronized, and consumed independently.
- **AICore-driven asynchronous rendezvous**: let device code observe data readiness and runtime state, then advance communication without repeatedly returning to host scheduling.
- **Dynamic communication semantics**: choose among IPC/MTE, direct-drive UDMA/RDMA-style paths, notify/data-as-flag synchronization, and future offload paths according to data size, link state, peer readiness, and resource pressure.

The current codebase implements the base communication runtime, flag-based synchronization, MC2 examples, and an A5 UDMA registered-memory path. Broader dynamic scheduling, CMO best-effort scheduling, and CCU offload are design targets and should be treated as roadmap unless a specific implementation file says otherwise.

## Features

- **Core communication runtime**: `libtile-comm.so` initializes ranks, shared buffers, peer memory mappings, socket exchange, device `CommArgs`, and DFX state. It does not depend on hcomm, HCCL, shmem, or ops-transformer.
- **Tile-level synchronization**: device-side flag regions and magic values support reusable fine-grained synchronization rounds.
- **MC2 fused operators**: AllGather+Add and AllGather+MatMul examples under `src/mc2/`.
- **Registered-memory UDMA path**: host code registers ordinary `aclrtMalloc` device memory with `TileXRUDMARegister`; device kernels use `tilexr_udma.h` wrappers for put/get/signal.
- **Self-contained TileXR communication library**: `src/comm` uses TileXR-owned types and HCCP/RA runtime integration; it does not include or link hcomm, HCCL, or shmem.
- **Operator simulator**: `op-simulator/` supports functional/performance simulation for selected AICore kernels without physical hardware.

## System Requirements

- **OS**: Ubuntu 20.04 LTS
- **User**: root access is typically required for NPU device operations
- **NPU driver**: 25.5.0 or later, check with `npu-smi info`
- **CANN**: current build scripts and CMake are aligned to CANN 9.1.0
- **Core supported chips**: Ascend 910B, 910A5, 310P3
- **UDMA runtime validation target**: A5 / Ascend950 / 950 only

UDMA builds or smoke tests on 910B, 310P, or other non-A5 devices are not valid UDMA data-plane validation.

### System Dependencies

```bash
apt install -y build-essential git git-lfs rdma-core kmod net-tools \
               libssl-dev libz-dev libeigen3-dev python3 python3-pip
```

## Quick Start

### 1. Clone Repository

```bash
git clone --recursive https://gitcode.com/LingquLab/TileXR.git
cd TileXR
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

### 2. Prepare Environment

```bash
source scripts/common_env.sh
```

`scripts/common_env.sh` sets `TILEXR_HOME`, `TILEXR_CANN_HOME`, `TILEXR_TEMP_HOME`, architecture, SOC name, and CANN paths.

For first-time setup of local utilities and optional operator dependencies:

```bash
bash scripts/prepare.sh
```

For the full optional MC2/operator stack, also build hcomm and ops-transformer:

```bash
bash scripts/cann_download_install.sh
bash scripts/hcomm_build_install.sh
bash scripts/ops_build_run.sh
```

Only building `libtile-comm.so` does not require `hcomm_build_install.sh` or `ops_build_run.sh`.

### 3. Build Core Runtime

```bash
source scripts/common_env.sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build -j"$(nproc)"
cmake --install build
```

Expected output:

```text
install/lib/libtile-comm.so
```

### 4. Run Basic Tests

```bash
bash scripts/test_build.sh
bash scripts/test_allreduce.sh
bash scripts/plog_grep.sh ERROR
```

## Repository Structure

```text
TileXR/
|-- src/
|   |-- comm/                 # Core communication runtime
|   |   `-- udma/             # TileXR-owned HCCP/RA UDMA transport
|   |-- include/              # Public C/C++ and device headers
|   `-- mc2/                  # Fused collective operators
|       |-- all_gather_add/
|       |-- all_gather_matmul/
|       `-- common/
|-- op-simulator/             # Ascend C kernel simulation
|-- tests/                    # Integration and UDMA tests
|   `-- udma/
|-- scripts/                  # Build, setup, test, and utility scripts
|-- 3rdparty/                 # spdlog, mki, plus optional hcomm, ops-transformer, shmem
`-- docs/                     # Design, migration, and validation notes
```

## Architecture

### Core Runtime

`src/comm/` builds `libtile-comm.so` and exposes the public API in `src/include/tilexr_api.h`. This library is intentionally independent of hcomm, HCCL, shmem, and ops-transformer. It uses CANN runtime/ACL/driver APIs plus TileXR-owned communication metadata and datatypes.

Important host-side entry points:

- `TileXRGetUniqueId`
- `TileXRCommInitRankLocal`
- `TileXRCommInitRank`
- `TileXRCommInitRankWithDomain`
- `TileXRGetCommArgsDev`
- `TileXRGetCommArgsHost`
- `TileXRCommDestroy`

The runtime allocates shared IPC buffers, exchanges peer mappings, uploads `CommArgs` to device memory, and records topology/capability flags in `CommArgs::extraFlag`.

### Device Synchronization

`src/include/tilexr_sync.h` provides device-side flag synchronization. Flags use magic values so multiple rounds can reuse the same flag memory without a full reset.

### UDMA Registered Memory

The current UDMA path is TileXR-owned:

- `TileXRComm::InitUDMA()` tries to initialize UDMA for multi-rank communicators.
- `src/comm/udma/tilexr_hccp_loader.*` dynamically loads CANN HCCP/RA runtime libraries such as `libra.so` and `libtsdclient.so`.
- `src/comm/udma/tilexr_udma_transport.*` creates contexts, queues, route metadata, and a device-side `TileXR::UDMAInfo` image.
- `TileXRUDMARegister` registers ordinary device memory and exchanges remote region metadata.
- `CommArgs::udmaInfoPtr` and `CommArgs::udmaRegistryPtr` make queue and registered-memory metadata visible to kernels.
- `src/include/tilexr_udma.h` provides `UDMAPutNbi`, `UDMAGetNbi`, `UDMAPutSignalNbi`, and `UDMAQuiet`.

If UDMA is unavailable, communicator initialization continues without setting `ExtraFlag::UDMA`. UDMA-specific registration or demo paths then report that UDMA is unavailable.

### MC2 Operators

`src/mc2/` contains fused communication+compute examples following the ops-transformer host/tiling/kernel split:

- `all_gather_add`: example AllGather plus element-wise Add, fixed shape and rank-size constraints.
- `all_gather_matmul`: AllGather plus MatMul with aclnn API, graph integration, and tests.
- `common`: shared MC2 tiling, topology, HCCL, and matrix multiplication utilities.

## Dependencies

| Component | Version / Source | Purpose |
| --- | --- | --- |
| CANN | 9.1.0 | Required for `libtile-comm.so`: Ascend ACL/runtime/driver headers and libraries |
| spdlog | submodule | Header-only logging dependency |
| mki | submodule | Matrix kernel interface and utilities |

Optional components:

| Component | Version / Source | Used by | Notes |
| --- | --- | --- | --- |
| hcomm / HCCL | submodule / CANN communication stack | MC2 fused-operator examples and HCCL tests | Not included or linked by `src/comm` / `libtile-comm.so` |
| ops-transformer | submodule | `src/mc2` operator build, packaging, and run scripts | Not needed when only compiling `libtile-comm.so` |
| shmem | submodule, reference/optional | Historical UDMA experiments and comparison examples | Not included or linked by current `src/comm` |

## UDMA Validation

Use the dedicated UDMA guides when validating A5 / Ascend950 / 950 hardware:

```bash
cd tests/udma
bash build.sh
./install/bin/test_tilexr_no_shmem_dependency
./install/bin/test_tilexr_udma_transport_layout
./install/bin/test_tilexr_udma_registry
```

Run data-plane demos only on A5 / Ascend950 / 950:

```bash
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

See:

- [tests/udma/README.md](tests/udma/README.md)
- [tests/udma/QUICKSTART.md](tests/udma/QUICKSTART.md)
- [tests/udma/demo/ASCEND_VERIFICATION.md](tests/udma/demo/ASCEND_VERIFICATION.md)

## Operator Simulator

```bash
cd op-simulator
bash compile_and_run.sh
```

Use `op-simulator/src/base_test.cpp` and `op-simulator/test_template.cpp` as templates for new operator simulations.

## Common Commands

```bash
source scripts/common_env.sh
bash scripts/ops_only_run.sh
bash scripts/device_connect.sh
bash scripts/watch.sh
bash scripts/plog_grep.sh "search_term"
bash scripts/driver_fix.sh
```

## Documentation

- [scripts/README.md](scripts/README.md): script reference and workflows
- [docs/BUILD_VERIFICATION.md](docs/BUILD_VERIFICATION.md): current build and verification checklist
- [docs/UDMA_INTEGRATION_SUMMARY.md](docs/UDMA_INTEGRATION_SUMMARY.md): current UDMA architecture summary
- [docs/SHMEM_INTEGRATION.md](docs/SHMEM_INTEGRATION.md): shmem status and historical notes
- [docs/CANN_VERSION_MIGRATION.md](docs/CANN_VERSION_MIGRATION.md): CANN 9.1.0 migration notes
- [CLAUDE.md](CLAUDE.md): repository guidance for AI coding agents

## Troubleshooting

Driver or device issues:

```bash
bash scripts/driver_fix.sh
npu-smi info
```

Build failures:

- Run `git submodule update --init --recursive`.
- Run `source scripts/common_env.sh` before CMake or scripts.
- Check `ASCEND_HOME_PATH`, `TILEXR_CANN_VER`, and CANN 9.1.0 include/library layout.
- Confirm `install/lib/libtile-comm.so` links only to the expected CANN runtime/driver libraries and does not require hcomm, HCCL, shmem, or ops-transformer.

Log analysis:

```bash
bash scripts/plog_grep.sh ERROR
bash scripts/plog_grep.sh WARNING
```

## License

Copyright (c) 2025 Huawei Technologies Co., Ltd.

This program is free software. You may redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0.

See the repository license notice for details.
