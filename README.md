# TileXR

**TileXR** (eXtreme Rendezvous for Asynchronous Tile Communication) is a distributed communication toolkit for Huawei Ascend NPU chips, built on the CANN stack. It provides tile-level asynchronous collective communication primitives optimized for distributed training.

## Features

- **Tile-level Communication**: Asynchronous collective operations at AICore tile granularity
- **UDMA Support**: High-performance inter-chip communication over Huawei UnifiedBus DMA
- **Fused Operators**: MC2 operators combining communication with computation (AllGather+Add, AllGather+MatMul)
- **IPC Shared Memory**: Efficient intra-node communication with 100MB buffer per rank
- **Operator Simulator**: Test and validate operators without physical hardware

## System Requirements

- **OS**: Ubuntu 20.04 LTS
- **NPU Driver**: ≥ 25.5.0 (check with `npu-smi info`)
- **CANN**: 9.1.0 (default), 9.0.0-beta.1, or 8.5.0
- **Supported Chips**: Ascend 910B, 910A5, 310P3; UDMA demo support currently targets Ascend950
- **User**: Root access required for NPU device operations

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

### 2. First-Time Setup

```bash
# Complete automated setup (CANN + dependencies)
bash scripts/prepare.sh
```

Or step-by-step:

```bash
# Install CANN toolkit
bash scripts/cann_download_install.sh

# Build hcomm
bash scripts/hcomm_build_install.sh

# Build and run ops-transformer
bash scripts/ops_build_run.sh
```

### 3. Build TileXR

```bash
source scripts/common_env.sh
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) && make install
```

Output: `install/lib/libtile-comm.so`

### 4. Run Tests

```bash
# Build test suite
bash scripts/test_build.sh

# Run AllReduce test
bash scripts/test_allreduce.sh

# Check logs
bash scripts/plog_grep.sh ERROR
```

## Repository Structure

```
TileXR/
├── src/
│   ├── comm/           # Core communication library → libtile-comm.so
│   ├── mc2/            # Fused collective operators (AllGather+Add, AllGather+MatMul)
│   └── include/        # Public C/C++ headers
├── op-simulator/       # Operator simulation without hardware
├── tests/              # Unit and integration test suites
├── scripts/            # Build and utility scripts (see scripts/README.md)
├── 3rdparty/           # Git submodules (hcomm, ops-transformer, spdlog, mki, shmem)
└── docs/               # Documentation (UDMA, CANN migration, etc.)
```

## Dependencies

| Component | Version | Purpose |
|-----------|---------|---------|
| [CANN](https://www.hiascend.com/software/cann) | 9.1.0 | Ascend Computing Architecture Neural Network toolkit |
| [hcomm](https://gitcode.com/cann/hcomm) | 9.0.0-beta.1 | HCCL communication library |
| [ops-transformer](https://gitcode.com/cann/ops-transformer) | 9.0.0-beta.1 | Operator compilation framework |
| [spdlog](https://github.com/gabime/spdlog) | submodule | Fast C++ logging library |
| [mki](https://gitcode.com/cann/mki) | submodule | Matrix kernel interface |
| [shmem](3rdparty/shmem) | submodule | Transitional low-level communication dependency |

## Documentation

- **[scripts/README.md](scripts/README.md)**: Complete script reference and workflows
- **[docs/CANN_VERSION_MIGRATION.md](docs/CANN_VERSION_MIGRATION.md)**: CANN version compatibility
- **[CLAUDE.md](CLAUDE.md)**: Project instructions for AI assistants
- **[BUILD_VERIFICATION.md](BUILD_VERIFICATION.md)**: Build verification checklist

## Common Commands

```bash
# Source environment (required before any operation)
source scripts/common_env.sh

# Rebuild operators after changes
bash scripts/ops_only_run.sh

# Check device status
bash scripts/device_connect.sh

# Monitor NPU devices
bash scripts/watch.sh

# Search logs
bash scripts/plog_grep.sh "search_term"

# Fix driver issues
bash scripts/driver_fix.sh
```

## UDMA Support

TileXR integrates UDMA (UnifiedBus DMA) for high-performance inter-chip communication on Huawei Ascend systems. UDMA is based on Huawei's UnifiedBus interconnect and is exposed through TileXR as an optional device-side transport for registered device memory.

- **Automatic fallback**: Gracefully degrades to IPC/MTE if UDMA unavailable
- **Registered memory API**: Host code registers ordinary device memory with `TileXRUDMARegister`
- **Device-side API**: `include/tilexr_udma.h` provides `UDMAPutNbi`, `UDMAGetNbi`, and `UDMAPutSignalNbi`
- **Build target**: Current UDMA demo kernel targets `Ascend950`

## Operator Simulator

Test AICore kernels without physical hardware:

```bash
cd op-simulator
bash compile_and_run.sh
```

Use `base_test.cpp` and `test_template.cpp` as templates for new operator tests.

## Troubleshooting

**Driver issues**:
```bash
bash scripts/driver_fix.sh
```

**Build failures**:
- Ensure `git submodule update --init --recursive` has been run
- Check CANN version: `source scripts/common_env.sh && echo $TILEXR_CANN_VER`
- Verify driver: `npu-smi info`

**Log analysis**:
```bash
bash scripts/plog_grep.sh ERROR
bash scripts/plog_grep.sh WARNING
```

## License

Copyright (c) 2025 Huawei Technologies Co., Ltd.

This program is free software, you can redistribute it and/or modify it under the terms and conditions of CANN Open Software License Agreement Version 2.0.

See [LICENSE](LICENSE) for details.
