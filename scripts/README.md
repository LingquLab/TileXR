# TileXR Scripts

This directory contains utility scripts for building, testing, and managing the TileXR project.

## Environment Setup

### `common_env.sh`
**Purpose**: Core environment configuration script that must be sourced before any build or test operations.

**Usage**:
```bash
source scripts/common_env.sh
```

**What it does**:
- Sets `TILEXR_HOME`, `TILEXR_CANN_HOME`, `TILEXR_TEMP_HOME`
- Detects CPU architecture (`TILEXR_OS_ARCH`)
- Detects SOC name and device count
- Configures CANN version (default: 9.1.0)
- Sets up PATH and LD_LIBRARY_PATH for CANN toolkit

**Environment variables**:
- `TILEXR_CANN_VER`: CANN version (9.1.0, 9.0.0-beta.1, 8.5.0)
- `TILEXR_SOC_NAME`: Detected chip name (Ascend910B, etc.)
- `TILEXR_HOME`: Repository root directory

### `common_util.sh`
**Purpose**: Utility functions used by other scripts.

**Functions**:
- `soc_name`: Detect NPU chip type
- `ops_name`: Get operator package name
- `device_count`: Count available NPU devices

## CANN Installation

### `cann_download_install.sh`
**Purpose**: Download and install CANN toolkit from Huawei OBS.

**Usage**:
```bash
bash scripts/cann_download_install.sh
```

**What it does**:
- Downloads CANN toolkit, kernels, and nnae packages
- Installs to `${TILEXR_CANN_HOME}`
- Configures environment variables

**Requirements**: Root user, internet connection

### `cann_local_install.sh`
**Purpose**: Install CANN from local packages (if already downloaded).

**Usage**:
```bash
bash scripts/cann_local_install.sh
```

## Dependency Building

### `download_open_source_deps.sh`
**Purpose**: Download fixed-version source archives used by `prepare.sh`.

**Usage**:
```bash
bash scripts/download_open_source_deps.sh
bash scripts/download_open_source_deps.sh --check
```

**What it does**:
- Downloads archives into `3rdparty/open_source/`
- Verifies every archive with SHA256
- Reuses valid existing archives

## Testing

### Pull-request CI

The CI operator runbook is [docs/CI.md](../docs/CI.md). Primary entrypoints are:

- `scripts/ci/host_checks.sh`: Ubuntu-compatible fast host validation.
- `scripts/ci/control/gate.py`: sealed `blue` build, queue, hardware, cleanup,
  and result controller; invoke it through the trusted workflow.
- `scripts/ci/provision/{account,cann,control,runner,verify}.sh`: idempotent
  `blue` provisioning and acceptance checks; each supports `--dry-run`.

### `test_build.sh`
**Purpose**: Build HCCL test suite.

**Usage**:
```bash
bash scripts/test_build.sh
```

### `test_allreduce.sh`
**Purpose**: Run AllReduce collective communication test.

**Usage**:
```bash
bash scripts/test_allreduce.sh
```

**What it does**:
- Launches multi-rank test via mpirun
- Tests AllReduce operation across NPU devices

## Utilities

### `plog_grep.sh`
**Purpose**: Filter and search device logs.

**Usage**:
```bash
bash scripts/plog_grep.sh <pattern>
```

**Examples**:
```bash
bash scripts/plog_grep.sh ERROR      # Find all errors
bash scripts/plog_grep.sh WARNING    # Find warnings
bash scripts/plog_grep.sh "AllGather" # Search for specific operation
```

**Log location**: `/var/log/npu/plog/`

### `device_connect.sh`
**Purpose**: Test NPU device connectivity.

**Usage**:
```bash
bash scripts/device_connect.sh
```

### `driver_fix.sh`
**Purpose**: Fix common NPU driver issues.

**Usage**:
```bash
bash scripts/driver_fix.sh
```

**What it does**:
- Resets NPU devices
- Clears driver state

**Requirements**: Root user

### `watch.sh`
**Purpose**: Monitor NPU device status in real-time.

**Usage**:
```bash
bash scripts/watch.sh
```

### `prepare.sh`
**Purpose**: Complete first-time setup script.

**Usage**:
```bash
bash scripts/prepare.sh
```

**What it does**:
- Installs repo-managed local build utilities
- Installs MPICH for multi-rank tests

**Use case**: First-time repository setup

## Typical Workflows

### First-time Setup
```bash
# 1. Install CANN
bash scripts/cann_download_install.sh

# 2. Build TileXR
source scripts/common_env.sh
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc) && make install
```

### Quick Setup (if packages already downloaded)
```bash
bash scripts/prepare.sh
```

### Development Workflow
```bash
# Source environment
source scripts/common_env.sh

# Run tests
bash scripts/test_allreduce.sh

# Check logs
bash scripts/plog_grep.sh ERROR
```

### Debugging
```bash
# Check device status
bash scripts/device_connect.sh

# Monitor devices
bash scripts/watch.sh

# Fix driver issues
bash scripts/driver_fix.sh

# Search logs
bash scripts/plog_grep.sh "your_search_term"
```

## Notes

- **Root user required**: Most scripts need root access for NPU device operations
- **Source common_env.sh**: Always source before building or testing
- **CANN version**: Default is 9.1.0, configurable via `TILEXR_CANN_VER`
- **Submodules**: Ensure `git submodule update --init --recursive` has been run
