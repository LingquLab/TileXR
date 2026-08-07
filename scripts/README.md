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

### `run_moonep.sh`
**Purpose**: Run the MoonEP benchmark, reference, or correctness flow with a requested
logical rank count.

**Usage**:
```bash
bash scripts/run_moonep.sh --mode correctness --rank-size 4
```

The equivalent short form is `bash scripts/run_moonep.sh -m correctness -r 4`.

The MoonEP wrapper exports `TILEXR_ENABLE_UDMA=0` because these stages currently use
the same-node IPC path. This avoids optional UDMA initialization sharing HCCP state
with the Torch-NPU/HCCL reference flow.

All modes continue to default to `skewed-padding`. Reference and correctness runs can
opt into the hand-checkable `manual-small` case
(`S=2,K=2,E=4,H=2,Hf=2,B=1`, default `P=1`) with
`--case-id manual-small`.

Without a device option, the script selects physical NPUs starting at 0, so a
four-rank run uses devices `0,1,2,3`. Select a different set without configuring the
shell environment:

```bash
bash scripts/run_moonep.sh --mode reference --rank-size 4 \
  --case-id manual-small \
  --visible-devices 4,5,6,7 \
  --hccl-npu-socket-port-range 47200-47300
```

The HCCL NPU socket range defaults to `47000-47100` inside the script. The CLI values
take precedence; existing `ASCEND_RT_VISIBLE_DEVICES` and
`HCCL_NPU_SOCKET_PORT_RANGE` remain supported only as compatibility fallbacks.

`reference` and `correctness` runs save complete inputs and outputs for every MoonEP
stage by default and print the first eight values of each tensor. Override the preview
length with `--tensor-preview-elements COUNT`, or disable snapshots with
`--no-dump-stage-tensors`. `benchmark` runs never enable snapshots because device-to-CPU
copies would invalidate performance measurements.

Each binary `input.pt`/`output.pt` has a matching `input.txt`/`output.txt` that includes
the field path, shape, dtype, original device, and every value without ellipsis. JSON
files retain compact metadata and terminal previews.

Add `--generate-flowcharts` to a `reference` or `correctness` run to generate a
left-to-right diagram for all six stage boundaries:

```bash
bash scripts/run_moonep.sh --mode reference --rank-size 2 \
  --case-id manual-2rank-dedup-3 \
  --generate-flowcharts
```

The option is disabled by default and requires the default tensor snapshots. It reads
the reference snapshots even in correctness mode, then writes numbered Mermaid, SVG,
and 2x PNG files under `<result>/<case_id>/flowcharts/`, from `1_planning-*` through
`6_reduce-grad-*`. Run `bash scripts/prepare.sh` first to install the PyPI Mermaid CLI
and browser. Flowchart export is rejected in benchmark mode so rendering and snapshot
work cannot affect performance measurements.

The script supports npm Mermaid CLI with Puppeteer and the Python Mermaid CLI with
Playwright. The selected CLI must also have a browser for the host architecture; for
the Playwright variant, install it once with `python -m playwright install chromium`.
The script performs a minimal render before launching NPU work and fails with this
guidance when the browser is missing or incompatible.

Use the two-rank manual case to inspect Planner load migration from owner load `[8,0]`
to execution load `[4,4]`:

```bash
bash scripts/run_moonep.sh --mode reference --rank-size 2 \
  --case-id manual-2rank-imbalanced
```

Both ranks start with local `tokens_per_expert=[2,2,0,0]`. Expert 0 is migrated to
rank 1 and occupies its single prefetch slot.

Use the compact mixed case to verify Planner load migration and one primary plus
three duplicate offsets in the same run:

```bash
bash scripts/run_moonep.sh --mode reference --rank-size 2 \
  --case-id manual-2rank-dedup-3
```

Initial expert-owner loads are `[12,4]`, and Planner moves execution to `[8,8]`.
Rank 0's first token routes to experts `[4,5,6,7]` on rank 1, producing a dedup group
with duplicate count 3.

The script sources `common_env.sh`, uses Conda environment `ai_moe_test`, and writes
results to a timestamped directory under `/tmp`. With eight available devices and no
explicit device selection, a four-rank run uses physical devices 0-3.
Running it without arguments or with `--help` prints usage.

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
- Checks that Python and pip are available before downloading dependencies
- Installs the PyPI `mermaid-cli` package and its Playwright Chromium browser
- Validates the installed `mmdc` interface and launches Chromium once
- Installs repo-managed local build utilities
- Installs MPICH for multi-rank tests

The Mermaid setup requires access to the configured PyPI index and Playwright browser
CDN. Installation failures stop the script before native dependency downloads begin.

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
