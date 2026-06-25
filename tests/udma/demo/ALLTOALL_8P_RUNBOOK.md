# TileXR UDMA All-to-All 8P Runbook

This runbook records the 8P all-to-all demo version based on commit
`e552736 Add UDMA demo checker collectives`.

## Scope

- Demo path: `tests/udma/demo`
- Operator mode: `test_type=2`, all-to-all UDMA put
- Target hardware: A5 / Ascend950 / 950
- Process model: one local process per rank
- Validated baseline: `rank_size=8`, `npu_count=8`, `first_npu=0`

The all-to-all layout is:

- rank `src` fills input slice `dst` with `100000 + src * 1000 + dst`;
- rank `src` sends slice `dst` to rank `dst`;
- rank `dst` output is ordered by source rank.

For rank `0`, the output sample should contain:

```text
from0=100000 from1=101000 from2=102000 from3=103000 from4=104000 from5=105000 from6=106000 from7=107000
```

For rank `7`, the output sample should contain:

```text
from0=100007 from1=101007 from2=102007 from3=103007 from4=104007 from5=105007 from6=106007 from7=107007
```

## Environment

Use a root shell on the Ascend machine.

```bash
cd /path/to/TileXR
source scripts/common_env.sh
npu-smi info
```

Expected:

- `npu-smi info` lists at least 8 usable devices;
- CANN environment variables are available after `source scripts/common_env.sh`;
- `bisheng` is available for building `tilexr_udma_demo_kernel.cpp`;
- MPI, if needed by surrounding scripts, is under `/usr/local/mpi/`.

## Build

Build `tile-comm` and install it into the repository `install` directory:

```bash
cd /path/to/TileXR
source scripts/common_env.sh
cmake -S . -B /tmp/tilexr-build-udma -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build /tmp/tilexr-build-udma --target tile-comm -j"$(nproc)"
cmake --install /tmp/tilexr-build-udma
```

Build the UDMA demo:

```bash
cd /path/to/TileXR/tests/udma
bash build.sh
```

Check artifacts:

```bash
test -x install/bin/tilexr_udma_demo
test -f install/lib/libtilexr_udma_demo_kernel.so
```

## Run 8P All-to-All

The script arguments are:

```text
run_tilexr_udma_demo.sh <test_type> <rank_size> <elements_per_rank> <npu_count> <first_npu>
```

Run the normal/default IPC initialization path:

```bash
cd /path/to/TileXR/tests/udma
export TILEXR_COMM_ID=127.0.0.1:10067
bash demo/run_tilexr_udma_demo.sh 2 8 16 8 0
```

Run with explicit PID IPC mode:

```bash
cd /path/to/TileXR/tests/udma
export TILEXR_COMM_ID=127.0.0.1:10077
TILEXR_IPC_PID_MODE=pid bash demo/run_tilexr_udma_demo.sh 2 8 16 8 0
```

Run with explicit SDID IPC mode:

```bash
cd /path/to/TileXR/tests/udma
export TILEXR_COMM_ID=127.0.0.1:10087
TILEXR_IPC_PID_MODE=sdid bash demo/run_tilexr_udma_demo.sh 2 8 16 8 0
```

Use a different `TILEXR_COMM_ID` port for concurrent or repeated runs to avoid
the demo TCP barrier colliding with a previous process.

## 1M Data Run

For a 1M-elements-per-peer checker-sized run:

```bash
cd /path/to/TileXR/tests/udma
export TILEXR_COMM_ID=127.0.0.1:10107
bash demo/run_tilexr_udma_demo.sh 2 8 1048576 8 0
```

This means:

- `elements_per_rank=1048576` int32 elements per destination slice;
- each rank input buffer has `8 * 1048576` int32 elements;
- each rank input/output buffer is 32 MiB.

## Log Checks

Each run writes logs under:

```text
tests/udma/logs/tilexr_udma_demo_YYYYmmdd_HHMMSS/
```

Quick success check:

```bash
cd /path/to/TileXR/tests/udma
latest=$(ls -td logs/tilexr_udma_demo_* | head -n1)
grep -R "TileXR UDMA demo success" "$latest"
grep -R "UDMA=enabled" "$latest"
grep -R "TileXRUDMARegister success" "$latest"
```

There should be 8 success lines, one for each rank.

Check all-to-all samples:

```bash
grep -R "alltoall output sample" "$latest"
```

Check for failures:

```bash
grep -R "ALLTOALL MISMATCH\|DATA MISMATCH\|TileXR UDMA demo failed\|ERROR" "$latest" || true
```

Expected: no mismatch or failure lines.

If UDMA CQ is incomplete, the demo may print:

```text
alltoall UDMA CQ incomplete, use IPC fallback
alltoall IPC fallback completed
```

In that case, the final correctness criterion is still the all-to-all output
validation and `TileXR UDMA demo success` on every rank.

## Offline Checker

The layout checker can be built without running the hardware demo:

```bash
cd /path/to/TileXR/tests/udma
g++ -std=c++14 -O2 \
  -I . \
  -DTILEXR_SOURCE_ROOT='"'/path/to/TileXR'"' \
  unit/test_tilexr_udma_alltoall_layout.cpp \
  -o /tmp/test_tilexr_udma_alltoall_layout
/tmp/test_tilexr_udma_alltoall_layout
```

Expected output:

```text
TileXR UDMA all-to-all layout checks passed
```

This checker validates the all-to-all input pattern, expected output layout, and
source-level debug layout assumptions. It does not execute UDMA or AICore code.

## Notes

- `test_type=2` is the all-to-all path.
- `test_type=3` is the all-reduce path and is not covered by this runbook.
- `TILEXR_IPC_PID_MODE=sdid` forces `rtSetIpcMemorySuperPodPid`.
- `TILEXR_IPC_PID_MODE=pid` forces `rtSetIpcMemPid`.
- Leaving `TILEXR_IPC_PID_MODE` unset uses TileXR's chip default.
- The demo host source does not use `shmem.h`; process synchronization is a
  local TCP barrier derived from `TILEXR_COMM_ID`.
