# TileXR UDMA 2-Card P2P Performance Test Guide

## Background

This guide designs a 2-card single-pair P2P performance test for the TileXR
UDMA path, using commit `378724c583c417ab2388a3c5354bfdb0e57dc589` as the
reference baseline. That commit already contains the UDMA demo infrastructure
under `tests/udma/demo`, including local rank launch, TileXR communicator
initialization, registered device memory, UDMA kernel launches, and per-rank logs.

The P2P test should reuse that infrastructure and add a focused perf mode for
two ranks only.

## Goal

Measure one directed UDMA P2P transfer at a time on 2 cards:

- `rank0 -> rank1`
- `rank1 -> rank0`

The final output should include:

- correctness status for each transfer size and direction;
- latency in microseconds;
- effective bandwidth in GB/s;
- a CSV result file;
- a bandwidth curve comparing `0->1` and `1->0`.

## Test Model

Use `rank_size=2`. Each run launches two local ranks, one rank per NPU.

The P2P perf mode now supports two transport modes:

- `direct_urma`: the AIV kernel posts a direct TileXR UDMA put from ordinary
  `aclrtMalloc` memory registered by `TileXRUDMARegister`.
- `memory`: the AIV kernel uses Ascend C `DataCopyPad` to copy bytes from the
  source rank's local GM through UB into the destination rank's TileXR IPC peer
  memory window, `CommArgs::peerMems[dst] + IPC_DATA_OFFSET`.

For `direct_urma`, for a selected direction:

- the sender rank posts one UDMA write to the receiver's registered memory;
- the sender calls `UDMAQuietStatus` and records the completion status;
- the receiver does not send data in that measurement round;
- after stream synchronization and a host barrier, the receiver copies its
  destination buffer back and validates the expected byte pattern.

For `memory`, for the same selected direction:

- the sender rank launches `tilexr_memory_p2p_perf_kernel`;
- only the sender rank performs the device-side copy;
- the destination address is its peer IPC data window on the receiver;
- the receiver validates by copying its local IPC data window back to host.

Run both directions independently so each result row has an unambiguous source
and destination. For current comparison work, `0->1` is usually enough because
`0->1` and `1->0` were observed to be effectively identical on the tested
2-card setup.

Important scope notes:

- `direct_urma` measures the TileXR registered-memory URMA/UDMA path.
- `memory` measures peer-memory IPC semantics implemented with AIV
  `DataCopyPad`. It is a useful baseline, not the same hardware data path as
  UDMA queue based direct URMA.
- `memory` is limited by the TileXR IPC data window. The current wrapper rejects
  sizes above 100 MiB.
- The current memory comparison kernel uses one AIV block in the runner. Large
  messages therefore reflect single-block GM->UB->peer-GM copy throughput.

## Metrics

Recommended CSV fields:

```text
direction,src,dst,ranks,bytes,iters,avg_us,min_us,max_us,bw_GBps,status,errors,log_dir
```

Definitions:

- `direction`: `0to1` or `1to0`.
- `bytes`: transfer bytes per measured UDMA write.
- `avg_us`: average measured time per transfer, excluding warmup.
- `min_us` and `max_us`: per-transfer minimum and maximum from measured
  iterations.
- `bw_GBps`: `bytes / avg_us / 1000`.
- `status`: UDMA completion status from `UDMAQuietStatus`; `0` means success.
- `errors`: validation mismatch count on the receiver side.

Do not include communicator initialization, memory registration, H2D/D2H copies,
or TCP barriers in the measured interval.

For both transports, timing is taken on the source rank and shared through
per-rank status files before rank 0 writes the CSV row. This avoids reporting
receiver-side empty-kernel timing for the reverse direction.

## Recommended Sweep

Use this default sweep first:

```text
rank_size     = 2
npu_count     = 2
first_npu     = 0
directions    = 0->1, 1->0
min_bytes     = 4096
max_bytes     = 268435456
step_factor   = 2
warmup_iters  = 10
iters         = 100
check         = 1
```

If large messages are unstable during bring-up, start with:

```text
min_bytes     = 4096
max_bytes     = 16777216
warmup_iters  = 5
iters         = 20
```

Then expand the range after both directions pass correctness.

## Code Changes

### 1. Add A P2P Perf Test Type

Extend `tests/udma/demo/tilexr_udma_demo.cpp` and
`tests/udma/demo/tilexr_udma_demo_kernel.cpp` with a new mode:

```text
test_type=4: 2-card directed UDMA P2P performance test
```

Suggested host options:

```text
--src-rank <0|1>
--dst-rank <0|1>
--min-bytes <bytes>
--max-bytes <bytes>
--step-factor <N>
--warmup-iters <N>
--iters <N>
--csv <path>
--check <0|1>
```

For compatibility with the existing positional demo style, a wrapper script can
translate simple positional arguments into these options.

### 2. Add A Device Kernel

Add a kernel shaped like this:

```cpp
extern "C" __global__ __aicore__ void tilexr_udma_p2p_perf_kernel(
    GM_ADDR commArgsGM, GM_ADDR srcGM, GM_ADDR debugGM,
    int32_t srcRank, int32_t dstRank, uint64_t dstByteOffset,
    uint32_t bytes, uint32_t pattern)
{
    auto args = reinterpret_cast<__gm__ TileXR::CommArgs*>(commArgsGM);
    auto src = reinterpret_cast<__gm__ uint8_t*>(srcGM);
    auto debug = reinterpret_cast<__gm__ uint32_t*>(debugGM);

    const int32_t rank = args->rank;
    if (rank != srcRank) {
        return;
    }

    TileXR::UDMAPutNbi<uint8_t>(args, dstRank, src, dstByteOffset, bytes);
    uint32_t status = TileXR::UDMAQuietStatus(args, dstRank);
    if (debug != nullptr) {
        debug[0] = status;
        debug[1] = bytes;
        debug[2] = pattern;
    }
}
```

The source buffer should already contain the host-generated pattern before the
measured launch. The receiver validates the destination buffer after the measured
iteration batch.

### 3. Allocate Registered Buffers

For each rank, allocate one registered memory region with:

```text
src buffer
dst buffer
debug/status buffer
```

Round the total registered bytes to the existing 2 MiB UDMA registration
alignment used by the demo.

The remote destination offset passed to the kernel is the byte offset of the
receiver's `dst buffer` inside that registered region.

### 4. Measure The Transfer

Use one stream and event timing around each measured kernel launch batch:

```text
warmup:
  launch kernel
  synchronize stream

measured:
  record start event
  repeat iters:
    launch kernel
  record stop event
  synchronize stop event
```

Compute:

```text
avg_us = elapsed_ms * 1000 / iters
bw_GBps = bytes / avg_us / 1000
```

If per-iteration min and max are required, record events around each measured
iteration. For initial bring-up, average timing is enough; min and max can be
added once correctness is stable.

### 5. Write CSV

Only rank 0 should create or append to the CSV. For direction `1->0`, rank 0 is
the receiver, so rank 0 can still write the result after copying back debug and
validation state.

Each size and direction emits one row.

Example:

```csv
direction,src,dst,ranks,bytes,iters,avg_us,min_us,max_us,bw_GBps,status,errors,log_dir
0to1,0,1,2,4096,100,8.31,0,0,0.493,0,0,logs/tilexr_udma_p2p_perf_20260622_120000
1to0,1,0,2,4096,100,8.45,0,0,0.485,0,0,logs/tilexr_udma_p2p_perf_20260622_120000
```

When min and max are not implemented yet, write `0` for those fields and keep
the columns stable.

## Runner Script

Add a script:

```text
tests/udma/demo/run_tilexr_udma_p2p_perf.sh
```

Suggested usage:

```bash
bash demo/run_tilexr_udma_p2p_perf.sh \
  <src_rank> <dst_rank> <min_bytes> <max_bytes> <step_factor> \
  <iters> <warmup_iters> <first_npu> <check> <transport>
```

Example:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_p2p_perf.sh 0 1 4096 67108864 2 100 10 2 1 direct_urma
bash demo/run_tilexr_udma_p2p_perf.sh 0 1 4096 67108864 2 100 10 2 1 memory
```

Arguments:

- `src_rank`, `dst_rank`: directed transfer, normally `0 1` or `1 0`.
- `min_bytes`, `max_bytes`, `step_factor`: byte sweep definition.
- `iters`, `warmup_iters`: measured and warmup launch counts.
- `first_npu`: first physical NPU id used by local rank 0. For physical cards
  2 and 3, pass `2`.
- `check`: `1` validates destination bytes after each size.
- `transport`: `direct_urma` or `memory`; default is `direct_urma`.

The script should:

- source `scripts/common_env.sh`;
- set `TILEXR_COMM_ID` if it is not already set;
- set `TILEXR_DEMO_NPUS=2`;
- set `TILEXR_DEMO_FIRST_NPU`;
- launch exactly two ranks;
- write per-rank logs under
  `tests/udma/logs/tilexr_udma_p2p_perf_*_<transport>_<direction>`;
- write CSV under the same log directory.

## Build

Build TileXR core first:

```bash
cd /path/to/TileXR
source scripts/common_env.sh
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=../install ..
make -j$(nproc)
make install
```

Build UDMA tests and demo:

```bash
cd /path/to/TileXR/tests/udma
bash build.sh
```

Expected binary:

```text
tests/udma/install/bin/tilexr_udma_demo
```

## Run

### Run direct URMA

Run a short correctness-oriented sweep first:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_p2p_perf.sh 0 1 4096 16777216 2 20 5 2 1 direct_urma
```

Then run the larger sweep. On the tested `141.62.19.144` environment, 64 MiB
was used as the stable upper bound because 128 MiB and above failed during
UDMA registration due to the registered region size.

```bash
bash demo/run_tilexr_udma_p2p_perf.sh 0 1 4096 67108864 2 100 10 2 1 direct_urma
```

### Run memory semantics

Use the same sweep and only change the transport:

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_p2p_perf.sh 0 1 4096 67108864 2 100 10 2 1 memory
```

For diagnosis around the observed large-message knee, run single-size points:

```bash
for bytes in 33554432 41943040 50331648 58720256 62914560 67108864 75497472 83886080 100663296; do
  TILEXR_COMM_ID=127.0.0.1:$((12000 + bytes / 1048576)) \
    bash demo/run_tilexr_udma_p2p_perf.sh 0 1 ${bytes} ${bytes} 2 50 5 2 1 memory
done
```

Success criteria:

- both rank processes exit with code 0;
- every row has `status=0`;
- every row has `errors=0`;
- for `direct_urma`, per-rank logs include UDMA enabled in `CommArgs`;
- for `memory`, per-rank logs show non-null `peerMems[]` for the tested ranks;
- no rank log contains `MISMATCH`, `TileXR UDMA demo failed`, or `ERROR`.

## Plot The Curve

Use the plotting helper:

```text
tests/udma/demo/plot_tilexr_udma_p2p_perf.py
```

For the old direction comparison:

```bash
python3 demo/plot_tilexr_udma_p2p_perf.py \
  logs/tilexr_udma_p2p_perf_*/p2p_perf.csv \
  --output logs/tilexr_udma_p2p_perf_curve.png
```

For the current direct URMA vs memory comparison, pass both CSVs and label the
series explicitly:

```bash
python3 demo/plot_tilexr_udma_p2p_perf.py \
  logs/tilexr_udma_p2p_perf_20260622_173102_direct_urma_0to1/p2p_perf.csv \
  logs/tilexr_udma_p2p_perf_20260622_173138_memory_0to1/p2p_perf.csv \
  --direction 0to1 \
  --labels direct_urma,memory \
  --output logs/tilexr_udma_vs_memory_0to1_bandwidth.png \
  --latency-output logs/tilexr_udma_vs_memory_0to1_latency.png \
  --latency-max-bytes 1048576
```

The script generates:

- bandwidth curve: all rows in the selected CSVs;
- latency curve: rows up to `--latency-max-bytes`, default 1 MiB.

The plot uses:

- x-axis: `bytes`, log scale;
- x tick labels: human-readable `KB`, `MB`, or `GB`;
- y-axis: `bw_GBps` for bandwidth or `avg_us` for latency;
- one line per direction or per explicit label.

If multiple CSV files are passed, merge rows by direction and bytes before
plotting. When `--labels` is used, merge rows by label and bytes instead, which
allows two `0to1` CSV files to appear as separate series.

### Local CSV and plot workflow

When the test runs on a remote Ascend server, pull only the CSVs to the local
workspace and plot locally:

```powershell
$out = "D:\workspace\TileXR\tests\udma\logs\p2p_compare_20260622_173138_64MiB"
New-Item -ItemType Directory -Force -Path $out | Out-Null

scp -i C:\Users\h30059441\.ssh\id_ed25519 `
  root@141.62.19.144:/home/h30059441/TileXR/tests/udma/logs/tilexr_udma_p2p_perf_20260622_173102_direct_urma_0to1/p2p_perf.csv `
  "$out\p2p_perf_direct_urma_0to1.csv"

scp -i C:\Users\h30059441\.ssh\id_ed25519 `
  root@141.62.19.144:/home/h30059441/TileXR/tests/udma/logs/tilexr_udma_p2p_perf_20260622_173138_memory_0to1/p2p_perf.csv `
  "$out\p2p_perf_memory_0to1.csv"

python tests\udma\demo\plot_tilexr_udma_p2p_perf.py `
  "$out\p2p_perf_direct_urma_0to1.csv" `
  "$out\p2p_perf_memory_0to1.csv" `
  --direction 0to1 `
  --labels direct_urma,memory `
  --output "$out\tilexr_udma_vs_memory_0to1_bandwidth.png" `
  --latency-output "$out\tilexr_udma_vs_memory_0to1_latency.png" `
  --latency-max-bytes 1048576
```

Example output files:

```text
tests/udma/logs/p2p_compare_20260622_173138_64MiB/p2p_perf_direct_urma_0to1.csv
tests/udma/logs/p2p_compare_20260622_173138_64MiB/p2p_perf_memory_0to1.csv
tests/udma/logs/p2p_compare_20260622_173138_64MiB/tilexr_udma_vs_memory_0to1_bandwidth.png
tests/udma/logs/p2p_compare_20260622_173138_64MiB/tilexr_udma_vs_memory_0to1_latency.png
```

## Result Review

Check the CSV first:

```bash
column -s, -t logs/tilexr_udma_p2p_perf_*/p2p_perf.csv | less -S
```

Look for:

- `status != 0`: UDMA completion failure or CQ polling issue;
- `errors != 0`: receiver data mismatch;
- sharp bandwidth drop at a specific size: possible registration range,
  alignment, CQ depth, or timeout issue;
- large directional difference between `0->1` and `1->0`: possible topology or
  route asymmetry.
- `memory` bandwidth decreasing after 32 MiB: this is not necessarily a data
  error. On the tested setup, single-size memory runs showed a gradual drop
  from about 48 GB/s at 32 MiB to about 39 GB/s at 64 MiB and above. That shape
  is consistent with the current single-AIV-block `DataCopyPad` IPC-window
  baseline entering sustained GM->UB->peer-GM throughput, not with a 64 MiB
  correctness boundary.

Keep the raw logs with the CSV and curve. They are needed to confirm UDMA was
enabled and to inspect `CommArgs`, registered memory offsets, and debug words.

## Example Result From 141.62.19.144

Environment:

```text
CANN: /usr/local/Ascend/ascend-toolkit
MPI:  /usr/local/mpich
NPUs: physical 2 and 3
Direction: 0->1
Sweep: 4096 -> 67108864, step_factor=2, iters=100, warmup=10
```

Selected rows:

```text
transport    bytes      avg_us    bw_GBps  status  errors
direct_urma  4096       6.504     0.630    0       0
direct_urma  1048576    26.040    40.267   0       0
direct_urma  16777216   322.074   52.091   0       0
direct_urma  67108864   1269.934  52.844   0       0
memory       4096       4.059     1.009    0       0
memory       1048576    23.484    44.651   0       0
memory       16777216   334.924   50.093   0       0
memory       67108864   1691.058  39.685   0       0
```

Memory-only diagnostic points:

```text
32MiB  48.16 GB/s
40MiB  45.54 GB/s
48MiB  42.09 GB/s
56MiB  40.59 GB/s
60MiB  40.16 GB/s
64MiB  39.56 GB/s
72MiB  39.21 GB/s
80MiB  39.15 GB/s
96MiB  39.10 GB/s
```

Interpretation:

- direct URMA reaches about 52.8 GB/s at 64 MiB on this pair;
- memory semantics is competitive for small and mid-sized messages, but the
  current single-block DataCopyPad baseline settles near 39-40 GB/s for larger
  messages;
- both paths passed payload validation with `status=0` and `errors=0`.

## Minimal Acceptance Checklist

- `test_type=4` exists and is documented in the UDMA demo README.
- `run_tilexr_udma_p2p_perf.sh` can run `0->1` and `1->0` with `rank_size=2`.
- CSV rows are produced for every requested size.
- The receiver validates the copied payload for every measured size.
- The plot helper generates a two-line bandwidth curve.
- A short sweep and full sweep both pass with `status=0` and `errors=0`.
