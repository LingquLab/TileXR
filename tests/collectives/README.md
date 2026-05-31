# TileXR Collectives Tests

This directory builds source/unit checks plus manual multi-process tools for `libtilexr-collectives`.
The API/library split is intentional: `tile-comm` owns communicator infrastructure, while `libtilexr-collectives`
owns `TileXRAllGather`, `TileXRAllToAll`, host launch, and CCE registration.

## Build

From the repository root:

```bash
source scripts/common_env.sh
cmake -S . -B build -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_BUILD_TESTS=ON -DBUILD_TESTING=OFF
cmake --build build --target test_tilexr_collectives_correctness tilexr_collective_perf -j"$(nproc)"
```

CTest source and CLI smoke checks are hardware-free. They verify headers, split ownership, scripts, docs, and
tool wiring. Physical multi-NPU execution is manual.

## Correctness Run

The correctness runner starts one process per rank through the helper script:

```bash
cd tests/collectives
./run_collectives_correctness.sh 2 16 0 ../../build/tests/collectives both
```

Arguments are `rank_size count first_npu bin_dir [op]`. The binary is
`test_tilexr_collectives_correctness` and also accepts `--rank-size`, `--rank`, `--count`, `--first-npu`,
and `--op allgather|alltoall|both`. It initializes ACL, selects `first_npu + rank`, creates a stream,
calls `TileXRCommInitRankLocal`, runs INT32 `TileXRAllGather` and equal `TileXRAllToAll`, synchronizes,
copies results back, and validates deterministic rank-specific patterns. The initial equal `TileXRAllToAll`
kernel path is available only when the communicator reports `TOPO_910_93`; other multi-rank topologies are
rejected instead of launching a kernel path that would do no work.

The script writes `collectives_correctness_rank*.log` and tails logs on failure. Multi-rank launches use
`TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC` as a whole-launch timeout, defaulting to 600 seconds. If any rank fails
or the timeout expires, the script kills remaining ranks and tails each rank log.

## Perf Run

`tilexr_collective_perf` is an nccl-tests-style correctness/performance tool:

```bash
cd tests/collectives
./run_collective_perf.sh 2 0 ../../build/tests/collectives \
  --op allgather --min-bytes 4 --max-bytes 4096 --step-factor 2 \
  --iters 20 --warmup-iters 5 --datatype int32 --check 1
```

Main options are `--op allgather|alltoall`, `--min-bytes`, `--max-bytes`, `--step-factor`, `--iters`,
`--warmup-iters`, `--datatype int8|int16|int32|int64|fp16|fp32|bf16`, `--rank-size`, `--rank`,
`--first-npu`, `--check 0|1`, and `--csv <path>`. Optional thresholds `--min-algbw` and
`--max-latency-us` are available but are not set by default.

Output fields:

```text
op dtype ranks bytes count iters algbw(GB/s) busbw(GB/s) avg(us) min(us) max(us) errors
```

The `bytes` column is the actual send bytes per rank for that operation and size row:
allgather: count * dtype_size; alltoall: count * rank_size * dtype_size. `algbw(GB/s)` is
`output_bytes_per_rank / avg_us / 1000`, where `output_bytes_per_rank` is the bytes received by one rank for
the measured message size. `busbw(GB/s)` is `algbw * (rank_size - 1) / rank_size` for both allgather and
equal alltoall. CSV output uses the same fields.

`--check=1` validates outputs. INT32 is checked element-by-element with operation-specific expected values;
other dtypes use deterministic byte-pattern checks. `--check=0` measures only.

### Operator-Internal Profiling

Build collectives with profiling enabled:

```bash
cmake -S . -B build-profile -DTILEXR_BUILD_COLLECTIVES=ON -DTILEXR_COLLECTIVES_ENABLE_PROFILING=ON
cmake --build build-profile --target tilexr_collective_perf -j"$(nproc)"
```

Run the perf tool with profiling:

```bash
./run_collective_perf.sh 2 0 ../../build-profile/tests/collectives \
  --op allgather --min-bytes 67108864 --max-bytes 67108864 \
  --profile 1 --profile-dir run/prof/collectives --profile-ai-prompt 1
```

The report directory contains `trace.json`, `summary.csv`, `analysis.md`, `report.html`, and `ai_prompt.md`
when prompt export is enabled. `--profile-dir` is a root directory; each rank writes its own report under
`run/prof/collectives/rank<N>/` in the example above.

## Skip Behavior

Manual scripts are strict by default. If `npu-smi info -l` or `TILEXR_AVAILABLE_NPUS` reports too few devices,
they exit nonzero. Set `TILEXR_SKIP_IF_INSUFFICIENT_NPUS=1` to skip cleanly instead:

```bash
TILEXR_SKIP_IF_INSUFFICIENT_NPUS=1 ./run_collectives_correctness.sh 8 16 0 ../../build/tests/collectives
```

Set `TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=N` to override the default 600-second launch timeout for either
manual script.

Common failure reasons include missing CANN environment, too few visible NPUs, devices already in use,
driver/runtime version mismatch, unsupported topology for the selected collective kernel, or `LD_LIBRARY_PATH`
not including the TileXR install/build library directories.
