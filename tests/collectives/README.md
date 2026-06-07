# TileXR Collectives Tests

This directory builds source/unit checks plus manual multi-process tools for `libtilexr-collectives`.
The API/library split is intentional: `tile-comm` owns communicator infrastructure, while `libtilexr-collectives`
owns `TileXRAllGather`, `TileXRAllToAll`, `TileXRAllReduce`, `TileXRReduceScatter`, `TileXRBroadcast`,
host launch, and CCE registration.

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

Arguments are `rank_size count first_npu bin_dir [op] [extra correctness args...]`. The binary is
`test_tilexr_collectives_correctness` and also accepts `--rank-size`, `--rank`, `--count`, `--first-npu`,
`--root`, and `--op allgather|alltoall|allreduce|reducescatter|broadcast|both`. It initializes ACL, selects
`first_npu + rank`, creates a stream, calls `TileXRCommInitRankLocal`, runs the selected INT32 collective,
synchronizes, copies results back, and validates deterministic rank-specific patterns. `both` runs the
original `TileXRAllGather` plus equal `TileXRAllToAll` checks. The initial equal `TileXRAllToAll` kernel path
is available only when the communicator reports `TOPO_910_93`; other multi-rank topologies are rejected
instead of launching a kernel path that would do no work.

Broadcast correctness should cover multiple root choices, for example:

```bash
./run_collectives_correctness.sh 2 16 0 ../../build/tests/collectives broadcast --root 0
./run_collectives_correctness.sh 2 16 0 ../../build/tests/collectives broadcast --root 1
```

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

Main options are `--op allgather|alltoall|allreduce|reducescatter|broadcast`, `--min-bytes`, `--max-bytes`, `--step-factor`, `--iters`,
`--warmup-iters`, `--datatype int8|int16|int32|int64|fp16|fp32|bf16`, `--rank-size`, `--rank`,
`--first-npu`, `--check 0|1`, and `--csv <path>`. Optional thresholds `--min-algbw` and
`--max-latency-us` are available but are not set by default.

Output fields:

```text
op dtype ranks bytes count iters algbw(GB/s) busbw(GB/s) avg(us) min(us) max(us) errors
```

The `bytes` column is the actual send bytes per rank for that operation and size row. Message-size semantics: allgather/allreduce/broadcast: count * dtype_size; alltoall/reducescatter: count * rank_size * dtype_size. `algbw(GB/s)` is
`output_bytes_per_rank / avg_us / 1000`, where `output_bytes_per_rank` is the bytes received by one rank for
the measured message size. `busbw(GB/s)` is `algbw * (rank_size - 1) / rank_size` for the measured operation.
CSV output uses the same fields.

`--check=1` validates outputs. INT32 is checked element-by-element with operation-specific expected values;
other dtypes use deterministic byte-pattern checks for allgather, alltoall, and broadcast. Checked
allreduce/reducescatter runs require `--datatype int32`; `--check=0` measures any supported dtype.

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

Each sampled measured launch writes `trace.json`, `summary.csv`, `analysis.md`, `report.html`, and `ai_prompt.md`
when prompt export is enabled. `--profile-dir` is a root directory; each rank writes sampled launches under
`run/prof/collectives/rank<N>/launch<M>/` in the example above.

After all rank processes finish successfully, `run_collective_perf.sh` also writes a root-level report:

```text
run/prof/collectives/report.html
run/prof/collectives/trace_index.json
run/prof/collectives/analysis.md
```

When prompt export is enabled, the aggregate prompt is written as `run/prof/collectives/ai_prompt.md`.
The root-level report.html keeps the bottleneck-first summary and adds a zoomable chronological timeline across
sampled measured iterations. Warmup execution is controlled by the existing `--warmup-iters` option and is reported
as metadata; warmup launches are not profiled by this report path. The per-launch `rank<N>/launch<M>/report.html`
files remain available for drilldown.

To regenerate the aggregate report from an existing profile directory:

```bash
python3 tilexr_collective_profile_report.py run/prof/collectives \
  --warmup-iters 5 --iters 20 --profile-sample-every 1 --emit-ai-prompt
```

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

## vllm-ascend Shim Smoke on Remote NPU Host

The Phase 2 vllm-ascend shim validation keeps all remote state under a scratch directory and does not modify
the remote system Python or shell startup files.

```bash
TILEXR_VLLM_REMOTE=blue \
TILEXR_VLLM_REMOTE_BASE=/path/to/remote/tilexr_vllm_collectives_$(date +%Y%m%d_%H%M%S) \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

If the remote login shell does not already expose CANN Toolkit or the CCE compiler, pass the remote paths
explicitly:

```bash
TILEXR_VLLM_REMOTE_ASCEND_HOME_PATH=/path/to/remote/ascend-toolkit \
TILEXR_VLLM_REMOTE_ASCEND_DRIVER_PATH=/path/to/remote/ascend-driver \
TILEXR_VLLM_REMOTE_CMAKE_CCE_COMPILER=/path/to/remote/ascend-toolkit/bin/ccec \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

If the remote default Python does not provide `torch_npu`, select a Python environment explicitly. A direct Python
path takes precedence over a conda environment name:

```bash
TILEXR_VLLM_REMOTE_PYTHON=/path/to/remote/python \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

```bash
TILEXR_VLLM_REMOTE_CONDA_ENV=tt4 \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

Set `TILEXR_VLLM_REMOTE_CONDA_SH` when the remote conda activation script is not available at
`/home/miniconda3/etc/profile.d/conda.sh`.

The script syncs the current TileXR commit, initializes submodules from local worktrees, dumps the NPU/CANN/Python
environment, builds `tile-comm` and `tilexr-collectives`, runs the standalone 2-rank AllGather correctness check, and
runs the Python torch-npu shim AllGather smoke for `int32` and `fp16`.

Expected success lines include:

```text
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=fp16
```

If `torch` or `torch-npu` is missing in the selected Python environment, the preflight fails before the multi-rank
shim smoke. Missing `vllm` or `vllm-ascend` is recorded in the environment dump but does not fail this shim phase.
