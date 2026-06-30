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

Main options are `--op allgather|alltoall|allreduce|reducescatter|broadcast|profile-probe`, `--min-bytes`, `--max-bytes`, `--step-factor`, `--iters`,
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
run/prof/collectives/perfetto_trace.json
```

When prompt export is enabled, the aggregate prompt is written as `run/prof/collectives/ai_prompt.md`.
The root-level report.html keeps the bottleneck-first summary, adds a rank-level summary for spotting slow ranks,
and adds a zoomable chronological timeline across sampled measured iterations. `perfetto_trace.json` uses
launch/rank/stage event names plus per-rank launch windows so `ui.perfetto.dev` can quickly filter by launch or rank.
Warmup execution is controlled by the existing `--warmup-iters` option and is reported as metadata; warmup launches are not profiled by this report path.
The per-launch `rank<N>/launch<M>/report.html` files remain available for drilldown.

To regenerate the aggregate report from an existing profile directory:

```bash
python3 tilexr_collective_profile_report.py run/prof/collectives \
  --warmup-iters 5 --iters 20 --profile-sample-every 1 --emit-ai-prompt
```

### Multi-Host Profiling

`tilexr_collective_perf` also supports a socket bootstrap mode for one process per host. The helper script below
starts each rank over SSH with `--comm-mode socket`, collects `rank<N>/` profile directories back to the first host,
and then writes a single aggregate `report.html` plus `perfetto_trace.json` containing host/rank labels.

Use `--op profile-probe` first as a multi-host profiling/report smoke mode. It initializes ACL and the socket
communicator on all hosts, launches a real AIV kernel on each rank, and records the same kernel-level stages as the
single-host collective report. The probe only performs local device-memory copies and does not touch cross-host
`peerMems[]`, so it is useful for two-host profiling/report validation but should not be reported as allgather
cross-host bandwidth.

```bash
cd tests/collectives
TILEXR_MULTIHOST_PEERS="0,root@141.62.24.62,141.62.24.62,0;1,root@141.62.24.70,141.62.24.70,0" \
TILEXR_COMM_ID=141.62.24.62:10067 \
TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=300 \
bash ./run_collective_perf_multihost.sh /home/l00929943/TileXR/run/prof/collectives-2host-profile-probe-62-70 \
  /home/l00929943/TileXR/build-profile-950/tests/collectives \
  --op profile-probe --min-bytes 4096 --max-bytes 4096 \
  --iters 2 --warmup-iters 1 --datatype int32 --check 0 \
  --profile 1 --profile-sample-every 1 --profile-ai-prompt 1
```

The current report normalizes each rank/launch independently because cross-host device cycles are not assumed to be
synchronized. Use the HTML rank-level summary to spot slow host/rank pairs, then drill into `rank<N>/launch<M>/`.
The helper labels hosts from the SSH target/IP in `TILEXR_MULTIHOST_PEERS`, so Perfetto event names look like
`launch0/rank1@141.62.24.70/kernel_total` in the example above.

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

The Phase 3 vllm-ascend shim validation keeps all remote state under a scratch directory and does not modify
the remote system Python or shell startup files. The TileXR collective path is opt-in: set
`VLLM_ASCEND_TILEXR_COLLECTIVES=1` to enable the adapter path used by the smoke script. Without that flag,
`TileXRVllmCollectivesAdapter` reports fallback so a vllm-ascend caller can keep its existing HCCL or
`torch.distributed` behavior.

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

Set `TILEXR_VLLM_REMOTE_PYTHONPATH` to prepend an isolated dependency directory or source checkout to the vLLM
environment probe and Python shim smoke without installing packages into the selected Python environment.

If vLLM and vllm-ascend are available as source trees rather than installed packages, pass their remote source
roots to make the environment probe add them to `PYTHONPATH`:

```bash
TILEXR_VLLM_REMOTE_VLLM_SOURCE=/path/to/remote/vllm \
TILEXR_VLLM_REMOTE_VLLM_ASCEND_SOURCE=/path/to/remote/vllm-ascend \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

Set `TILEXR_VLLM_REMOTE_DUMMY_MODEL` to a remote local HuggingFace config directory to additionally run a minimal
vllm-ascend inference probe with `skip_tokenizer_init=True` and `load_format="dummy"`. This avoids HuggingFace network
access and checks that the opt-in `sitecustomize` hook is loaded by the real vLLM engine and worker processes. The
probe prints `PASS TileXR vllm dummy inference probe` when `LLM.generate()` returns token ids. Set
`TILEXR_VLLM_REMOTE_VLLM_PLUGINS` only when the remote environment needs a plugin selection other than the default
`ascend`.
Set `TILEXR_VLLM_REMOTE_NNAL_ATB_SET_ENV` when the selected vllm-ascend worker path needs an NNAL/ATB environment
script to resolve `libatb.so`, for example `/usr/local/Ascend/nnal/atb/set_env.sh`.

The probe imports vLLM modules in child Python processes, including
`vllm.distributed.device_communicators.base_device_communicator` and
`vllm_ascend.distributed.device_communicators.npu_communicator`, so an import crash or missing dependency is
reported without aborting the TileXR shim validation. After CANN setup, the remote script also runs an
`NPUCommunicator` patch probe against the real vllm-ascend communicator class. That probe replaces the TileXR
adapter with a lightweight stub, verifies that `all_reduce` and `broadcast` route through the TileXR hook, verifies
that `all_gather` falls back when the adapter returns `None`, and prints TileXR route counts. This proves the
opt-in hook reaches the vllm-ascend communicator class; it still does not replace full vllm-ascend inference
validation with a model.

The script syncs the current TileXR commit, initializes submodules from local worktrees, dumps the NPU/CANN/Python
environment, probes whether `vllm` and `vllm_ascend` are importable before and after CANN setup, builds
`tile-comm` and `tilexr-collectives`, runs the standalone 2-rank AllGather correctness check, and runs Python
torch-npu shim smoke for `int32` and `fp16`.

Current shim coverage is:

- `allgather`
- `allreduce`
- `reducescatter`
- `broadcast`
- equal-split `alltoall` when the selected topology supports the standalone TileXR AllToAll path

The adapter intentionally falls back for disabled feature flags, non-NPU tensors, non-contiguous tensors, uneven
`all_gatherv` / `reduce_scatterv` style size lists, mismatched AllToAll scatter/gather dimensions, unsupported
dtypes or shapes, and missing TileXR libraries. The shim smoke validates TileXR's Python ABI and torch-npu tensor
pointer path; it is not a substitute for final vllm-ascend inference validation with a real model.

Expected success lines include:

```text
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=fp16
PASS TileXR vllm collectives smoke rank_size=2 op=allreduce dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=allreduce dtype=fp16
PASS TileXR vllm collectives smoke rank_size=2 op=reducescatter dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=reducescatter dtype=fp16
PASS TileXR vllm collectives smoke rank_size=2 op=broadcast dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=broadcast dtype=fp16
PASS TileXR vllm NPUCommunicator patch probe
TileXR vllm NPUCommunicator route counts: ...
PASS TileXR vllm dummy inference probe
```

If `torch` or `torch-npu` is missing in the selected Python environment, the preflight fails before the multi-rank
shim smoke. Missing or non-importable `vllm` / `vllm-ascend` is recorded in the environment dump and isolated import
probe but does not fail this shim phase. On `blue`, the older `tt4` environment records `libhccl.so` missing during
default pre-CANN vLLM import, then exposed missing `zmq` and incompatible torch versions for the available vLLM and
vllm-ascend source trees. The isolated `tilexr-vllm29` environment uses `torch==2.9.0`, `torch-npu==2.9.0`, and
`triton-ascend==3.2.0`; after CANN setup, its `vllm.distributed.device_communicators.base_device_communicator` and
`vllm_ascend.distributed.device_communicators.npu_communicator` probes import with `rc=0`, and the manual
`NPUCommunicator` patch probe printed route counts showing `all_reduce` and `broadcast` through TileXR plus
`all_gather` fallback. A manual `TILEXR_VLLM_REMOTE_DUMMY_MODEL=/home/<remote-user>/vllm-ascend/tests/ut/fake_weight`
probe reached the real vLLM V1 engine and spawned workers with `NPUCommunicator patched` printed in each process, then
stopped on a vllm-ascend worker import dependency (`torchvision`) before `LLM.generate()` completed. After installing
`torchvision` / `torchaudio` and sourcing NNAL ATB to resolve `libatb.so`, the same probe reached HCCL/Gloo
initialization and model weight loading, then stopped on a vLLM/vllm-ascend source mismatch:
`set_forward_context() got an unexpected keyword argument 'virtual_engine'`. The PR is not complete until
vllm-ascend inference validation is run or the remaining inference-level blocker is resolved and documented.
