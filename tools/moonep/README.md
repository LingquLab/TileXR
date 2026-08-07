# TileXR MoonEP Benchmark

The native five-stage API, tensor contracts, status ranges, and validation
boundaries are documented in `docs/moonep/DISPATCH_COMBINE.md`.

This benchmark runs the complete protocol with TileXR bootstrap and no HCCL. Expert
Forward has one implementation: BF16 `torch_npu.npu_grouped_matmul`,
`torch_npu.npu_swiglu`, then `torch_npu.npu_grouped_matmul`:

```text
Planning -> Dispatch -> PrefetchWeight -> expert forward -> Combine
         -> Dispatch(saved Plan) -> expert backward -> Combine(saved Plan)
         -> ReduceGrad
```

That statement applies to `--mode benchmark`. The tool also provides two untimed
correctness modes:

- `--mode reference` runs all five native stages through the independent baseline V1
  Torch reference (`torch_npu_reference_v1`) on NPU tensors, executes the shared
  Torch-NPU Expert Forward between PrefetchWeight and Combine, and uses an HCCL
  process group for cross-rank data.
- `--mode correctness` runs the reference and the built-in TileXR candidate backend after
  independently cloning every stage input, then compares every defined output and
  required/forbidden mutation before proceeding.

The default candidate is `tools.moonep.tilexr_backend:create_backend`. It requires an
installed five-stage native TileXR library, `transport_correctness_valid=true`, exact
launcher/context dimensions, and a same-node topology. An explicit
`--candidate-backend MODULE:FACTORY` overrides the built-in adapter. External factories
receive keyword arguments `torch_module`, `dimensions`, `case`, and `args`, and must
return an object satisfying `tools.moonep.contracts.MoonEPBackend`.

## Correctness Pipeline

The backend protocol remains five native MoonEP stages. The correctness runner adds an
untimed Expert Forward checkpoint, so every rank emits six stage artifacts:

```text
Planning:
  topk_experts [S,K] int32, tokens_per_expert [E] int32
  -> dst, cu_seqlens, experts_to_copy, zero_fill_ranges, remote_stats
Dispatch:
  plan, hidden [S,H] bf16, optional route weights [S,K] fp32
  -> hidden [NvS,H], optional weights [NvS], semantic dedup plan
PrefetchWeight:
  plan, gate/up/down weights [E+B,...] bf16
  -> the same tensors updated in place
ExpertForward:
  dispatched hidden [NvS,H], cu_seqlens [E+B], prefetched weights, weights [NvS]
  -> GMM1 [H,2I], NPU SwiGLU, GMM2 [I,H], route weighting, hidden [NvS,H]
Combine:
  plan, expert output [NvS,H] bf16, optional weights [NvS] fp32
  -> hidden [S,H], optional weights [S,K]
ReduceGrad:
  plan, full grads [E+B,...] fp32, reduce buffers [R,B,...] fp32
  -> owner rows accumulated and consumed live slots cleared in place
```

Every stage checks input structure and values, output structure and values, input
immutability or required in-place effects, and rank-wide agreement. Planning fields are
exact; Dispatch compares the `cu_seqlens`-defined region and semantic dedup sets;
padding is zero; Prefetch and route weights are exact. Expert Forward uses the defined
Dispatch prefix, zero-fills the remaining `NvS` capacity, and is compared with BF16
tolerance; Combine hidden output uses the same tolerance. The undefined Dispatch tail
after the final `cu_seqlens` end is reported but not compared as semantic output.

Run the CPU/fake-backend suite without NPU hardware:

```bash
python -m pytest tests/moonep/python -q
```

Run the Torch-NPU/HCCL reference on four devices:

```bash
python tools/moonep/run_benchmark.py \
  --mode reference \
  --cases tools/moonep/cases/correctness.json \
  --world-size 4 \
  --physical-device-count 4 \
  --output-dir output/moonep-reference
```

Add `--dump-stage-tensors --tensor-preview-elements 8` to a reference or correctness
run to persist every stage boundary. Each rank writes complete CPU snapshots to
`rank_<rank>/tensor_dumps/<stage>/<reference|candidate>/<input|output>.pt`, plus a
matching human-readable `.txt` file containing every value without ellipsis, plus a
JSON manifest with shape, dtype, original device, element count, and the bounded
preview. Workers also write `tensor_dumps/preview.log`; the managed launcher
echoes rank 0's preview after the case. Tensor dumping is rejected in `benchmark` mode
so snapshot copies cannot enter native performance measurements.

Run a one-rank differential case first to isolate installation, capability, and stage
status failures:

```bash
ASCEND_RT_VISIBLE_DEVICES=4 python -m tools.moonep.launcher \
  --mode correctness \
  --cases tools/moonep/cases/correctness.json \
  --case-ids planning-no-dedup \
  --world-size 1 --physical-device-count 1 \
  --install-prefix "$TILEXR_INSTALL_PREFIX" \
  --output-dir output/moonep-correctness-1r
```

Then run the same padded single-route differential case on four physical devices:

```bash
bash scripts/run_moonep.sh --mode correctness --rank-size 4
```

For `reference` and `correctness`, this script enables stage tensor snapshots by
default and reports the directory and `.pt/.json/.txt` file counts. It runs
`planning-no-dedup`; use `--case-id manual-small` for reference-only hand verification. Use
`--tensor-preview-elements COUNT` to control terminal preview length or
`--no-dump-stage-tensors` for summary-only runs.

Pass `--generate-flowcharts` to the script for an opt-in six-stage visual report. The
report module loads each rank's reference `input.pt` and `output.pt`, writes numbered
left-to-right `.mmd` sources to `<result>/<case_id>/flowcharts`, and the script renders
matching `.svg` and 2x `.png` files with Mermaid CLI. Correctness runs deliberately use
the reference role rather than candidate values. The option requires tensor dumps,
rejects benchmark mode, and runs a minimal `mmdc` browser render before launching NPU
work. Both npm/Puppeteer and Python/Playwright CLI parameter forms are supported. The
PyPI CLI and Playwright browser are installed and validated by
`bash scripts/prepare.sh`; the direct browser command remains
`python -m playwright install chromium`.

`--case-id manual-2rank-imbalanced --rank-size 2` provides a compact load-balancing
case: global Expert 0/1 counts are `[4,4]`, initial owner loads are `[8,0]`, and
Planner redistributes execution to `[4,4]` with one remote expert prefetch on rank 1.

`--case-id manual-2rank-dedup-3 --rank-size 2` combines Planner load migration with
duplicate Dispatch routes. Initial expert-owner loads are `[12,4]`; Planner balances
execution to `[8,8]`. Rank 0's first token routes to experts `[4,5,6,7]` on rank 1,
where the reference emits one primary plus three duplicates in one dedup group. The
current registered-workspace URMA candidate rejects this case before Dispatch because
that path does not build dedup metadata.

The script selects devices starting at 0 by default and owns the HCCL NPU socket
range, so the normal command requires no environment exports. Use
`--visible-devices 4,5,6,7` or
`--hccl-npu-socket-port-range 47200-47300` when a shared host needs explicit
isolation.

Reference and correctness modes initialize HCCL on a launcher-reserved rendezvous port
separate from TileXR bootstrap and teardown rendezvous. Their summaries always record
`performance_valid=false`; reference/HCCL work is never included in native timing.

Planning, Dispatch, PrefetchWeight, Combine, and ReduceGrad are native A5 stages.
ExpertForward is not a TileXR native stage or ABI: it is the Torch-NPU compute step
between the native PrefetchWeight and Combine calls. There is no precomputed,
pure-Python, quantized, or native TileXR GMM fallback.
Reports expose native transport correctness separately from performance closure:
`transport_correctness_valid=true` and `transport_performance_valid=false`. Normal runs use
`performance_scope=native_correctness_only`; 16-rank/8-device runs use
`performance_scope=oversubscribed_functional_only`.

Optional route weights use the same replaceable ABI as hidden data:
Dispatch maps `[S,K]` weights to `[NvS]`, forward and backward both consume the
dispatched weights, and Combine maps them back to `[S,K]`. Both weight stages
are native and bit-exact. Hidden transport supports BF16 only; combine reduces
K routes in FP32 and casts once to BF16. Timings remain non-performance evidence
until the precision matrix and msprof profiling gates are completed.

## Environment

Build and install TileXR with the MoonEP options, source the CANN environment,
and make the integration package importable:

```bash
source scripts/common_env.sh
export TILEXR_INSTALL_PREFIX="$PWD/install"
export PYTHONPATH="$PWD/integrations/moonep_torch:$PWD:${PYTHONPATH:-}"
```

The launcher sets `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, a shared
`TILEXR_COMM_ID`, and the planner-group variables for every child. Pass
`--comm-id <rank0-ip:unused-port>` when `127.0.0.1` is not reachable by all
ranks. The native facade, rank worker, and launcher all default the bounded
Planner ready-poll budget to 1,000,000 iterations. After each case, every rank
synchronizes its local NPU stream and joins a TCP completion rendezvous before
destroying the TileXR communicator, so peer windows remain alive until all
Planner kernels have completed. The launcher gives each run a public launch id
and authenticates each case with an HMAC secret that is never sent or written
to artifacts. Teardown is released only after every rank proves local
quiescence. Any case failure is propagated to all ranks; an unquiesced worker
keeps its communicator open for launcher-coordinated termination.

## 8 Physical NPUs

```bash
python tools/moonep/run_benchmark.py \
  --cases tools/moonep/cases/smoke.json \
  --output-dir output/moonep-8p \
  --physical-device-count 8 \
  --ranks-per-device 1 \
  --comm-id 127.0.0.1:10087
```

One rank per device defaults `TILEXR_MOONEP_PLANNER_BLOCK_DIM=64`.

## 16 Logical Ranks on 8 NPUs

```bash
python tools/moonep/run_benchmark.py \
  --cases tools/moonep/cases/smoke.json \
  --output-dir output/moonep-16r-8p \
  --physical-device-count 8 \
  --ranks-per-device 2 \
  --comm-id 127.0.0.1:10088
```

The launcher maps `logical_rank % 8`, defaults the planner block dimension to
32, and records `oversubscribed=true`. This validates logical-rank indexing and
call ordering; it is not 16-device throughput evidence. An explicit planner
block dimension must satisfy `planner_group_size <= blockDim <= 64`.
Before each Planning launch, this artificial two-process-per-device mode uses
the authenticated host rendezvous to align otherwise independently scheduled
Torch processes. The rendezvous runs before device-event timing; Planning data
and device synchronization still use TileXR peer memory.

The current direct Planner is bounded to 64 ranks per planner group. Global and
lane metadata can describe 256P/512P/1024P layouts; a 128-rank lane at 1024P is
represented by two planner groups until the hierarchical Planner is implemented.

The current MoonEP native stages are same-node IPC-only and use TileXR peer windows;
they do not select UDMA even when the core communicator exposes that capability.
Cross-node MoonEP is rejected before stage launch. Core communicator UDMA probing is
best-effort and any initialization failure leaves the IPC path available.

On Ascend950, same-node IPC whitelist selection defaults to PID-only.
`TILEXR_IPC_PID_MODE=pid` can force that mode for diagnostics.

## Artifacts

Each case contains `rank_<rank>/samples.jsonl`, `rank_<rank>/result.json`,
`summary.json`, and `summary.csv`. `summary.json` uses the maximum rank latency
for each measured iteration before computing p50/p90/p99 and global tokens/s.
Aggregation fails when rank case, capability, topology, sample count, or timing
metric metadata differs.

Reference/correctness cases instead contain `rank_<rank>/stages/<case>.<stage>.json`,
optional `rank_<rank>/tensor_dumps/`, `rank_<rank>/result.json`, and `summary.json`.
A failed artifact identifies the stage,
tensor, first mismatch index, expected/actual values, error maxima, valid-region mask,
and mutation failure when applicable. No timing CSV is generated for these modes.
