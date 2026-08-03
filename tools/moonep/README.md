# TileXR MoonEP Benchmark

This benchmark runs the complete protocol with TileXR bootstrap and no HCCL:

```text
Planning -> Dispatch -> PrefetchWeight -> expert forward -> Combine
         -> Dispatch(saved Plan) -> expert backward -> Combine(saved Plan)
         -> ReduceGrad
```

Planning is the native A5 Planner. Dispatch, PrefetchWeight, Combine, and
ReduceGrad are native ABI stubs in this phase. Reports therefore set
`transport_performance_valid=false`. Normal runs use
`performance_scope=stub_contract_only`; 16-rank/8-device runs use
`performance_scope=oversubscribed_functional_only`.

Optional route weights use the same replaceable ABI as hidden data:
Dispatch maps `[S,K]` weights to `[NvS]`, forward and backward both consume the
dispatched weights, and Combine maps them back to `[S,K]`. The current stubs
exercise these shapes with bounded local copies; real transport results remain
invalid until the four stub capability bits are cleared.
Their timings validate invocation, stream ordering, shapes, and lifecycle only;
they are not TileXR transport-performance results.

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

On Ascend950, TileXR IPC whitelist selection defaults to PID-only for same-node
peers and SuperPod SDID for cross-node peers. `TILEXR_IPC_PID_MODE=pid` or
`TILEXR_IPC_PID_MODE=sdid` forces one mode for diagnostics.

## Artifacts

Each case contains `rank_<rank>/samples.jsonl`, `rank_<rank>/result.json`,
`summary.json`, and `summary.csv`. `summary.json` uses the maximum rank latency
for each measured iteration before computing p50/p90/p99 and global tokens/s.
Aggregation fails when rank case, capability, topology, sample count, or timing
metric metadata differs.
