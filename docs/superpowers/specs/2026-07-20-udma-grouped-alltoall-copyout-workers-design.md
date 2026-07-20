# UDMA Grouped AllToAll Configurable Copyout Workers Design

## Goal

Make the grouped AllToAll receive-copy worker count selectable between 16 and
8, then compare both configurations on the same physical 2x8, 128 MiB/rank
workload. The experiment tests whether lower GM copy concurrency improves UDMA
progress enough to offset the additional copy work per core.

The dual-route result at commit `fc38975` is the baseline:

- Host mean: 773.490 us
- Host min/max: 767.636 / 775.511 us
- Loop-49 kernel-envelope mean: 762.332 us

## Configuration

Host mode `testType=8` reads:

```text
TILEXR_DEMO_ALLTOALL_GROUP_COPYOUT_WORKERS=8|16
```

The default is 16. Any other value is rejected before allocation or launch.
The Host launches `16 + copyoutWorkers` AIV blocks and passes the selected
worker count to the grouped kernel.

## Core And Lane Mapping

Cores 0-15 remain send workers. Receive workers begin at core16.

With 16 copyout workers, the existing mapping remains unchanged: receive core
`16 + lane` handles one lane.

With 8 copyout workers, physical receive worker `worker` handles logical lanes
`worker` and `worker + 8`:

```text
core16: lane0, lane8
core17: lane1, lane9
...
core23: lane7, lane15
```

For each group, a worker processes its first lane and then its second lane.
Invalid lanes are skipped through the existing peer mapper. At rankSize=16,
lane15 is invalid, so core23 processes one remote peer while cores16-22 process
two.

The self-copy range is divided by `copyoutWorkers`, producing eight equal
shards in 8-worker mode and preserving sixteen shards in 16-worker mode.

## Protocol

The change is receive scheduling only. It does not alter:

- grouped peer order
- dual-route 6:2 QP selection
- payload or signal offsets
- source-rank receive slots
- ready tokens
- payload-plus-signal QP affinity
- quiet behavior
- ping-pong planes

No barrier, ACK, signal, or registered-memory region is added.

## Trace Mapping

One physical 8-worker core handles two logical peers that share the same
group/pass/phase dimensions. Recording both under the physical core would
overwrite one trace cell.

Receive wait/copy task spans therefore use logical trace core `16 + lane`.
This preserves one task cell per lane and keeps 8-worker traces comparable with
16-worker traces. Whole-kernel and self-copy spans use the actual physical core
`blockIdx`; cores24-31 have no kernel span in 8-worker mode.

Debug errors continue to identify the physical core and include the peer.

## Validation

Focused tests cover:

- Host acceptance of 8 and 16 and rejection of all other values
- block dimensions 24 and 32
- 8-worker lane ownership covering lanes0-15 exactly once
- 16-worker mode preserving one lane per worker
- self-copy partitioning by the configured worker count
- logical trace core `16 + lane` for receive wait/copy
- absence of `SyncAll` and generic UDMA route calls

Physical validation runs both worker counts with:

```text
rankSize=16
input/output=128 MiB per rank
chunk=8 MiB per peer
warmup=5
repeat=50
dual route=6:2 peer distribution
trace=enabled
```

Both runs must pass all 16-rank output checks with no quiet failure or wait
timeout. Report Host mean/min/max, loop-49 kernel envelope, send quiet,
receive wait, receive copy, trace event counts, and QP distribution. Keep 8
workers only if correctness holds and measured performance is competitive with
or better than 16 workers.
