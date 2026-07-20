# Grouped AllToAll Route-Stage Diagnostic Design

## Goal

Add an opt-in diagnostic mode that measures local traffic, the six-port route,
and the two-port route without overlap. The normal grouped AllToAll algorithm
and its default performance path remain unchanged.

The first validation target is physical 2x8 with 128 MiB input/output per rank,
16 copyout workers, warmup 5, repeat 50, and a 60-second process timeout.

## Stage Schedule

Each logical AllToAll invocation is split into three kernel launches:

1. `local`: self-copy and peers whose `rank / 8` matches the local rank's node.
2. `primary`: cross-node peers selected for the primary QP, currently QP0 and
   the six-port aggregate route.
3. `secondary`: cross-node peers selected for the secondary QP, currently QP4
   and the two-port aggregate route.

The Host synchronizes the stream after each launch and executes the existing
TCP all-rank barrier before launching the next stage. Therefore no rank can
start one route stage while another rank is still using the preceding route.

The route-stage mode is enabled only through a new environment option. Without
that option, the Host performs the current single kernel launch and does not add
stage barriers.

## Kernel Filtering

The grouped kernel receives a route-stage selector. Send and receive workers use
the same predicate for every peer:

- `local` accepts only same-node peers; receive workers also perform self-copy.
- `primary` accepts only cross-node peers whose selected QP is the primary QP.
- `secondary` accepts only cross-node peers whose selected QP is the secondary
  QP.

Peers rejected by the current stage are skipped before signal waiting or UDMA
submission. A peer belongs to exactly one stage, so existing payload slots,
signal slots, tokens, passes, groups, and ping-pong invocation slots remain
unchanged. No ACK or device-wide `SyncAll` is introduced.

The node identity is `rank / 8`, consistent with the current `rankSize = N * 8`
scope.

## Timing And Trace

The Host records one ACL event duration for each stage. The reported staged
kernel time is the sum of the three device durations and excludes TCP barrier
latency.

When tracing is enabled, each stage owns a separate existing-size trace buffer.
This avoids changing or multiplying the trace layout and prevents kernel-span
or task-span overwrites. Each rank writes three binary files with `local`,
`primary`, and `secondary` in their names. Existing trace conversion remains
compatible and can extract loop49 independently for each stage.

For the physical 2x8 run, loop49 must contain:

- `local`: 112 same-node sends plus 16 self-copy tasks across all ranks.
- `primary`: 96 cross-node sends on QP0.
- `secondary`: 32 cross-node sends on QP4.
- no peer duplicated across stages and no invalid QP.

Bandwidth is calculated per source node from the selected stage only:

```text
bandwidth = stage payload bytes /
            (latest send-quiet end - earliest send-put-signal begin)
```

For each node, the primary stage transfers `8 ranks * 6 peers * 8 MiB = 384
MiB`; the secondary stage transfers `8 ranks * 2 peers * 8 MiB = 128 MiB`.

## Correctness And Failure Handling

Output validation runs only after all three stages complete. Together the stages
must cover self, all same-node peers, and all cross-node peers exactly once.

An invalid stage selector is rejected before UDMA work. A kernel error, stream
synchronization error, or TCP barrier failure aborts the staged run. Remote test
processes run under `timeout 60s` so a failed barrier or device wait cannot hold
the cards indefinitely.

## Tests

Host-side unit tests cover:

- peer classification for local, primary, and secondary stages;
- exact and disjoint peer coverage for rank sizes 8, 16, and 128;
- unchanged default launch behavior;
- three launches and three barriers in staged mode;
- invalid environment values.

Kernel source guards require symmetric send/receive filtering and continue to
forbid `SyncAll` in the grouped kernel.

Physical verification uses `/home/pkg/b101/cann` on `141.61.50.31` and
`141.61.49.223`, runs correctness first, then extracts loop49 from each stage
and reports six-port and two-port bandwidth independently.
