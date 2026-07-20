# UDMA Grouped AllToAll Dual-Route Design

## Goal

Extend the standalone grouped-fullmesh AllToAll kernel so cross-node traffic
uses both aggregate UDMA routes. Preserve the existing one-peer/one-route
protocol while distributing peers according to the route capacities.

The initial target remains `rankSize = N * 8`, with contiguous groups of eight
ranks representing physical nodes. The physical 2x8, 128 MiB/rank result at
commit `68e3b5f` is the comparison baseline:

- Host mean: 824.632 us
- Host min/max: 822.646 / 827.564 us
- Loop-49 kernel-envelope mean: 822.583 us

## Current Behavior

With `TILEXR_UDMA_ROUTE_POLICY=all` and `TILEXR_UDMA_QP_NUM=4`, cross-node
peers expose two aggregate routes from `/etc/hccl_rootinfo.json`:

- `plane_pg_0`: six ports, represented by QP 0-3 with weight 6
- `plane_pg_1`: two ports, represented by QP 4-7 with weight 2

The grouped kernel currently chooses only the highest-weight QP. Ties preserve
the lowest QP index, so all payload and ready traffic uses QP0 and the two-port
route remains idle.

## Scope

- Change only the standalone grouped AllToAll kernel and its focused tests.
- Keep the existing 35-core fullmesh kernel unchanged.
- Keep one complete peer payload on one route; do not split one peer payload.
- Keep payload, ready signal, and immediate quiet on the same selected QP.
- Keep the receive-side signal count and registered-memory layout unchanged.
- Apply dual-route selection only to cross-node peers.
- Continue to support `8 <= rankSize <= 128` and `rankSize % 8 == 0`.
- Treat each contiguous group of eight global ranks as one node.

Payload striping within each peer is explicitly deferred. It would require two
independent ready signals and receive-side completion of both segments.

## Route Discovery

For each peer, device code reads all QP weights from `UDMAInfo`.

1. The primary QP is the lowest QP index with the maximum weight.
2. The secondary QP is the lowest QP index with the largest weight strictly
   below the primary weight.
3. If no lower nonzero weight exists, secondary equals primary.

This discovers QP0 and QP4 for the current 6-port/2-port topology without
hardcoding either QP index. It also preserves deterministic fallback on
single-route or equal-weight topologies.

## Balanced Peer Assignment

For a cross-node pair:

```text
sourceLocal = sourceRank % 8
targetLocal = targetRank % 8
routeIndex = (sourceLocal + targetLocal) % 8
selectedQp = routeIndex < 6 ? primaryQp : secondaryQp
```

For every source rank and every remote node, six target peers use the primary
route and two target peers use the secondary route. For every destination rank
and every remote node, the eight incoming source ranks also divide 6:2. This
avoids concentrating all secondary-route traffic on two destination cards.

Same-node peers continue to use the primary max-weight QP. Self traffic remains
a local MTE copy and does not use UDMA.

## Data Flow

The sender chooses the peer's QP once before its pass loop. Every pass then
uses that same QP for:

1. `UDMAPutSignalNbiOnQp` payload plus ready signal
2. `UDMAQuietStatusOnQp`

The receiver continues waiting on the same source-rank signal slot and copies
the source-rank payload slot into output. No new signal, ACK, barrier, or
registered-memory region is introduced.

## Error Handling

- Invalid rank dimensions remain rejected by the grouped layout plan.
- Missing secondary route falls back to primary rather than selecting an
  unavailable QP.
- Existing quiet-status and wait-timeout debug records remain authoritative.
- Trace records continue storing the selected QP for every send phase.

## Verification

Focused unit/source tests will cover:

- deterministic primary/secondary selection from weights `{6,6,6,6,2,2,2,2}`
- fallback when all QP weights are equal
- balanced 6:2 row and column counts for every pair of eight-rank nodes
- same-node peers selecting primary
- payload signal and quiet using the same selected QP
- absence of `SyncAll` and generic UDMA route calls

Physical 2x8 validation will rerun 128 MiB/rank with warmup5/repeat50 and GM
trace enabled. Loop49 must show:

- 128 cross-node send events total
- 96 cross-node events on QP0
- 32 cross-node events on QP4
- 112 same-node send events on their primary QP
- 15 unique send and receive peers per rank
- no quiet failure, wait timeout, or output mismatch

Correctness and QP-distribution results are evaluated before performance. Host
mean and loop-49 kernel envelope are compared against the baseline without
assuming that route utilization must improve latency.
