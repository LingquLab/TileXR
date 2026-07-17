# Full-Mesh 16:0 Trace Experiment Design

## Goal

Run the existing physical 2x8 full-mesh trace with core16 sending all remote
payload and core17 sending no payload while still publishing its completion.

## Scope

- Change the remote-send split from 12:4 to 16:0.
- Keep core16 on the max-weight QP and core17 on the min-weight QP.
- Keep core17 active so it publishes `segmentDone` for every remote peer.
- Keep ready publication on core16 after both segment completion tokens arrive.
- Keep direct registered-memory ACK and the existing trace format unchanged.
- Make the change on a separate experimental branch.

## Implementation

Set `TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_PRIMARY_SHARD_END` to `16U` and
`TILEXR_UDMA_DEMO_BIGDATA_REMOTE_SEND_SECONDARY_AGGREGATOR` to `15U`.
The existing segment-range helper then gives core16 all 16 copy shards and
core17 an empty `[16, 16)` range. Core17 skips `UDMAPutNbiOnQp` because
`segmentBytes == 0`, but still writes its local segment-done token.

## Validation

Build and run source/layout tests on both hosts, then run physical 2x8 with
128 MiB input/output per rank, one pass, repeat50, and full-mesh GM tracing.
Require 16 successful ranks, sixteen 8 MiB trace binaries, no core17
`data-put` events, and eight 8 MiB core16 `data-put` events per rank in
iteration 49.
