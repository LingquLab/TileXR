# Grouped AllToAll 1024-Rank Support Design

## Goal

Extend the existing grouped full-mesh AllToAll path from its current 128-rank
algorithm limit and 256-rank TileXR communication limit to 1024 ranks.

The first version preserves the current transport architecture and is validated
with static/unit coverage at 1024 ranks plus a physical 2x8 regression. A real
1024-rank deployment is deferred until a larger environment is available.

## Scope

- Raise `TILEXR_MAX_RANK_SIZE` from 256 to 1024.
- Raise the grouped AllToAll rank limit from 128 to 1024.
- Keep the existing requirement that `rankSize` is a multiple of 8.
- Keep 16 peers per group: eight positive and eight negative ring distances.
- Keep one full-mesh UDMA peer/QP entry per remote rank and the existing QP
  count and route-selection behavior.
- Keep the existing double registered receive planes and the 1 GiB registered
  memory limit.
- Update rank-sized host/device structures and boundary checks through the Comm
  initialization, UDMA registry, QP metadata, and grouped kernel launch path.

Other collective algorithms are not redesigned. They inherit the larger
`CommArgs` ABI and rank-array capacity but receive no new scale optimization.

## Scheduling

At 1024 ranks, each rank communicates with 1023 peers in 64 groups. The existing
schedule remains unchanged:

- group width: 16 peers
- lanes 0-7: increasing positive distance
- lanes 8-15: increasing negative distance
- the diameter peer is emitted only once
- the final group skips invalid lanes

Every peer must appear exactly once for each source rank, and the schedule must
remain reciprocal between peers.

## Capacity And Compatibility

Increasing the global maximum enlarges fixed-size structures. In particular,
`CommArgs::sendCountMatrix` grows to 1024 x 1024 `int64_t` entries (8 MiB), and
the peer-memory, magic, and UDMA registry arrays grow linearly. This is accepted
for the first version.

The UDMA transport retains its existing O(rankSize^2) metadata exchange and
full-mesh QP construction. Integer arithmetic used for allocation and indexing
must use checked `size_t` multiplication where sizes are derived from rankSize.

Raise the grouped trace buffer from 8 MiB to 128 MiB. This covers 1024 ranks,
64 groups, four passes, 50 iterations, and 64 traced cores (approximately
118 MiB with the current six-phase task-span layout). Larger combinations still
reject tracing with the existing explicit capacity error. Trace capacity does
not limit execution when tracing is off.

## Validation

Unit tests will cover rank sizes 8, 128, 256, 512, and 1024 and verify:

- accepted and rejected rank-size boundaries
- expected group count, including 64 groups at 1024 ranks
- every non-self peer appears exactly once
- no invalid or duplicate peers are scheduled
- reciprocal peer scheduling
- registered-memory planning remains within 1 GiB for a 128 MiB per-rank
  payload and rejects oversized layouts
- Comm and UDMA registry structures accept rank 1023 and reject rank 1024
- the 128 MiB trace layout accepts 64 groups, four passes, and 50 iterations
  while rejecting the first larger unsupported layout

After local and remote builds pass, the current CANN b101 physical 2x8 test will
run with 128 MiB per rank, warmup 5, repeat 50, and a 60-second process timeout.
All 16 ranks must complete with correct output. Performance is recorded only as
a regression reference; this change is primarily a scale-capacity change.
