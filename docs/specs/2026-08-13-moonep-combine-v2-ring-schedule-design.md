# MoonEP Combine V2 Selectable Ring Schedule Design

Status: Implemented; hardware validation pending

Date: 2026-08-13

## Goal

Preserve the existing MoonEP Combine V2 send schedule and add a selectable
bidirectional-ring schedule. The active schedule is selected by one
compile-time parameter in the operator, so changing that parameter and
rebuilding is sufficient to switch schedules.

The bidirectional schedule must:

1. support 8P, 16P, 32P, 64P, and 128P;
2. keep the current active-core and step-count contracts;
3. for rank sizes of at least 16, keep cores 0-7 responsible for the absolute
   lower half of ranks and cores 8-15 responsible for the absolute upper half;
4. expand around the source rank in alternating positive and negative ring
   distances;
5. represent the missing Self entry with one invalid peer slot and process
   that slot through the existing Self-copy path; and
6. keep peer selection, receive-step inversion, and grant propagation
   mutually consistent.

## Non-Goals

This change does not:

- delete or alter the existing schedule;
- change the Host API or kernel launch ABI;
- select the schedule at runtime without rebuilding;
- change the number or ownership of QPs;
- change the 3:1 payload distribution between the two QPs owned by a core;
- change grant, done-token, failure-record, or workspace layouts;
- change route selection, payload construction, Self-copy, reduction, CQ, or
  doorbell implementations; or
- guarantee a physical Fullmesh or CLOS path merely from the logical send
  order. Physical paths remain determined by communicator `(peer, qp)` route
  configuration.

## Existing Contracts

The current scheduling dimensions remain unchanged:

| Rank size | Active cores | Step count | Scheduled slots per rank |
| ---: | ---: | ---: | ---: |
| 8 | 8 | 1 | 8 |
| 16 | 16 | 1 | 16 |
| 32 | 16 | 2 | 32 |
| 64 | 16 | 4 | 64 |
| 128 | 16 | 8 | 128 |

For rank sizes greater than 8:

```cpp
activeCoreCount = 16U;
stepCount = rankSize / 16U;
```

For 8P:

```cpp
activeCoreCount = 8U;
stepCount = 1U;
```

Each core continues to own two QPs:

```cpp
sixPortQp = core;
twoPortQp = 16U + core;
```

The mode changes only the `(sourceRank, step, core) -> peer` mapping and the
dependent inverse and grant mappings.

## Schedule Mode

Add the following type to `combine_v2_schedule.h`:

```cpp
enum MoonEpCombineV2ScheduleMode : uint32_t {
    MOONEP_COMBINE_V2_SINGLE_RING = 0U,
    MOONEP_COMBINE_V2_BIDIRECTIONAL_RING = 1U,
};

constexpr uint32_t kMoonEpCombineV2InvalidPeer = UINT32_MAX;
```

Add one compile-time selection point in the active Combine V2 kernel:

```cpp
constexpr TileXRMoonEp::MoonEpCombineV2ScheduleMode
    kCombineV2ScheduleMode =
        TileXRMoonEp::MOONEP_COMBINE_V2_SINGLE_RING;
```

Changing only this initializer and rebuilding switches between the two
schedules. The default is `SINGLE_RING`; set it to
`MOONEP_COMBINE_V2_BIDIRECTIONAL_RING` to select the new order.

All schedule helpers receive the mode explicitly. This keeps Host unit tests
able to evaluate both modes even though the active kernel mode is currently
compile-time fixed:

```cpp
MoonEpCombineV2Peer(sourceRank, step, core, rankSize, mode);
MoonEpCombineV2ReceiveStep(destinationRank, sourceRank, rankSize, mode);
MoonEpCombineV2Successor(sourceRank, step, core, rankSize, mode);
```

## Existing Single-Ring Mode

`MOONEP_COMBINE_V2_SINGLE_RING` executes the current formulas without any
behavioral change. The existing code should be moved into a clearly named
single-ring helper or retained as the default branch of each public schedule
helper.

The following properties remain exactly as they are today:

- rank sizes from 2P through 8P map `peer = core`;
- 16P through 128P use the existing group-based forward/backward order;
- Self is returned as a valid peer and remains in the current final step;
- the current receive-step inverse remains in use; and
- the current step-independent successor remains in use internally for this
  mode, although the public successor helper gains a `step` argument shared
  with the bidirectional mode.

## Bidirectional Mode for 16P Through 128P

For these sizes, define:

```cpp
halfRankCount = rankSize / 2U;
stepCount = rankSize / 16U;
targetHalf = core / 8U;       // 0 for core 0-7, 1 for core 8-15
lane = core % 8U;
ordinal = step * 8U + lane;
sourceHalf = sourceRank / halfRankCount;
sourceCenter = sourceRank % halfRankCount;
```

`targetHalf` is absolute rather than relative to the source rank. Therefore:

- cores 0-7 always target ranks `[0, halfRankCount)`; and
- cores 8-15 always target ranks `[halfRankCount, rankSize)`.

For every source rank, one core group targets the source rank's own half and
the other targets the opposite half.

### Same-Half Ring

The group targeting `sourceHalf` excludes Self and walks alternating positive
and negative distances:

```text
+1, -1, +2, -2, ..., +(H/2), invalid
```

where `H = halfRankCount`.

For `ordinal < H - 1`:

```cpp
distance = ordinal / 2U + 1U;
offset = (ordinal & 1U) == 0U ? distance : -distance;
```

For `ordinal == H - 1`, return `kMoonEpCombineV2InvalidPeer`. This slot
represents Self and is always the final lane of the final step for that
same-half core group.

### Opposite-Half Ring

The group targeting the opposite half covers all `H` positions:

```text
0, +(H/2), +1, -1, +2, -2, ..., +(H/2-1), -(H/2-1)
```

The offset is:

```cpp
if (ordinal == 0U) {
    offset = 0;
} else if (ordinal == 1U) {
    offset = H / 2U;
} else {
    pairOrdinal = ordinal - 2U;
    distance = pairOrdinal / 2U + 1U;
    offset = (pairOrdinal & 1U) == 0U ? distance : -distance;
}
```

### Target Rank

Use signed arithmetic for the offset or an explicit modular helper. Do not
cast a negative offset to `uint32_t` before applying the ring modulus.

Conceptually:

```cpp
targetLocal = Mod(sourceCenter + offset, H);
peer = targetHalf * H + targetLocal;
```

`Mod(value, H)` must return a value in `[0, H)` for both positive and negative
inputs.

### Invalid Slot Location

The invalid slot belongs to the absolute half containing the source rank:

| Rank size | Final step | Lower-half source | Upper-half source |
| ---: | ---: | --- | --- |
| 16 | 0 | core 7 | core 15 |
| 32 | 1 | core 7 | core 15 |
| 64 | 3 | core 7 | core 15 |
| 128 | 7 | core 7 | core 15 |

After replacing the one invalid slot with Self, every source rank visits every
rank exactly once.

## Bidirectional Mode for 8P

8P retains one step and eight active cores. It does not split into two
simultaneous half rings because there are not 16 active cores. The mode
therefore degenerates to one bidirectional ring over all eight ranks:

```text
core 0-7: +1, -1, +2, -2, +3, -3, +4, invalid
```

The invalid `core 7` slot represents Self. After substituting Self, all eight
destinations are covered once.

For 2P through 7P, bidirectional mode falls back to the existing schedule.
This avoids introducing special odd-size diameter and invalid-slot rules for
configurations that are not part of the requested optimization.

## Effective Peer and Self Processing

The raw schedule helper returns `kMoonEpCombineV2InvalidPeer` for the one
bidirectional Self slot. The send loop converts it to local processing:

```cpp
const uint32_t scheduledPeer = MoonEpCombineV2Peer(
    rank_, step, core_, rankSize_, kCombineV2ScheduleMode);

if (scheduledPeer == rank_ ||
    scheduledPeer == TileXRMoonEp::kMoonEpCombineV2InvalidPeer) {
    succeeded = SendSelfStep(rank_);
} else {
    succeeded = SendRemoteStep(scheduledPeer, step);
}
```

Define the effective peer used by schedule reasoning as:

```cpp
effectivePeer = scheduledPeer == kMoonEpCombineV2InvalidPeer ?
    sourceRank : scheduledPeer;
```

The invalid slot does not generate UDMA payload, grant, done, CQ, or doorbell
work. This is safe because:

- it occurs only in the final step;
- there is no next-step grant to publish from it; and
- `WaitInboundDone()` already treats the local source as ready without a done
  token.

`InitLaneStates()` runs before the send loop and must not pass
`kMoonEpCombineV2InvalidPeer` into UDMA lookup. Convert an invalid first
scheduled peer to `rank_` before initializing the lane state. This matters for
16P, where the invalid slot is in step zero, and for 8P. Existing communicator
construction already provides the Self/fallback queue image needed by the
current Self-assigned core.

## Receive-Step Inversion

Changing only the peer formula is incorrect because inbound done-token waits
derive their expected step from `MoonEpCombineV2ReceiveStep()`.

Single-ring mode retains the existing inverse. For bidirectional mode at 16P
through 128P, define:

```cpp
H = rankSize / 2U;
sourceHalf = sourceRank / H;
destinationHalf = destinationRank / H;
sourceLocal = sourceRank % H;
destinationLocal = destinationRank % H;
clockwise = (destinationLocal + H - sourceLocal) % H;
distance = Min(clockwise, H - clockwise);
```

For a destination in the same half:

```cpp
if (distance == 0U) {
    return stepCount - 1U;  // Self is represented by the final invalid slot.
}
return (distance - 1U) / 4U;
```

For a destination in the opposite half:

```cpp
if (distance == H / 2U) {
    return 0U;
}
return distance / 4U;
```

The `H/2` opposite-half diameter is deliberately placed in step zero by the
opposite-half sequence.

For 8P there are no remote step distinctions, so receive step remains zero.
For 2P through 7P, use the existing receive-step behavior.

## Grant Successor

The admission protocol for step `s + 1` requires the source that sent to a
given destination in step `s` to grant the source that will send to that same
destination on the same core in step `s + 1`.

The required invariant is:

```cpp
EffectivePeer(sourceRank, step, core) ==
    EffectivePeer(successor, step + 1U, core);
```

The current single-ring successor remains valid for single-ring mode.

For bidirectional mode, the successor must depend on `step`. Let
`EffectiveOffset(step, core, sourceRank)` be the offset used by the selected
target half, with the invalid same-half slot interpreted as offset zero. Then:

```cpp
H = rankSize / 2U;
sourceHalfBase = (sourceRank / H) * H;
sourceLocal = sourceRank % H;
currentOffset = EffectiveOffset(step, core, sourceRank);
nextOffset = EffectiveOffset(step + 1U, core, sourceRank);

successorLocal = Mod(
    sourceLocal + currentOffset - nextOffset, H);
successor = sourceHalfBase + successorLocal;
```

The successor stays in the same source half. This is required because, for a
fixed absolute target-half core group, sources from the same half share the
same offset family.

The helper is called only when `step + 1U < stepCount`. No successor is needed
after the final step. Grant indices, grant tokens, and grant workspace sizing
remain unchanged.

## Affected Implementation

The intended production-code changes are limited to:

1. `src/moonep/combine_v2/common/combine_v2_schedule.h`
   - add the schedule mode and invalid-peer constant;
   - retain the existing schedule as the single-ring path;
   - add bidirectional peer and offset helpers;
   - add the mode parameter to peer and receive-step helpers;
   - add `step` and mode parameters to the successor helper; and
   - implement the generalized 8P/16P/32P/64P/128P rules.
2. `src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel.h`
   - add the one compile-time mode parameter;
   - pass it to schedule helpers;
   - convert an invalid first peer before lane initialization;
   - dispatch an invalid send peer to `SendSelfStep(rank_)`; and
   - pass `step` to successor calculation.

The experimental untracked
`src/moonep/combine_v2/kernels/tilexr_moonep_combine_v2_kernel_back.h` is not
an active implementation source and is outside the change.

No Host layout or launch changes are required.

## Required Invariants

For every supported rank and both schedule modes:

1. every effective peer is in `[0, rankSize)`;
2. the complete `(step, core)` schedule covers every destination exactly once;
3. Self occurs exactly once;
4. raw single-ring peers are always valid;
5. raw bidirectional peers contain exactly one invalid slot for 8P, 16P, 32P,
   64P, and 128P;
6. bidirectional mode for 2P through 7P contains no new invalid behavior;
7. `ReceiveStep(EffectivePeer(...), sourceRank, ...) == step` for remote peers;
8. the grant successor invariant holds for every non-final step and core;
9. cores 0-7 target the lower absolute half and cores 8-15 target the upper
   absolute half for rank sizes of at least 16; and
10. changing the compile-time mode does not change QP indices, workspace
    bytes, launch arguments, or public ABI.

Self is excluded from the receive-step round-trip assertion because
`WaitInboundDone()` does not wait on a local done token. Its conceptual step is
the final step in bidirectional mode.

## Validation Plan

Update `tests/moonep_combine_v2/unit/test_combine_v2_schedule.cpp` to evaluate
both modes independently.

Required unit coverage:

1. Exact peer tables for representative rank 0 and a nonzero center rank for
   8P, 32P, 64P, and 128P.
2. Exhaustive coverage and uniqueness over every source rank for 8P, 16P,
   32P, 64P, and 128P.
3. Proof that single-ring output is unchanged for every currently supported
   rank size.
4. Exactly one invalid raw bidirectional peer and exactly one effective Self
   peer for each optimized rank size.
5. Exhaustive receive-step inversion for every remote peer.
6. Exhaustive grant successor checks for every non-final step and core.
7. Absolute target-half ownership checks for cores 0-7 and 8-15.
8. A source guard or compile check proving every active schedule call supplies
   the selected mode.
9. Focused Host unit build and test execution.
10. Target CANN kernel compilation.

Hardware acceptance should compare both modes with identical rank placement,
shape, warmup, iteration count, communicator route configuration, and CANN
build. Correctness must pass at each tested size before performance results are
compared. Server synchronization must use Mutagen, and NPU occupancy handling
must follow the repository `AGENTS.md` rules.

## Risks and Boundaries

- A peer-only change would mismatch receive done-token steps and can hang.
- A peer-and-receive-only change would send grants to the wrong next source
  and can hang from step one onward.
- The invalid peer must be consumed before any UDMA table lookup.
- 8P does not provide two half rings; it provides a single bidirectional ring
  using the same selectable mode.
- Logical ordering does not prove physical Fullmesh/CLOS selection. Route
  configuration and hardware profiling are required for that conclusion.
- Performance may differ by rank size even when schedule correctness is
  identical, so both modes remain available for comparative hardware
  validation.

## Recorded Decisions

- Preserve the current schedule as `SINGLE_RING`.
- Add `BIDIRECTIONAL_RING` without changing public APIs.
- Select the mode with one compile-time constant in the operator.
- Generalize the bidirectional schedule to 8P, 16P, 32P, 64P, and 128P.
- Fall back to the existing schedule for 2P through 7P.
- Preserve absolute lower-half/upper-half ownership for the two eight-core
  groups at 16P and above.
- Represent Self with one invalid peer in the final same-half slot.
- Reuse the existing Self-copy path for that invalid slot.
- Keep QP count and ownership unchanged.
- Update receive-step inversion and grant successor together with peer order.
- Require `step` in successor calculation for bidirectional mode.
- Use `SINGLE_RING` as the compile-time default while retaining
  `BIDIRECTIONAL_RING` as a one-line selectable alternative.
