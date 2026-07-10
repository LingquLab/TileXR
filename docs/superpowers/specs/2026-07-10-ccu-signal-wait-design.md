# CCU Signal/Wait and Barrier Design

## Goal

Implement a two-rank synchronization capability over TileXR Direct CCU and wire it into the internal `TileXRComm` CCU backend. The work is intentionally internal first: no public C API is added until the runtime semantics, resource ownership, and hardware smoke are stable.

The implementation order is:

1. Single-direction signal/wait between two ranks.
2. Two-direction barrier built from the same signal/wait machinery.

## Scope

In scope:

- Add internal `TileXRCcuBackend` capability for two-rank signal/wait.
- Reuse the existing Direct CCU runtime lifecycle, resource-window registration, peer allgather, lower-layer install, repository install, mission install, and `rtCCULaunch` submit path.
- Add smoke coverage for `rank0 -> rank1`, `rank1 -> rank0`, and two-rank barrier.
- Keep the feature behind internal/test-only entry points until validated.

Out of scope:

- Public C API such as `TileXRCommSignal`, `TileXRCommWait`, or `TileXRCommBarrier`.
- N-rank barrier.
- Alltoall or general collective backend dispatch.
- Host marker based success criteria.

## Existing Building Blocks

The design reuses these current modules:

- `TileXRCcuBackend`: internal backend owned by `TileXRComm`.
- `TileXRCcuRuntimeSession`: rank/device state, Direct CCU runtime availability, socket/thread allgather.
- `TileXRCcuDirectRuntime`: HCCP/RA/runtime loading, RA ctx resource window, endpoint route collection, peer buffer export.
- `TileXRCcuCollectivePlanner`: lower-layer template generation and direct install attempt preparation.
- `TileXRCcuResourceAllocator`: mission, repository, XN, GSA, CKE, and channel allocation.
- `TileXRCcuBuildBarrierProgram`: existing CCU microcode builder for post/wait style synchronization.
- `TileXRCcuSubmitPreparedTasks`: prepared task submission through `rtCCULaunch`.

## Internal API

Add internal request and plan types under `src/comm/ccu`:

```cpp
enum class TileXRCcuSignalWaitRole {
    Signal,
    Wait,
    SignalAndWait,
};

struct TileXRCcuSignalWaitRequest {
    int peerRank = -1;
    TileXRCcuSignalWaitRole role = TileXRCcuSignalWaitRole::Signal;
    uint32_t timeout = 0;
};

struct TileXRCcuSignalWaitPlan {
    bool ready = false;
    TileXRCcuDirectInstallAttempt attempt;
    std::vector<TileXRCcuTask> submitTasks;
};
```

Add internal backend methods:

```cpp
int PrepareSignalWait(
    const TileXRCcuSignalWaitRequest& request,
    TileXRCcuSignalWaitPlan* plan);

int SubmitSignalWait(
    const TileXRCcuSignalWaitPlan& plan,
    aclrtStream stream,
    TileXRCcuDirectSubmitReport* report);
```

These methods are C++ internal only. They are not declared in `src/include/tilexr_api.h`.

## Signal/Wait Semantics

For two ranks, one rank is the signaler and the other rank is the waiter.

Signal rank:

- Installs a CCU task that posts to the peer rank's notify/wait CKE through the lower-layer channel.
- The task should be post-only and should not wait for the peer.
- `aclrtSynchronizeStream()` returning on the signal rank only proves the signal task has been submitted and completed locally.

Wait rank:

- Installs a CCU task that waits on its local wait CKE.
- The task should not complete until the peer signal arrives.
- `aclrtSynchronizeStream()` returning on the wait rank is the synchronization proof.

The smoke test must verify this by delaying the signal rank and checking that the wait rank's stream synchronize time exceeds the configured threshold.

## Barrier Semantics

The two-rank barrier is built from two opposing signal/wait operations:

- rank0 signals rank1 and waits for rank1.
- rank1 signals rank0 and waits for rank0.

The first implementation should prefer one CCU task per rank containing both post and wait instructions. This avoids a host scheduling gap between separate signal and wait submissions and keeps the barrier semantics close to the device timeline.

If the one-task form exposes hardware ordering issues, the fallback is two prepared tasks per rank: post first, wait second. The fallback must remain internal and be selected only for diagnostics or if hardware behavior requires it.

## Resource Flow

The resource flow matches the current Direct CCU install path:

1. Refresh Direct CCU basic info for the selected die.
2. Decode resource spec from basic info.
3. Allocate mission, repository instruction, XN, CKE, and channel resources.
4. Register the local CCU resource window through RA ctx.
5. Export local resource-window token and endpoint route.
6. Allgather peer resource-window and endpoint data.
7. Exchange peer XN/CKE/channel ownership proof.
8. Build lower-layer transport snapshot and install plan.
9. Build a CCU synchronization program from the selected role.
10. Build repository image and launch package.
11. Install repository, lower-layer resources, and mission/key.
12. Generate submit tasks only after install evidence matches the launch package.

No env override should mutate prepared task fields, peer binding proof, mission key, or instruction ranges.

## Microcode Plan

Signal uses a post-only synchronization instruction. The preferred instruction mode is `SyncCkePostOnly` when the lower-layer proof contains the CKE resources needed for peer notification. `SyncXnPostOnly` can be retained as a diagnostic fallback only if the CKE path is not viable on hardware.

Wait uses the existing local CKE wait encoding. If current `TileXRCcuBuildBarrierProgram` cannot express a pure remote-triggered wait cleanly, add a small dedicated builder such as:

```cpp
int TileXRCcuBuildSignalWaitProgram(
    const TileXRCcuSignalWaitProgramSpec& spec,
    std::vector<TileXRCcuInstr>* program,
    TileXRCcuBarrierProgramReport* report);
```

That builder should still call the existing low-level microcode encoders and should avoid duplicating instruction encoding logic.

## Error Handling

Preparation returns explicit errors for:

- Direct CCU runtime unavailable.
- Rank size not equal to 2.
- Invalid peer rank.
- Missing Direct CCU basic info.
- Resource allocation failure.
- Resource-window registration failure.
- Peer route/token exchange failure.
- Lower-layer install-plan failure.
- Repository/mission install failure.
- Install evidence mismatch.

Submission returns explicit errors for:

- Empty or not-ready plan.
- Null stream.
- Runtime launch failure.
- Mid-batch submit failure.

Reports should include enough task detail for diagnosis: mission id, key, instruction range, argument size, submitted task count, and runtime return code when available.

## Testing

Unit tests:

- Signal/wait request validation.
- Signal-only program generation.
- Wait-only program generation.
- Signal-and-wait program generation.
- Resource allocation shape for one peer route.
- Submit path rejects null stream and empty task list.
- Backend boundary tests verify the feature remains internal and does not appear in public headers.

Smoke tests:

- `rank0 signal -> rank1 wait`.
- `rank1 signal -> rank0 wait`.
- Two-rank barrier.
- Delayed signal rank proves wait rank blocks on CCU completion.
- Timeout wrapping remains enabled around the whole runner.

Expected smoke evidence:

```text
tilexr_ccu_signal_wait prepare ret=0 ... installSucceeded=1 ... submitReady=1
tilexr_ccu_signal_wait submit ret=0 ... submitted=1
tilexr_ccu_signal_wait timing rank=<waiter> syncMs=<value>
tilexr_ccu_signal_wait result passed=1
```

For barrier:

```text
tilexr_ccu_barrier prepare ret=0 ... installSucceeded=1 ... submitReady=1
tilexr_ccu_barrier submit ret=0 ... submitted=1
tilexr_ccu_barrier result passed=1
```

## Validation Gates

Before considering the feature complete:

1. CCU unit tests pass.
2. `tile-comm` builds on the NPU server.
3. No hcomm/HCCL private CCU dependency is introduced.
4. Two-card signal/wait smoke passes in both directions.
5. Two-card barrier smoke passes.
6. Delayed waiter/signal timing proves device-side synchronization rather than host-side gating.

## Open Decisions

No open product decisions remain for the first implementation. Public API shape and N-rank barrier are intentionally deferred until after the internal backend path is validated.

