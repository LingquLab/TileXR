# TileXR EP Dispatch MVP Design

## Goal

Build a first TileXR-native Expert Parallelism dispatch path that moves MoE token payloads between ranks without depending on `src/mc2`, HCCL windows, shmem, or UDMA. The first version is a peer-memory window implementation based on `CommArgs::peerMems[]` and `SyncCollectives`.

## Reference

The design is informed by `3rdparty/ops-transformer/mc2/moe_distribute_dispatch_v2`, especially its V2 dispatch flow:

```text
AlltoAllDispatch -> SetStatus -> WaitDispatch -> LocalWindowCopy -> SetExpertTokenNums
```

TileXR keeps the same high-level protocol shape but replaces the MC2/HCCL window backend with TileXR peer-memory windows.

## Scope

The MVP supports:

- EP-only dispatch.
- Non-quantized token payloads.
- `x` shaped `[BS, H]`.
- `expertIds` shaped `[BS, K]`.
- Uniform expert sharding where `moeExpertNum % rankSize == 0`.
- One output token for each `(token, topK)` route.
- Output metadata sufficient for a later Combine implementation to map dispatched rows back to original `(srcRank, tokenId, topKId, expertId)`.

The MVP does not support:

- Tensor-parallel AllGather.
- Quantization or dynamic scales.
- Shared experts.
- Active masks.
- Elastic rank shrink.
- Performance info accumulation.
- HCCL window/context APIs.
- UDMA data movement in the first implementation.

## Module Placement

EP is a new TileXR runtime module, not an MC2 operator. New code lives under:

```text
src/ep/
src/include/tilexr_ep.h
tests/ep/
```

The EP library links against `tile-comm` and Ascend runtime libraries. It must not include or link `src/mc2`, `3rdparty/ops-transformer`, shmem, or HCCL window-only helpers.

## Public API

The first host API is intentionally smaller than `aclnnMoeDistributeDispatchV2`.

```cpp
int TileXRMoeEpDispatch(
    void* x,
    int32_t* expertIds,
    TileXRCommPtr comm,
    int64_t bs,
    int64_t h,
    int64_t topK,
    int64_t moeExpertNum,
    void* expandXOut,
    int64_t* expertTokenNumsOut,
    int32_t* epRecvCountsOut,
    int32_t* assistInfoForCombineOut,
    TileXR::TileXRDataType dtype,
    aclrtStream stream);
```

The function launches a TileXR AIV kernel. It returns `TileXR::TILEXR_SUCCESS` on success and an existing TileXR error code on invalid arguments, unsupported dtype, insufficient peer-memory window capacity, missing comm args, or launch failure.

## Data Model

The expert mapping is:

```text
localExpertNum = moeExpertNum / rankSize
dstRank        = expertId / localExpertNum
localExpert    = expertId % localExpertNum
```

`expertIds` must contain values in `[0, moeExpertNum)`. Each input token may be routed to `topK` experts. The maximum number of routed records produced by one rank is `bs * topK`.

## Peer-Memory Window Layout

Each rank has a receive window beginning at:

```text
CommArgs::peerMems[rank] + TileXR::IPC_DATA_OFFSET
```

The MVP uses fixed-size source slots for clarity:

```text
rank receive window
  EpWindowHeader
  srcSlots[rankSize]
    EpSrcSlotHeader
    payload[maxRoutesPerSrc * h * dtypeBytes]
    assist[maxRoutesPerSrc * 4 * sizeof(int32_t)]
```

`maxRoutesPerSrc = bs * topK`.

Each source slot contains only records written by one source rank. This avoids a distributed prefix-sum requirement in the first version. The receiver compacts the fixed slots into `expandXOut`.

The host validates that the required bytes fit within `TileXR::IPC_BUFF_MAX_SIZE` before launching the kernel.

## Record Metadata

Each routed token has one metadata tuple:

```text
assist[record * 4 + 0] = srcRank
assist[record * 4 + 1] = tokenId
assist[record * 4 + 2] = topKId
assist[record * 4 + 3] = expertId
```

The tuple is written into the sender's slot in the receiver window. During `LocalWindowCopy`, the receiver copies these tuples into `assistInfoForCombineOut` in the same order as `expandXOut`.

## Kernel Flow

The TileXR kernel follows the V2 flow with TileXR primitives:

0. `ClearWindow`
   - Clear this rank's receive-window source-slot headers before peers write into them.
   - Publish a cleared flag and wait for all ranks to clear their receive windows.

1. `AlltoAllDispatch`
   - The logical operation can be sliced across `(BS * K)` routes, but the MVP launch uses one AIV block so fixed-slot counters are deterministic.
   - For each route, compute `dstRank` and `localExpert`.
   - Copy `x[tokenId]` into the current rank's fixed source slot inside `dstRank`'s receive window.
   - Write the metadata tuple next to the payload.
   - Accumulate per-destination send counts in the source slot header.

2. `SetStatus`
   - After all route writes for all destinations are complete, publish the source rank's ready flag for the launch.
   - The flag value includes an epoch/magic value to prevent stale-window reads.

3. `WaitDispatch`
   - Each rank waits for all source ranks to publish their launch ready flags.
   - Waits use `SyncCollectives` and its existing magic/value style.

4. `LocalWindowCopy`
   - Read each source slot header.
   - Copy payload records into dense `expandXOut`.
   - Copy metadata records into dense `assistInfoForCombineOut`.
   - Fill `epRecvCountsOut[srcRank]` with the count received from each source rank.

5. `SetExpertTokenNums`
   - Count received records by local expert.
   - Write `expertTokenNumsOut[localExpert]` as counts, not prefix sums, for the MVP.

## Synchronization

The MVP uses the existing first 2MB peer-memory flag area through `SyncCollectives`; EP payloads live after `IPC_DATA_OFFSET`.

Each launch uses a monotonically changing magic value. The MVP host launcher obtains this value with `TileXRCommNextMagic(comm, &magic)` and passes `magic` to the kernel as an explicit launch argument.

The protocol has one ready flag per source rank for the MVP. A source rank publishes the flag only after it has finished writing all destination windows for the launch, so each receiver can safely read its source slots after waiting for every source. More granular token-level flags are unnecessary while fixed source slots are used and the receiver waits for the full source slot to be complete.

## Error Handling

The host API rejects:

- Null input/output pointers.
- Null communicator.
- Missing host or device `CommArgs`.
- `rankSize <= 0`.
- `bs <= 0`, `h <= 0`, `topK <= 0`, or `moeExpertNum <= 0`.
- `moeExpertNum % rankSize != 0`.
- Unsupported dtype.
- Required receive-window bytes greater than `TileXR::IPC_BUFF_MAX_SIZE`.

The kernel treats invalid `expertIds` as a debug error by recording a nonzero debug/status word if a debug buffer is added. The MVP host demo also validates inputs on CPU before copying them to device.

## Build

Add a top-level option:

```cmake
option(TILEXR_BUILD_EP "Build TileXR EP communication library" OFF)
```

When enabled, build `libtilexr-ep.so`. The EP kernel build follows the pattern in `tests/memory/CMakeLists.txt` and uses `bisheng` for the demo kernel artifact.

## Tests

Create a new independent test tree under `tests/ep`.

Unit tests:

- Mapping: `expertId -> dstRank/localExpert`.
- Window layout: byte offsets, alignment, and capacity checks.
- Source guards: EP sources do not mention `src/mc2`, `GetHcclContext`, `TileXRUDMARegister`, `UDMAPut`, or shmem.

Demo test:

- Multi-rank process demo based on `tests/memory/demo`.
- Constructs deterministic `x` and `expertIds`.
- Runs `TileXRMoeEpDispatch`.
- Copies outputs back and validates token values, source counts, local expert counts, and assist tuples.

## Remote Hardware Verification

Hardware validation runs on an explicitly supplied SSH target and remote scratch directory:

```text
TILEXR_EP_REMOTE=<ssh-target>
TILEXR_EP_REMOTE_BASE=<remote-scratch-dir>
```

The verification procedure must deploy a complete working tree, not a partial file copy:

1. Create `${TILEXR_EP_REMOTE_BASE}/TileXR`.
2. Sync the full repository and this branch into that directory.
3. Initialize required submodules.
4. Source `scripts/common_env.sh`.
5. Build and install with `TILEXR_BUILD_EP=ON` and tests enabled.
6. Build `tests/ep`.
7. Run the EP dispatch demo with at least two ranks.
8. Preserve per-rank logs under `tests/ep/logs/`.

## Future TODO: UDMA Backend

Add a second EP backend after the peer-memory MVP is correct:

- Register the EP dispatch receive window with `TileXRUDMARegister`.
- Add a UDMA-capable window layout that keeps signal slots within the registered range.
- Replace peer-memory remote writes with `UDMAPutSignalNbi`.
- Keep the peer-memory backend as fallback when `CommArgs::extraFlag & ExtraFlag::UDMA` is not set.
- Add A5/Ascend950 validation on hardware that supports TileXR UDMA.
- Add multi-AIV route slicing after per-destination record offsets are made deterministic without serial slot counters.

## Open Decisions

No design blockers remain for the MVP.
