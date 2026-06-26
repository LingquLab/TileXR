# DataAsFlag Epoch-Ordered P2P Design

Date: 2026-06-26

## Background

The current P2P `data_as_flag` transport uses a 512-byte block layout:

```text
[0, 480)     payload
[480, 512)   ready flag area
```

The sender initializes the local UB scratch with ready flags and then writes the
packed `payload + flag` block layout to the peer data-as-flag IPC data window.
The receiver polls each batch by copying one flag value per block and using
`Sum` to decide whether all blocks in the batch are ready.

This has two important limitations in the P2P benchmark:

- The host does not pass a per-iteration value into `data_as_flag`. Measured
  iterations currently rely on clearing the receiver window. If the window is
  cleared only once before a measured loop, later queued kernel launches can see
  stale ready flags from earlier iterations.
- A flag written as part of the same MTE3 transfer as the payload is not a
  sufficient proof that the payload bytes from that same MTE3 transfer are
  already visible. The correctness argument should depend on the order between
  distinct MTE3 writes, not on byte-level visibility within a single MTE3 write.

`memory_consume` already avoids stale synchronization by passing `magic` and
`step` for each launch. `data_as_flag_epoch_ordered` brings the same
per-iteration discipline to data-as-flag while keeping the flag embedded in the
data window.

## Goals

- Add a new P2P transport named `data_as_flag_epoch_ordered`.
- Keep the existing 512-byte data-as-flag block layout unchanged.
- Keep ready information embedded in the data-as-flag window, not in an
  independent sync region.
- Use a per-launch epoch so receiver polling waits for the current iteration
  and does not depend on clearing the window between launches.
- Send payload and flag with ordered, separate MTE3 stages:
  1. MTE3 payload write.
  2. MTE3 flag write with the current epoch.
- Reduce receiver overhead by making the default receive path poll only the last
  flag in each batch as a batch commit marker.
- Keep a strict/debug path that can validate all flags in a batch after the
  commit flag is observed.
- Validate the new transport side-by-side with the existing `data_as_flag`.

## Non-Goals

- Do not change the block layout or payload-per-block ratio.
- Do not increase payload bytes per flag.
- Do not move flags to `SyncCollectives` outer flags or any other independent
  sync area.
- Do not immediately remove the existing `data_as_flag` implementation. Removal
  happens only after the new mode is validated.
- Do not depend on same-MTE3 payload and flag visibility.

## Transport Rollout

The rollout has two phases.

Phase 1 adds a new transport:

```text
data_as_flag_epoch_ordered
```

Suggested aliases:

```text
daf_epoch_ordered
data-as-flag-epoch-ordered
```

The old `data_as_flag` remains available for A/B comparison.

Phase 2 happens after remote validation shows correct data and stable
performance. In that phase, remove or hide the legacy implementation and make
the epoch-ordered implementation use the public `data_as_flag` transport name.

## Epoch Format

The P2P host already creates one `magic` per phase and size, then passes
`step = iteration + 1` for each warmup or measured launch. The new transport
uses the same inputs.

Device-side data-as-flag epoch:

```cpp
uint64_t epoch = (static_cast<uint64_t>(static_cast<uint32_t>(magic)) << 32) |
                 static_cast<uint32_t>(step);
```

The full 64-bit epoch is written into the 32-byte flag area. The default
implementation should duplicate it four times as `uint64_t` values:

```text
flag area: epoch, epoch, epoch, epoch
```

The receiver fast path only needs to compare one `uint64_t`, preferably the
first word in the flag area. The strict path can check all four words and all
blocks in the batch.

## Sender Protocol

For each batch:

```text
MTE3 #1: payload blocks -> peer data-as-flag payload areas
MTE3 #2: flag blocks    -> peer data-as-flag flag areas, value = epoch
```

The sender must not prove payload arrival using a flag from the same MTE3 write
as the payload. The flag is a commit marker only because the flag MTE3 is issued
after the payload MTE3.

Suggested device helper:

```cpp
__aicore__ inline uint32_t DataAsFlagSendEpochOrdered(
    __gm__ uint8_t* dstDataAsFlagGM,
    const __gm__ uint8_t* srcGM,
    uint64_t dataBytes,
    uint64_t epoch,
    AscendC::LocalTensor<uint8_t>& scratch);
```

Implementation shape:

1. Compute `totalBlocks = ceil(dataBytes / 480)`.
2. Split by UB capacity.
3. Copy continuous payload from `srcGM` into a payload scratch area.
4. MTE3-copy payload into block payload regions in `dstDataAsFlagGM`, using a
   destination stride that skips each 32-byte flag area.
5. Prepare a flag scratch area filled with the current epoch.
6. MTE3-copy flags into the block flag regions.
7. Advance to the next batch.

If a direct strided `DataCopyPad` from continuous payload scratch to
`480B payload + 32B gap` GM layout is not accepted by AscendC or performs badly,
the conservative first implementation may keep the existing payload packing
shape but must still issue the flag write as a separate MTE3 stage after the
payload write.

## Receiver Protocol

For each batch:

```text
poll last block flag == epoch
copy batch payload -> dstGM
```

Suggested device helper:

```cpp
__aicore__ inline bool DataAsFlagCheckAndRecvEpochOrdered(
    const __gm__ uint8_t* dataAsFlagGM,
    uint64_t dataBytes,
    __gm__ uint8_t* recvGM,
    uint64_t epoch,
    AscendC::LocalTensor<uint8_t>& scratch,
    bool strict);
```

Implementation shape:

1. Compute `totalBlocks = ceil(dataBytes / 480)`.
2. Split by receive UB capacity.
3. For each batch, compute `lastBlock = processedBlocks + batchBlocks - 1`.
4. Poll the selected word in `lastBlock`'s flag area until it equals `epoch`.
5. If strict mode is enabled, validate the batch flags after the last flag is
   observed.
6. Copy the batch payload from data-as-flag layout into continuous `recvGM`.
7. Advance to the next batch.

The fast path removes the repeated copy of all flags and the vector `Sum` used
by the legacy receiver. The strict path keeps a validation option for testing
the MTE3 order assumption and diagnosing data errors.

## Strict Mode

Strict mode is not the default performance path.

It should be enabled by a host option or environment flag, for example:

```text
TILEXR_DATA_AS_FLAG_STRICT=1
```

After the last block flag equals the current epoch, strict mode checks the whole
batch. A simple first implementation may copy one `uint64_t` flag per block and
compare each value to `epoch`. It does not need to use `Sum`, because the epoch
is a 64-bit token rather than `float(1.0f)`.

Strict mode failure should set a nonzero debug status so the host CSV row reports
the issue through the existing P2P status path.

## Host Changes

The new launcher takes the same launch metadata style as `memory_consume`:

```cpp
void launch_tilexr_data_as_flag_epoch_ordered_p2p_perf(
    uint32_t blockDim,
    void* stream,
    GM_ADDR commArgs,
    GM_ADDR src,
    GM_ADDR dst,
    GM_ADDR debug,
    int32_t srcRank,
    int32_t dstRank,
    uint64_t dstByteOffset,
    uint32_t bytes,
    uint32_t pattern,
    int32_t traffic,
    int32_t magic,
    int32_t step);
```

`LaunchP2PKernel` passes `magic` and `step` to the new transport. Warmup and
measured loops use the existing values:

```text
warmup:  magic = warmupMagic,  step = i + 1
measure: magic = measuredMagic, step = i + 1
```

The new transport does not need per-iteration clear-window calls or the extra
clear barriers currently used for legacy `data_as_flag`. A single initialization
clear per size is still acceptable, mainly to reduce the chance that old data
accidentally equals an epoch during early bring-up. The epoch value should be
large enough that accidental matches are practically impossible.

The transport helper predicates should treat `data_as_flag_epoch_ordered` like
`data_as_flag`:

- Uses IPC peer window.
- Both ranks are active in unidirectional traffic.
- Window size is still `ceil(payload / 480) * 512`.

## Kernel Integration

Add a new P2P kernel beside the legacy one:

```cpp
extern "C" __global__ __aicore__ void tilexr_data_as_flag_epoch_ordered_p2p_perf_kernel(...);
```

The role resolution can reuse `TileXRUdmaDemoResolveDataAsFlagRole`.

The slicing can reuse `TileXRUdmaDemoDataAsFlagSlice` because the layout and
block ownership do not change.

Sender path:

```text
DataAsFlagSendEpochOrdered(peerBase + dstByteOffset + dataAsFlagOffset,
                           src + payloadOffset,
                           sliceBytes,
                           epoch,
                           scratch)
```

Receiver path:

```text
DataAsFlagCheckAndRecvEpochOrdered(localBase + dstByteOffset + dataAsFlagOffset,
                                   sliceBytes,
                                   dst + payloadOffset,
                                   epoch,
                                   scratch,
                                   strict)
```

Debug words should record enough information to distinguish the new mode from
legacy `data_as_flag`, including block count, status, and at least the low 32
bits of the epoch.

## Correctness Argument

Stale flags are avoided because every launch writes and waits for a unique epoch.
The receiver no longer treats a generic ready value from an older launch as
current data.

Payload visibility depends on MTE3 ordering across distinct MTE3 operations:
payload MTE3 is issued before flag MTE3. When the receiver sees the batch commit
flag from the flag MTE3, it can treat the earlier payload MTE3 for the same
batch as visible.

The protocol does not assume that all bytes from a single MTE3 write become
visible at the same time. In particular, it does not use a flag written by the
same MTE3 operation as proof for that MTE3 operation's payload.

The block semantic remains data-as-flag because the flag is still embedded in
the 512-byte data window block.

## Tests

Unit tests:

- `P2PTransportName` returns `data_as_flag_epoch_ordered`.
- Parser accepts the new name and aliases.
- Window size matches legacy `data_as_flag`.
- `P2PTransportUsesIpc` and `P2PTransportBothRanksActive` include the new mode.
- CSV formatting preserves the transport name.

Source guard tests:

- New launcher declaration exists.
- New kernel exists.
- New device helpers exist in `tilexr_data_as_flag.h`.
- New helper signatures include `epoch`.
- Host passes `magic` and `step` to the new launcher.
- Legacy per-iteration clear-window branches do not apply to the new transport.

Remote validation:

- Build `tests/udma` on the Ascend environment.
- Run unit tests.
- Run focused P2P smoke tests for `data_as_flag_epoch_ordered` on 6/7 or other
  healthy cards.
- Run 8B to 64MB unidirectional sweep with `block_dim=1`.
- Compare old `data_as_flag`, new `data_as_flag_epoch_ordered`, `memory`, and
  `memory_consume` CSVs.
- Verify all rows have `status=0` and `errors=0`.
- Run strict mode on a smaller sweep to validate all batch flags.

## Completion Criteria

- `data_as_flag_epoch_ordered` is selectable from the P2P demo CLI.
- The new mode uses per-launch epoch values and does not require per-iteration
  clear-window barriers.
- Sender emits payload and flag in separate ordered MTE3 stages.
- Receiver fast path waits on one batch commit flag and then copies payload.
- Strict mode can validate all flags for debug runs.
- Remote build and tests pass.
- Remote P2P smoke and sweep runs complete with `status=0` and `errors=0`.
- CSV output allows direct A/B comparison with legacy `data_as_flag`.

## Later Migration

After validation, replace the public `data_as_flag` behavior with the
epoch-ordered implementation:

1. Remove or hide the legacy transport from sweeps.
2. Make `data_as_flag` parse to the epoch-ordered implementation.
3. Keep `data_as_flag_epoch_ordered` as a temporary alias if useful.
4. Update docs and plots so the public name is again `data_as_flag`.
5. Remove legacy clear-window workarounds from the data-as-flag host path.
