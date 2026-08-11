# MoonEP Combine V1 Memory Single-Launch Design

## Goal

Replace the existing MoonEP Combine V1 implementation with a single-launch
peer-memory baseline. The baseline keeps the saved duplicate metadata algorithm
but follows the Combine V2 rank/core/peer schedule and receive-side reduction
flow. Combine V2 remains the default complete-flow backend.

## Scope

- Direct AICore binary registration and `rtKernelLaunchWithFlagV2` invocation.
- Ascend 910A5 / Ascend950 with CANN 9.1.
- BF16 hidden input/output and optional FP32 route weights.
- Rank sizes supported by `MoonEpCombineV2RankSizeSupported`.
- One Host launch per Combine call, including hidden and optional weights.
- Memory transport through `CommArgs.peerMems[]` for same-node and cross-node
  mappings.

## Non-Goals

- Preserve the old V1 publish-only or consume-only split-launch behavior.
- Preserve V1 status values or internal kernel ABI.
- Add a topology-only memory/UDMA selector.
- Change Combine V2, Dispatch V2, or the Planner output contract.
- Claim real multi-node validation while only one host is available.

## Public And Runtime Contract

`TileXRMoonEpCombineV1` remains the V1 entry point. Its descriptor is replaced,
without binary-compatibility guarantees, to carry the Planner V3 reverse route
map required by a V2-style push.

- `args.flags` must be `TILEXR_MOONEP_FLAG_NONE`.
- `args.dstLocal` points to `NvS` `int32_t` entries in device memory.
- `dstLocal[expertRecvSlot] = srcRank * NvS + token * K + topk`; `-1` marks an
  unused slot. `plan.dst` is not a valid substitute because it has the opposite
  route direction.
- `plan.dst`, `plan.dupGroups`, `plan.dupLoffs`, and `plan.dupCounts` remain
  required for the legacy duplicate contract and plan validation.
- Hidden tensors are BF16 `[NvS,H]` and `[S,H]`.
- Route weights are either both absent or FP32 `[NvS]` and `[S,K]`.
- A successful kernel stores `0` in `plan.status`; positive values identify
  device validation, timeout, or peer failures.
- The call is asynchronous on the supplied stream and launches exactly one
  registered AICore kernel.

Python selects the backend once when the runtime is constructed:

- unset or `TILEXR_MOONEP_COMBINE_VERSION=2`: Combine V2;
- `TILEXR_MOONEP_COMBINE_VERSION=1`: Combine V1 Memory;
- any other value: fail before communicator initialization.

Benchmark metadata and trace labels report the selected implementation.

## Host Layout

The peer data window is limited by `TileXR::IPC_BUFF_MAX_SIZE`. Host layout uses
checked arithmetic to reserve:

1. a source region containing `NvS` chunk rows and optional route weights;
2. a receive region containing `NvS` chunk rows and optional route weights;
3. magic-tagged per-epoch, per-rank, per-core completion and drain records;
4. per-core failure records.

The largest 32-byte-aligned hidden chunk stride that fits all regions is chosen.
The kernel loops over `ceil(hiddenRowBytes / hiddenChunkBytes)` chunks. A zero
chunk, arithmetic overflow, or a layout beyond the peer window is rejected.

`blockDim` is `MoonEpCombineV2ActiveCoreCount(rankSize)`, bounded by the device
vector-core count. Every Host layout field has an explicit byte unit and is
passed in the exact direct-launch ABI order.

## Kernel Flow

For every hidden chunk, within one kernel launch:

1. Cores cooperatively copy local hidden rows into the source region. Route
   weights are copied during the first chunk.
2. Duplicate groups are partitioned across active cores. Each core marks the
   duplicate source slots, accumulates their BF16 hidden rows into the primary
   row in FP32, and converts the primary row back to BF16 once. Route weights
   are not pre-reduced because every TopK route retains its own weight.
3. A local magic-tagged barrier prevents any core from sending before all
   duplicate groups are complete.
4. Each core follows `MoonEpCombineV2Peer(rank, step, core, rankSize)`. It scans
   `dstLocal` as V2 does and MTE3-copies rows from its source region into the
   target rank's receive region. Duplicate hidden rows are skipped after their
   contribution has been folded into the primary; weight rows are all copied.
5. After payload MTE3 completion, the source core publishes a magic-tagged done
   record to the target peer window. The receiver waits only for the source set
   assigned by the V2 schedule.
6. Receiver cores partition output tokens exactly as V2. BF16 routes are loaded
   from the local receive region, converted to FP32, summed, converted once to
   BF16, and written directly to `hiddenSh`.
7. Optional FP32 route weights are copied by route without arithmetic.
8. A magic-tagged drain barrier completes before source/receive regions are
   reused for the next chunk.

GM-to-UB and UB-to-GM transfers use the CANN 9.1 `DataCopyPad` forms already
compiled by the neighboring kernels. Queue/event dependencies must order MTE2,
Vector, and MTE3 operations. The completion record must never become visible
before its payload MTE3 writes complete.

## Failure Handling

- Host validation returns without launching on invalid descriptors, flags,
  rank topology, core count, peer mappings, or capacity.
- Device validation checks destination encoding and duplicate metadata bounds.
- Each core publishes a failure record; core 0 converges records and writes the
  final plan status.
- All waits are bounded by the existing MoonEP wait-iteration contract.
- Magic-tagged epochs avoid clearing the complete peer window between calls.

## Verification

1. Host layout/schedule tests cover smallest, tail, multi-chunk, overflow, rank
   sizes, core coverage, and control-region bounds.
2. Host/launch tests prove flags are rejected, duplicate pointers are forwarded,
   blockDim follows V2, and one API call performs one kernel launch.
3. Source guards prove the split-launch branches and UDMA WQE/CQ paths are absent
   from V1 and V2 schedule helpers are used.
4. Python tests cover default V2, explicit V1, invalid switch values, dynamic
   metadata, and one V1 FFI call containing both hidden and weights.
5. The target CANN 9.1 build compiles the kernel and all focused CTest targets.
6. On `141.61.49.223`, run V1 benchmark/reference/correctness on one 8-NPU host,
   then rerun the same three modes with the default V2 backend.
7. Multi-node behavior is limited to Host/mock protocol coverage.
