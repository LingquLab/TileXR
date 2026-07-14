# TileXR Direct CCU 2-Rank AllToAll MVP Design

## Scope

Build a TileXR-owned direct CCU 2-rank AllToAll smoke path for a fixed P2P
payload of 2 MB per rank. The MVP follows the HCCL CCU AllToAll algorithm
shape, but it must not include or call any HCCL or hcomm interface.

The MVP validates:

- Rank 0 sends 2 MB to rank 1's receive buffer.
- Rank 1 sends 2 MB to rank 0's receive buffer.
- The CCU mission performs PreSync, data movement, copy completion wait, and
  PostSync.
- Host-side validation reads local receive buffers and checks peer-specific
  data patterns.

This is a hardware data-plane and synchronization smoke. It is not yet the
LoopEngine validation target.

## Non-Goals

- Do not include HCCL/hcomm headers.
- Do not call HCCL/hcomm launch, channel, resource, or CCU wrapper APIs.
- Do not link new HCCL/hcomm libraries.
- Do not claim LoopEngine is validated by the MVP.
- Do not add a broad public collective API before the smoke path proves the
  direct CCU sequence.

## Reference Model

The local HCCL source under `.tmp/hccl-gitcode` uses this AllToAll shape:

1. `PreSync`: publish output address and token to peers and wait for peer
   readiness.
2. `DoAlltoAll`: write local source slices into peer output through CCU
   channels and wait for copy events.
3. `PostSync`: notify and wait for peers after data movement.

TileXR will use the same algorithm shape and size model, but express every
operation through TileXR-owned direct CCU program builders, resources, and
submit tasks.

## Data Model

The MVP is fixed-size and two-rank:

```text
rankSize = 2
totalBytesPerRank = 2 * 1024 * 1024
memorySliceBytes = 4096
memSlicePerLoop = 8
bytesPerBlock = memorySliceBytes * memSlicePerLoop = 32768
blockCount = totalBytesPerRank / bytesPerBlock = 64
```

`memSlicePerLoop` is capped at 8. The default is 8. The MVP should reject
non-4KB-aligned sizes and any block configuration that would produce a partial
MemorySlice.

The data direction is LocalToRemote:

```text
rank0: rank0.send -> rank1.recv
rank1: rank1.send -> rank0.recv
```

This matches the HCCL-style AllToAll write model and keeps the operation
semantically close to send-to-peer communication.

## Mission Flow

Each rank submits one CCU mission. The mission contains four phases.

### 1. PreSync

The rank confirms that peer output address/token resources are ready before
remote writes begin.

The MVP can implement this with TileXR-owned CKE/checklist operations:

```text
SetCke(local source CKE)
SyncCke(peer pre-sync wait CKE, local source CKE)
ClearCke(local pre-sync wait CKE)
```

PreSync is intentionally separate from final PostSync. This makes readiness
and completion failures distinguishable in logs and mission traces.

### 2. Data Move

The MVP expands the 2 MB transfer into 64 fixed 32 KB blocks at host program
build time:

```text
for block in 0..63:
  localAddr = sendBase + block * 32768
  remoteAddr = peerRecvBase + block * 32768
  length = 32768
  TransLocMemToRmtMem(localAddr, remoteAddr, length)
  ClearCke(copy completion CKE)
```

This represents the HCCL-style `memSlicePerLoop=8` work unit without relying
on LoopEngine support. The block size is deliberately the same as eight 4 KB
MemorySlices.

### 3. Copy Completion Wait

Each block waits for its own transfer completion using the existing memory
copy completion CKE semantics. The same completion CKE can be reused because
the blocks are emitted serially and each block consumes completion before the
next block begins.

### 4. PostSync

After all 64 blocks complete, the rank performs a final two-way completion
barrier:

```text
SetCke(local source CKE)
SyncCke(peer post-sync wait CKE, local source CKE)
ClearCke(local post-sync wait CKE)
```

This is the TileXR-owned `SignalAndWait` completion barrier.

## Program Builder

Add a new builder under `src/comm/ccu`:

```text
tilexr_ccu_alltoall_program.h
tilexr_ccu_alltoall_program.cpp
```

Main types:

```cpp
struct TileXRCcuAllToAll2RankProgramSpec {
    uint64_t localSendAddr;
    uint64_t localSendToken;
    uint64_t remoteRecvAddr;
    uint64_t remoteRecvToken;
    uint64_t bytes;
    uint32_t memorySliceBytes;
    uint32_t memSlicePerLoop;
    uint16_t localGsa;
    uint16_t remoteGsa;
    uint16_t localXn;
    uint16_t remoteXn;
    uint16_t lengthXn;
    uint16_t channelId;
    uint16_t copyCompletionCke;
    uint16_t preSyncLocalWaitCke;
    uint16_t preSyncRemoteNotifyCke;
    uint16_t postSyncLocalWaitCke;
    uint16_t postSyncRemoteNotifyCke;
    uint16_t sourceCke;
};
```

The builder validates:

- `bytes == 2 MB` for the initial smoke.
- `memorySliceBytes == 4096`.
- `1 <= memSlicePerLoop <= 8`.
- `bytes % (memorySliceBytes * memSlicePerLoop) == 0`.
- all XN/GSA/CKE/channel/token/address resources are non-zero.

The builder emits one instruction stream:

```text
PreSync instructions
64 * 32KB LocalToRemote copy block instructions
PostSync instructions
Finish instruction
```

## Planner and Orchestrator

Add a testing/private direct CCU path first:

```text
TileXRCcuCollectivePlanner::PrepareDirectCcuAllToAll2RankInstallAttempt
TileXRCcuRunDirectAllToAll2RankInstallAttempt
BuildDirectAllToAll2RankLaunchPackage
```

The planner reuses the existing direct CCU lower-layer path:

- direct runtime session
- driver adapter
- endpoint route/channel install
- repository install
- mission/task build
- prepared task submit

It should allocate enough CKE resources for separate pre-sync, copy-completion,
post-sync, and source CKE use. If the first implementation must reuse the
existing one-resource allocator shape, the resource mapping must be explicit in
trace output and unit tests.

## Smoke Test

Extend `tests/ccu/ccu_tilexr_direct_smoke_probe.cpp` with an opt-in mode:

```text
TILEXR_CCU_DIRECT_SMOKE_ALLTOALL=1
TILEXR_CCU_ALLTOALL_BYTES=2097152
TILEXR_CCU_ALLTOALL_MEM_SLICE_PER_LOOP=8
```

The runner should:

1. allocate local send and receive buffers on each rank;
2. fill send with a rank-specific byte pattern;
3. fill receive with a sentinel;
4. prepare the alltoall mission;
5. wait until both ranks are submit-ready;
6. submit the prepared task and synchronize the stream;
7. wait for both ranks to report done;
8. copy local receive buffer back to host;
9. verify it equals the peer rank pattern.

Expected passing evidence:

```text
tilexr_ccu_alltoall config rank=<r> bytes=2097152 memSlicePerLoop=8 blockCount=64
tilexr_ccu_alltoall submit ... submitRet=0 syncRet=0
tilexr_ccu_alltoall result passed=1 rank=<r> mismatches=0
```

All hardware runs must use an outer `timeout`.

## Unit Tests

Add focused tests before hardware validation:

- program builder rejects invalid `memSlicePerLoop > 8`;
- program builder rejects non-4KB-aligned size;
- 2 MB generates exactly 64 copy blocks;
- PreSync instructions appear before the first copy block;
- PostSync instructions appear after the last copy block;
- no HCCL/hcomm symbols are introduced into `src/comm`.

The dependency guard remains mandatory:

```bash
bash tests/ccu/check_tile_comm_no_hcomm_deps.sh build/src/comm/libtile-comm.so
```

## LoopGroup Follow-Up

After the MVP passes, replace host-expanded 64-block emission with a TileXR-owned
LoopGroup implementation:

- add TileXR LoopCtx/LoopGroup encoding;
- add LoopEngine allocation and lifecycle tracking;
- add GoSize calculation equivalent to the HCCL model:
  - `addrOffset`
  - `loopParam`
  - `parallelParam`
  - `residual`
- keep the smoke API and validation unchanged;
- add evidence that LoopEngine, not host expansion, performed the 2 MB
  MemorySlice traversal.

The follow-up must be a separate change so the MVP result cannot be mistaken
for LoopEngine validation.

## Open Risks

- TileXR currently lacks a first-class LoopGroup/LoopEngine encoder.
- Current direct memory-copy builder is single-copy oriented; the alltoall
  builder must avoid accidental resource overlap across 64 copy blocks.
- PreSync and PostSync require enough independent CKE/checklist resources to
  keep readiness and completion distinguishable.
- LocalToRemote validation writes peer memory, so host-side checks must read
  each rank's local receive buffer after both ranks complete.
