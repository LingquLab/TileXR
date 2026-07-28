# TileXR Direct CCU AllToAll, Loop Reuse, and Signal/Wait Design

## Status

This document describes the direct CCU implementation in the current TileXR
tree. It replaces the original fixed two-rank MVP proposal.

The implemented test surface provides:

- a two-rank LocalToRemote AllToAll long mission;
- a runtime-sized AllToAll mesh for 2 through 64 ranks;
- repeated submission of one installed mission;
- internal two-rank signal, wait, and signal-and-wait plans;
- an opt-in hardware smoke runner with data and synchronization checks.

The implementation is TileXR-owned. Active code does not include, call, or
link HCCL or hcomm private CCU interfaces. HCCL topology files and captured
microcode may be used as behavior and topology references only.

## Scope and Boundaries

The direct CCU path is currently an internal and test-oriented capability.
AllToAll preparation entry points used by the smoke probe are compiled under
`TILEXR_CCU_TESTING`. Signal/wait is exposed through the internal
`TileXRCcuBackend` C++ interface. No public collective or signal/wait C API
is added to `src/include/tilexr_api.h`.

The current implementation uses host-expanded 32 KB copy blocks. Loop reuse
means repeated submission of one prepared CCU mission; it does not mean that a
CCU LoopEngine or LoopGroup traverses the payload.

Hardware validation is narrower than the accepted software shape:

- the mesh builder and planner accept 2 through 64 ranks;
- the smoke probe accepts 128 KB or 2 MB per mesh destination chunk;
- the two-rank smoke accepts 2 MB or 8 MB payloads;
- the maintained validation in this change covers 2-rank and 4-rank 2 MB
  runs on Ascend950PR.

## Components

The feature reuses the direct CCU stack:

- `TileXRCcuRuntimeSession`: rank, device, socket exchange, runtime and RA
  lifecycle, resource-window registration, and endpoint allgather.
- `TileXRCcuCollectivePlanner`: endpoint import, peer resource exchange,
  topology-aware route construction, and install-attempt preparation.
- `TileXRCcuResourceAllocator`: mission, repository, XN, GSA, CKE, channel,
  and source-CKE allocation.
- `tilexr_ccu_alltoall_program`: two-rank and mesh microcode builders and
  mesh binding validation.
- `tilexr_ccu_signal_wait_program`: Signal, Wait, and SignalAndWait program
  generation.
- `TileXRCcuDirectOrchestrator`: launch package, repository image, lower
  layer install, mission install, and prepared task generation.
- `run_tilexr_ccu_direct_smoke.sh` and
  `ccu_tilexr_direct_smoke_probe.cpp`: opt-in hardware execution and result
  validation.

## Common Prepare and Submit Lifecycle

All modes use the same ownership sequence:

1. Initialize the direct CCU runtime for the selected logical and physical
   device.
2. Refresh and decode CCU basic resource information for the install die.
3. Register the local CCU resource window through RA ctx.
4. Register or import operation buffers and exchange endpoint metadata.
5. Resolve peer EIDs and ports from `/etc/hccl_rootinfo.json` and its
   referenced topology file.
6. Allocate mission, repository, XN, GSA, CKE, source CKE, and channel ranges.
7. Exchange peer-local XN/CKE ownership and build verified channel routes.
8. Install lower-layer PFE, channel, and jetty contexts.
9. Build and install the instruction repository, mission, key, and task
   windows.
10. Expose prepared tasks only when required install surfaces are verified.
11. Submit through `rtCCULaunch` and wait through
    `aclrtSynchronizeStreamWithTimeout`.

One verified endpoint jetty may be shared by multiple logical channels. A
channel must not reference an invented jetty context. Peer notification uses
the peer's exchanged local wait CKE and channel-bound remote XN.

Topology resolution selects CTP when the topology edge advertises `UB_CTP`;
otherwise it selects RTP. `TILEXR_CCU_DIRECT_FORCE_TP_TYPE=ctp|rtp` is a
diagnostic override.

## Data Layout

### Two-Rank Long Mission

Each rank owns one send buffer and one receive buffer:

```text
rank 0 send[bytes] -> rank 1 receive[bytes]
rank 1 send[bytes] -> rank 0 receive[bytes]
```

The program requires a nonzero 4 KB-aligned payload. The program builder
accepts `memSlicePerBlock` in `[1, 8]`; the maintained smoke path requires
eight 4 KB memory slices per block:

```text
blockBytes = 4096 * 8 = 32768
blockCount = bytes / 32768
```

The current long-mission smoke defaults to 2 MB, so it emits 64 copy blocks.

### Runtime-Sized Mesh

For rank count `N` and per-destination chunk size `C`, each rank allocates
`N * C` bytes for both source and destination.

```text
source[targetRank][chunkOffset]
destination[sourceRank][chunkOffset]
```

For every remote peer, rank `r` writes:

```text
local source[peerRank] -> peer destination[r]
```

The self chunk uses local memory-to-memory transfer through local MS:

```text
local source[r] -> local destination[r]
```

The mesh requires `2 <= N <= 64`, exactly `N - 1` unique peer routes, and
a chunk size divisible by 32 KB.

## Two-Rank Long-Mission Microcode

The two-rank launch package allocates three synchronization resources:

1. copy and loop-marker route;
2. PreSync address/token route;
3. reserved post route.

The installed smoke program intentionally enables PreSync and disables a
separate PostSync and finish instruction. Completion of every copy block is
already consumed in-order before stream completion.

### PreSync and Loop Marker

PreSync publishes three values:

- SQE argument zero loop marker with mask `0x1`;
- local receive address with mask `0x2`;
- local receive token with mask `0x4`.

The receiver waits for mask `0x7`. The marker instruction sequence is:

```text
LoadSqeArgsToX(localMarkerXn, arg0)
SyncXn(remoteMarkerXn, localMarkerXn, channel, notifyCke, 0x1)
```

Address and token use `LoadImdToXn` plus `SyncXn`. The final PreSync
`SetCke` wait consumes all three presence bits. This prevents an old CKE
arrival from being accepted as the current loop.

### Copy Blocks

Each 32 KB block emits a seven-instruction LocalToRemote memory-copy program.
The completion CKE is consumed before the next block reuses it. For `B`
blocks, the current long mission contains:

```text
instructions = 7 + 7 * B
```

The leading seven instructions are marker/address/token PreSync and its wait.
For 2 MB, `B = 64` and the mission contains 455 instructions.

## N-Rank Mesh Microcode

Let:

```text
P = N - 1                         # remote peer count
G = ceil(P / 16)                  # grouped completion CKE count
B = chunkBytes / 32768            # blocks per destination chunk
```

Each peer has one logical synchronization resource and one channel. Shared
local address, token, and length XNs are reused across peer channels. Remote
completion bits are grouped in sets of at most 16 because one CKE mask is
16 bits.

### Mesh PreSync

The program loads the local receive address and token once, initializes the
source CKE, then posts both values to every peer. It waits once per peer.

```text
preSyncInstructions = 3 + 3 * P
```

Mesh PreSync does not use the two-rank SQE loop marker.

### Mesh Copy

For each block:

- every remote peer contributes six transfer instructions;
- the self chunk contributes nine local-MS instructions;
- one completion wait is emitted for each group of at most 16 remote peers.

```text
copyInstructionsPerBlock = 6 * P + 9 + G
```

Remote peer ordinal `i` sets bit `i % 16` in completion CKE group
`i / 16`. The builder validates that completion CKEs are unique and do not
overlap the source CKE.

### Mesh PostSync and Total Size

After all blocks, every peer receives one `SyncCke` completion notification
and every local peer wait CKE is consumed. One finish instruction closes the
program.

```text
postSyncInstructions = 2 * P
totalInstructions =
    (3 + 3 * P) +
    B * (6 * P + 9 + G) +
    2 * P +
    1
```

The resource request uses:

- `P` synchronization resources and channels;
- `P` peer local-wait CKEs;
- `P` remote-notify CKEs;
- one source CKE plus `G` grouped completion CKEs;
- at least three local and three remote XNs;
- two local GSAs for the self copy.

The builder performs a second binding validation pass over the encoded
program before repository installation.

## Loop Reuse

`TILEXR_CCU_ALLTOALL_LOOP_COUNT` controls repeated submission and defaults
to one. The probe accepts values from 1 through 1024.

Preparation, buffer registration, lower-layer installation, repository
installation, and mission installation occur once. Each loop:

1. fills source data with a rank-and-loop-specific pattern;
2. resets destination data to a loop-specific sentinel;
3. enters a host ready gate using `phase=loopIndex`;
4. submits the same prepared task and synchronizes the same stream;
5. validates the complete destination buffer;
6. enters a host done gate with the local result.

For the two-rank long mission, each loop also:

- writes a rank-and-loop marker to prepared task argument zero;
- reads the peer marker from the remote XN after stream completion;
- requires the exact expected marker before accepting the data result.

The marker format is:

```text
0x4343554c00000000 | (rank << 16) | loopIndex
```

For the mesh path, the current program has no SQE marker. Loop identity is
validated by source/destination data patterns and loop-specific host gates.
The probe additionally requires mission id, key, instruction range, task
shape, XN, CKE, and channel resources to remain identical after every loop
and prints `stableResources=1`.

This is mission reuse, not LoopEngine execution.

## Signal/Wait and Barrier

Signal/wait is an internal two-rank backend feature:

```cpp
TileXRCcuBackend::PrepareSignalWait(...)
TileXRCcuBackend::SubmitSignalWait(...)
```

`PrepareSignalWait` rejects any rank size other than two and requires the
other rank as `peerRank`. It allocates one synchronization resource and
uses the same direct runtime, endpoint exchange, lower-layer install,
repository, mission, and submit lifecycle as AllToAll.

### Roles

`TileXRCcuSignalWaitProgramRole` has three roles:

- `Signal`: reserve prelude, source CKE initialization, peer `SyncCke`,
  and finish. The emitted program has five instructions.
- `Wait`: one local CKE wait instruction. The allocated repository window
  may be larger, but the installed task window is reduced to the actual
  emitted program.
- `SignalAndWait`: Signal followed by a local wait and finish. The emitted
  program has six instructions.

The Signal role proves local post completion. The Wait role proves that the
peer signal reached its local wait CKE. SignalAndWait forms a two-rank
barrier when both ranks run it.

### Smoke Role Selection

With `TILEXR_CCU_DIRECT_SMOKE_SIGNAL_WAIT=1`,
`TILEXR_CCU_DIRECT_SMOKE_SIGNAL_RANK` selects the Signal rank (default 0);
the other rank is Wait.

With `TILEXR_CCU_DIRECT_SMOKE_BARRIER=1`, both ranks use SignalAndWait. The
barrier flag also enables the generic barrier-program override, so callers
should set `TILEXR_CCU_DIRECT_BARRIER_MODE` explicitly. The maintained CKE
form uses `sync_cke`.

`TILEXR_CCU_DIRECT_SMOKE_DELAY_RANK` and
`TILEXR_CCU_DIRECT_SMOKE_PRE_SUBMIT_DELAY_MS` can delay one rank before
submit. Delaying the signaler provides timing evidence that the waiter's
stream blocks on the device-side event rather than a host marker.

## Hardware Runner Safety

`tests/ccu/run_tilexr_ccu_direct_smoke.sh` is safe by default. It exits
without touching ACL or NPU runtime unless:

```text
TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1
```

When enabled, the runner:

- checks `npu-smi info`;
- rejects selected busy or unhealthy devices by default;
- compiles the private smoke probe against `libtile-comm.so`;
- launches one rank process per selected device;
- installs CCU resources;
- submits real CCU tasks only when
  `TILEXR_CCU_DIRECT_SMOKE_SUBMIT=1`;
- enforces per-process and stream timeouts;
- validates result counts before reporting success.

`TILEXR_CCU_SMOKE_ALLOW_UNHEALTHY_NPU=1` permits Warning/Alarm devices but
still rejects busy devices. It must be used only with explicit authorization.

## Usage

The examples assume `tile-comm` is already built and use the b110 CANN
package on the validation server.

### Two-Rank AllToAll With Loop Reuse

```bash
cd /home/TileXR
source /home/pkg/b110/cann-9.1.0/set_env.sh

TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1 \
TILEXR_TILE_COMM_LIB=/home/TileXR/build_ccu_merge_abe4bd38/src/comm/libtile-comm.so \
TILEXR_CCU_SMOKE_DEVICES=6,7 \
TILEXR_CCU_RANK_SIZE=2 \
TILEXR_CCU_DIRECT_SMOKE_ALLTOALL=1 \
TILEXR_CCU_DIRECT_SMOKE_ALLTOALL_LONG_MISSION=1 \
TILEXR_CCU_DIRECT_SMOKE_SUBMIT=1 \
TILEXR_CCU_ALLTOALL_BYTES=2097152 \
TILEXR_CCU_ALLTOALL_MEM_SLICE_PER_LOOP=8 \
TILEXR_CCU_ALLTOALL_LOOP_COUNT=10 \
TILEXR_CCU_SMOKE_TIMEOUT=180 \
TILEXR_CCU_DIRECT_SUBMIT_TIMEOUT=12000 \
timeout 220s bash tests/ccu/run_tilexr_ccu_direct_smoke.sh
```

Success requires 20 passing loop results, 20 matched peer markers, and zero
data mismatches.

### Four-Rank Mesh

```bash
cd /home/TileXR
source /home/pkg/b110/cann-9.1.0/set_env.sh

TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1 \
TILEXR_TILE_COMM_LIB=/home/TileXR/build_ccu_merge_abe4bd38/src/comm/libtile-comm.so \
TILEXR_CCU_SMOKE_DEVICES=0,1,2,3 \
TILEXR_CCU_RANK_SIZE=4 \
TILEXR_CCU_DIRECT_SMOKE_ALLTOALL=1 \
TILEXR_CCU_DIRECT_SMOKE_ALLTOALL_MESH=1 \
TILEXR_CCU_DIRECT_SMOKE_SUBMIT=1 \
TILEXR_CCU_ALLTOALL_BYTES=2097152 \
TILEXR_CCU_ALLTOALL_MEM_SLICE_PER_LOOP=8 \
TILEXR_CCU_ALLTOALL_LOOP_COUNT=1 \
TILEXR_CCU_SMOKE_TIMEOUT=240 \
TILEXR_CCU_DIRECT_SUBMIT_TIMEOUT=12000 \
timeout 280s bash tests/ccu/run_tilexr_ccu_direct_smoke.sh
```

For loop count one, success requires four passing results, zero data
mismatches, and `stableResources=1` on every rank.

### One-Way Signal/Wait

```bash
cd /home/TileXR
source /home/pkg/b110/cann-9.1.0/set_env.sh

TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1 \
TILEXR_TILE_COMM_LIB=/home/TileXR/build_ccu_merge_abe4bd38/src/comm/libtile-comm.so \
TILEXR_CCU_SMOKE_DEVICES=6,7 \
TILEXR_CCU_RANK_SIZE=2 \
TILEXR_CCU_DIRECT_SMOKE_SIGNAL_WAIT=1 \
TILEXR_CCU_DIRECT_SMOKE_SIGNAL_RANK=0 \
TILEXR_CCU_DIRECT_SMOKE_DELAY_RANK=0 \
TILEXR_CCU_DIRECT_SMOKE_PRE_SUBMIT_DELAY_MS=1000 \
TILEXR_CCU_DIRECT_SMOKE_SUBMIT=1 \
TILEXR_CCU_SMOKE_TIMEOUT=120 \
TILEXR_CCU_DIRECT_SUBMIT_TIMEOUT=12000 \
timeout 150s bash tests/ccu/run_tilexr_ccu_direct_smoke.sh
```

Rank 0 is Signal and rank 1 is Wait. Reverse the direction by setting
`TILEXR_CCU_DIRECT_SMOKE_SIGNAL_RANK=1`.

### Two-Rank Barrier

```bash
cd /home/TileXR
source /home/pkg/b110/cann-9.1.0/set_env.sh

TILEXR_RUN_CCU_DIRECT_SMOKE_PROBE=1 \
TILEXR_TILE_COMM_LIB=/home/TileXR/build_ccu_merge_abe4bd38/src/comm/libtile-comm.so \
TILEXR_CCU_SMOKE_DEVICES=6,7 \
TILEXR_CCU_RANK_SIZE=2 \
TILEXR_CCU_DIRECT_SMOKE_BARRIER=1 \
TILEXR_CCU_DIRECT_BARRIER_MODE=sync_cke \
TILEXR_CCU_DIRECT_SMOKE_SUBMIT=1 \
TILEXR_CCU_SMOKE_TIMEOUT=120 \
TILEXR_CCU_DIRECT_SUBMIT_TIMEOUT=12000 \
timeout 150s bash tests/ccu/run_tilexr_ccu_direct_smoke.sh
```

## Validation and Evidence

Source-level coverage checks:

```bash
python3 -m unittest \
  tests.ccu.test_tilexr_ccu_alltoall_program \
  tests.ccu.test_tilexr_ccu_barrier_program \
  tests.ccu.test_tilexr_ccu_direct_orchestrator \
  tests.ccu.test_tilexr_ccu_direct_smoke_probe \
  tests.ccu.test_tilexr_ccu_direct_smoke_runner \
  tests.ccu.test_tilexr_ccu_lower_layer_plan_builder \
  tests.ccu.test_tilexr_ccu_resource_allocator \
  tests.ccu.test_tilexr_ccu_signal_wait_program

bash tests/ccu/check_tile_comm_no_hcomm_deps.sh \
  build_ccu_direct/src/comm/libtile-comm.so
```

Maintained hardware evidence for the current change:

- 2-rank, devices 6 and 7, 2 MB long mission: both ranks returned
  `syncRet=0`, peer markers matched, and `mismatches=0`.
- 4-rank, devices 0 through 3, 2 MB mesh: all ranks returned `syncRet=0`,
  `mismatches=0`, and `stableResources=1`.

Signal/wait and barrier are implemented and covered by source-level tests.
They require their own hardware run before making a validation claim for a
new server or CANN/driver combination.

## Failure Diagnostics

On a timeout, the probe reports:

- rank and loop index;
- submit and stream synchronization return codes;
- mission id, key, start/end/current instruction;
- local and remote XN values;
- local wait, remote notify, source, and completion CKE values;
- first and last data mismatch offsets and affected 32 KB blocks.

All ranks stopping on the same wait instruction indicates a route, remote XN,
or CKE synchronization problem. A zero mismatch count before stream
completion does not prove success because destination validation has not run.

## Remaining Limitations

- AllToAll remains an internal/test-only direct CCU path.
- The mesh software contract reaches 64 ranks, but hardware validation in
  this change covers four ranks.
- The smoke probe supports only selected payload sizes even though the
  builders accept aligned sizes.
- Mesh loop reuse currently has no device generation marker.
- LoopEngine and LoopGroup are not implemented.
- Signal/wait and barrier remain two-rank only.
- Cleanup after timeout may be skipped by the smoke probe to isolate runtime
  cleanup hangs; the outer timeout remains mandatory.
