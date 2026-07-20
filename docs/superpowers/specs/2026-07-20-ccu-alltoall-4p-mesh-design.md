# CCU AlltoAll 4P Mesh Design

## Goal

Extend the validated direct CCU AlltoAll path from two ranks to a true four-rank
Mesh1D collective. Each rank uses one communicator, one installed mission, and
one stable set of QP, jetty, CKE, XN, channel, and registered-memory resources.
Validation first runs one submission and then ten consecutive submissions with
the same prepared resources.

The hardware target is `141.61.49.192` on NPU devices `4,5,6,7`. Each rank owns
an 8 MiB send buffer and an 8 MiB receive buffer. Each peer chunk is 2 MiB.

## Collective Semantics

Buffers use rank-major chunk layout:

```text
send[targetRank][2 MiB]
recv[sourceRank][2 MiB]
```

For local rank `r`, the collective must produce:

```text
recv_r[s] == send_s[r]
```

for every source rank `s` in `0..3`. The test pattern encodes source rank,
target rank, loop index, and byte offset. This detects incorrect source,
destination, generation, chunk placement, and partial-copy behavior.

## Architecture

Add a general four-rank Mesh spec and program instead of changing the meaning
of `TileXRCcuDirectAllToAll2RankSpec`. The existing 2P interfaces remain as a
regression baseline. Shared block-copy, marker, and diagnostic helpers may be
factored only where this does not change 2P behavior.

Each rank has three peer descriptors sorted by peer rank. A descriptor contains
the imported remote destination address and token, peer rank, loop-marker XNs,
and the copy, PreSync, and token route resources.

The fixed sync-resource layout is:

```text
peer[0]: copy=0, pre=1, token=2
peer[1]: copy=3, pre=4, token=5
peer[2]: copy=6, pre=7, token=8
```

The planner performs one endpoint AllGather, validates all four endpoints,
imports the other three destination buffers, and installs route-specific
memory overrides for all nine sync routes. The lower-layer override state must
therefore become a collection indexed by sync-route index instead of a single
optional override.

The program uses one mission. It publishes marker, output address, and token to
all three peers before waiting on any peer, then waits for every peer's PreSync
mask `0x7`. This all-post-before-wait ordering prevents rank-dependent peer
iteration from creating a wait cycle.

After PreSync, the mission copies the three remote chunks to
`peer.recv[localRank]` and copies the self chunk from `send[localRank]` to
`recv[localRank]`. It waits for all local and remote completion signals before
returning.

## CCU Self Copy

The self chunk is copied by CCU, not by ACL. HCCL/hcomm reference code maps
`GroupCopy` through `CcuRepLocCpy` to the hardware
`TransLocMemToLocMemInstr` instruction. TileXR will add an independent
`TileXRCcuEncodeTransLocMemToLocMem` encoder using the verified instruction
field layout without including, linking, or calling private HCCL/hcomm code.

The AlltoAll program builder will load self source and destination addresses,
their tokens, and length, issue local-to-local transfers, and wait for local
completion. Unit tests must decode every relevant instruction field and verify
the self offsets for local ranks zero through three.

## Instruction And Resource Capacity

Each 2 MiB remote chunk retains the validated 64 blocks by 7 instructions copy
shape. Three remote chunks plus PreSync are expected to require about 1365
instructions, with additional instructions for CCU self copy and final waits.

Prepare must calculate the exact program size before installation and compare
the requested repository range with device basic-info capacity. Insufficient
instruction, channel, XN, CKE, GSA, or route resources must fail with requested
and available counts. The implementation must not truncate a program or reuse
one peer's resource IDs for another peer.

## Repeated Submission Protocol

Allocation, endpoint exchange, remote import, route construction,
registration, mission installation, prepared-task creation, and stream creation
all occur once outside the loop.

For each loop index:

1. Fill all four send chunks with source/target/loop-specific patterns.
2. Reset the complete 8 MiB receive buffer.
3. Update prepared-task argument zero with a rank-and-loop marker.
4. Enter the four-rank `ready.phaseN` gate.
5. Submit the same prepared task and synchronize the same stream.
6. Read and validate the current marker from all three peers.
7. Read and compare the complete 8 MiB receive buffer.
8. Enter the four-rank `done.phaseN` gate with the local validation result.

The marker contains a fixed magic prefix, sender rank, and loop index. CKE bits
remain presence flags and are cleared by their waits; marker XNs carry the
generation identity needed to reject stale synchronization.

If any rank reports an error, all ranks stop before the next loop. Diagnostics
must include local rank, loop index, peer rank, route and channel IDs, marker
XN/CKE values, mission current instruction, local-copy completion, and the first
mismatch's source rank, chunk offset, global offset, expected byte, and observed
byte.

## Runner

The smoke runner must accept rank size four and a four-device list. It will
construct rank-specific environment arrays for ranks zero through three,
launch four processes, track four PIDs and statuses, and validate four logs.
Hard-coded rank0/rank1 loops and result thresholds must become rank-size-driven.

Rank-specific endpoint and resource-window fields, including EID index, must be
forwarded for all four ranks. Four-rank AlltoAll defaults use EID index 3 for
each rank unless explicitly overridden.

The runner must require the exact number of successful results and marker
checks for the requested loop count. Seeing one successful line is not
sufficient.

## Testing

Automated coverage includes:

- microcode encoding and decoding for `TransLocMemToLocMem`;
- rank 0..3 program generation, peer ordering, all-post-before-wait ordering,
  self offsets, remote offsets, completion waits, and instruction counts;
- planner endpoint exchange, three remote imports, nine route mappings, and
  resource-exhaustion diagnostics;
- smoke pattern/reset, three peer-marker checks, full 8 MiB mismatch reporting,
  and prepare-before-loop ordering;
- dynamic four-process runner launch, environment forwarding, status handling,
  exact result counts, and dry-run output;
- all existing 2P program, planner, smoke, runner, and loop-reuse regression
  tests.

## Hardware Validation

Before every hardware run, query `npu-smi` and apply the repository busy guard
to devices `4,5,6,7`. If any selected device is busy, poll every 30 seconds and
do not terminate or bypass existing jobs.

Run these stages in order:

1. Build `tile-comm` and the four-rank smoke probe.
2. Run 4P with loop count one in a fresh work directory.
3. Run 4P with loop count ten in another fresh work directory.
4. Re-run the existing 2P loop-count-ten test on its validated device pair.

The 4P loop-one run requires four successful rank results, twelve matching peer
markers, and zero mismatches over every 8 MiB receive buffer.

The 4P loop-ten run requires forty successful rank results and 120 matching
peer markers. Each rank must retain the same mission ID, key, instruction start,
instruction count, task count, and installed resource IDs across all ten loops.
Every loop must report zero mismatches.

## Deployment And Cleanup

Use Mutagen as the preferred source synchronization mechanism. First determine
whether the installed Mutagen version supports an explicit SSH identity without
the disabled Windows `ssh-agent`. If it does not, record that incompatibility
and use the previously approved `scp -i` fallback for only the changed files.

A temporary root SSH key may be installed for validation. After successful or
aborted validation, remove only its uniquely tagged remote authorized-key entry
and delete its local private key, public key, askpass helper, and any temporary
SSH wrapper or configuration. Do not modify unrelated untracked files or stop
unrelated remote workloads.

## Non-Goals

- Supporting arbitrary rank sizes beyond four in this validation cycle.
- Adding AlltoAllV variable counts or displacements.
- Matching multi-jetty performance tuning from HCCL.
- Replacing TileXR's runtime with HCCL/hcomm private APIs.
- Claiming network-copy concurrency or performance until profiling proves it.
