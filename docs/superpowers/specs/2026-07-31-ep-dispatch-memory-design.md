# EP Dispatch Memory Design

Status: implemented design, documented from the current TileXR behavior on 2026-07-31.

## Purpose

The dispatch memory path routes token rows from every EP rank to the ranks that
own their selected experts. It preserves the full-mesh algorithm's ordering and
window protocol while using TileXR communicator state and one direct kernel
launch.

This path is a standalone TileXR EP backend. Active code must not include or
link comparison sources under `reference/`.

## Scope

The current design supports:

- FP16 and BF16 token inputs;
- non-quantized output with the same type as the input;
- MXFP8 dispatch output in E4M3 or E5M2 format;
- ordinary MoE experts and shared experts;
- multiple ranks assigned to each shared expert;
- no active mask, a token mask shaped `[bs]`, or an expert mask shaped
  `[bs, topK]`;
- expert token totals represented as either counts or prefix sums;
- one EP communicator invocation at a time on the supplied stream.

The current design does not support TP execution, non-default expert sharding,
INT8 quantization, smooth scales, or separate scale expansion. These cases are
rejected by the host path instead of being approximated in the kernel.

## Ownership Boundaries

The public memory entry points are compatibility adapters. They build one
dispatch parameter object and converge on the same validation and launch path.

The host path owns:

- pointer, type, shape, rank, expert-layout, and optional-input validation;
- agreement between explicit EP values and the communicator;
- derivation and capacity checking of the peer-window layout;
- selection of the device's vector-core count as the kernel block dimension;
- allocation of a communicator-wide magic value for the invocation;
- registration and direct launch of the embedded memory kernel binary.

The communicator owns rank identity and peer address discovery. The kernel
receives device `CommArgs` and obtains all peer window bases from
`CommArgs::peerMems`. Dispatch does not accept a caller-managed workspace and
does not select a separate same-node or cross-node algorithm.

The kernel owns route construction, payload transfer, distributed receive-count
calculation, output compaction, and output metadata production.

## Routing Model

Shared-expert ranks occupy the leading EP ranks. When shared experts are
enabled, their rank count is divided evenly among shared experts. The source
rank's position inside a shared-expert group selects the destination rank, so
traffic is distributed without changing the logical shared-expert identity.

Ordinary MoE experts are divided evenly across the remaining ranks. Each valid
`expertIds[token, topK]` entry determines a destination rank and a local expert.
Masked routes are omitted before counts and output positions are calculated.

The receiver produces rows in deterministic expert-major, then source-rank
order. Original send order is preserved within each source/expert group. This
ordering is the contract shared by `expandXOut`, scales, counts, and assist
metadata.

## Output Contract

For each compacted row, `assistInfoForCombineOut` contains:

```text
[source rank, source token index, topK or shared slot, logical expert id]
```

Shared slots follow the normal `topK` slots. The logical expert field remains
available for test and downstream interoperability even though memory combine
routes a row using the first three fields.

`sendCountsOut` is cumulative and uses expert-major, source-rank order. A
shared-expert rank has one local expert and therefore one entry per source rank.
A normal MoE rank has one source-rank sequence for each local expert. Its last
entry is the total number of valid compacted rows on that rank.

`expertTokenNumsOut` reports one value per local expert. The selected mode
chooses either independent counts or a prefix sum across local experts.

## Window And Round Protocol

Every peer data region is treated as three logical areas:

```text
peerMems[rank] + IPC_DATA_OFFSET
  state window
  two ping-pong data halves
  local receive-count workspace reservation
```

The state window carries full-mesh run state, count exchange, and local cumsum
coordination. The data region reserves combine slots at the front of each half;
dispatch payloads begin after that reservation. Dispatch and combine derive the
same reservation and tail workspace size so their views of the shared window do
not overlap.

`TileXRCommNextMagic` supplies the operation epoch. Its low bit directly selects
the data and state half for all ranks. The kernel does not flip this state
internally. Reusing a half is valid only after the previous operation using that
half has completed on every participating rank.

Each receive count occupies one aligned status record. The count and its ready
flag are adjacent and are sent together by one data copy. The receiver polls the
ready field before consuming the count. A separate `SyncCollectives` notification
is intentionally not part of this protocol because it would add another remote
transaction and round trip.

Token payloads use DataAsFlag records:

```text
512-byte block = 480-byte payload + 32-byte ready area
```

Payload and ready state are written in the same transfer. A token may span
multiple blocks. The receiver consumes a token only after all of its block flags
have arrived, then clears the consumed flags before that window half is reused.

## Execution Flow

One launch performs initialization and processing as a single operation:

1. Resolve communicator state, tensor views, expert layout, window half, and
   per-core work ownership.
2. Split vector cores between payload routing and count/cumsum work.
3. Route valid shared-expert and MoE rows directly into destination peer
   windows, including source metadata in each token record.
4. Compute per-destination counts and send each count together with its arrival
   flag.
5. Receive counts, compute distributed cumulative output positions, and publish
   them to all participating receive cores.
6. Poll token DataAsFlag records, compact arrived rows into the output order,
   emit assist records and optional MXFP8 scales, and clear consumed flags.

The count path and payload path run concurrently. Their coordination is carried
by the state window; no host synchronization is inserted between them.

## MXFP8 Contract

MXFP8 dispatch is selected by dispatch quantization mode `4`. Input remains
FP16 or BF16. `expandXOut` is FP8 E4M3 or E5M2 and
`dynamicScalesOut` is mandatory.

Each group of 32 hidden elements shares one E8M0 scale. The per-row scale count
is padded to an even number, and the FP8 payload is padded independently for the
communication layout. Quantization happens before the row is sent, so the peer
window and compacted output carry the same FP8 bytes and scale bytes. Padding is
not part of the logical `expandXOut` tensor.

Metadata, route ordering, counts, masks, and shared-expert behavior are
unchanged by quantization. INT8 and mixed dispatch quantization modes are outside
this contract.

## Validation And Failure Semantics

Before consuming a magic value or launching the kernel, the host verifies:

- required pointers and stream/communicator state;
- FP16/BF16 input and the output type required by the selected quantization;
- mask type and pointer agreement;
- EP rank/world agreement and `globalBs == bs * rankSize`;
- valid shared-expert grouping and divisible MoE expert placement;
- TP size one and default expert sharding;
- peer address availability for every rank;
- state, data-half, workspace, and UB capacity for the complete configuration.

Invalid configurations fail synchronously. A successful return means the kernel
was enqueued on the supplied stream; completion and asynchronous device errors
remain subject to normal stream synchronization rules.

## Verification Strategy

The design is covered at four levels:

- host validation tests for supported and rejected contracts;
- layout tests for checked arithmetic, shared-expert placement, window capacity,
  and MXFP8 row sizing;
- source and target-toolchain checks for the direct launch ABI and required
  protocol branches;
- multi-rank hardware correctness tests for masks, expert layouts, repeated
  magic epochs, FP16/BF16, and MXFP8 output against a host golden model.

Performance measurements must warm up the communicator and kernel, time only
post-warmup device work, and retain rank count, shape, expert placement, type,
quantization format, and build metadata with the result.
