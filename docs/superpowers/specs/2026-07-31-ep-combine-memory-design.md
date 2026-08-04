# EP Combine Memory Design

Status: implemented design, documented from the current TileXR behavior on 2026-07-31.

## Purpose

The combine memory path returns expert output rows to their source ranks and
reduces all routes belonging to each source token. It consumes the ordering and
metadata produced by dispatch memory, communicates through TileXR peer windows,
and completes in one direct kernel launch.

This path is a standalone TileXR EP backend. Active code must not include or
link comparison sources under `reference/`.

## Scope

The current design supports:

- FP16 and BF16 expert input and token output;
- ordinary MoE routes and shared-expert routes;
- optional additional shared-expert input added locally at the source rank;
- no active mask or a token mask shaped `[bs]`;
- optional per-route expert scales;
- non-quantized communication;
- MXFP8 E5M2 or E4M3 communication quantization;
- one EP communicator invocation at a time on the supplied stream.

The current design does not support TP execution, non-default expert sharding,
expert-shaped active masks, INT8 communication quantization, or fused
post-combine operations such as AddRmsNorm.

## Ownership Boundaries

The public memory entry points are compatibility adapters. They build one
combine parameter object and converge on the same validation and launch path.

The host path owns:

- pointer, type, shape, rank, expert-layout, and optional-input validation;
- agreement between explicit EP values and the communicator;
- derivation and capacity checking of the shared peer-window layout;
- selection of the device's vector-core count as the kernel block dimension;
- allocation of a communicator-wide magic value for the invocation;
- registration and direct launch of the embedded memory kernel binary.

The communicator owns rank identity and peer address discovery. The kernel
receives device `CommArgs` and obtains all peer window bases from
`CommArgs::peerMems`. Combine does not accept a caller-managed workspace and
does not select a separate same-node or cross-node algorithm.

The kernel owns reverse routing, communication packing, arrival detection,
dequantization when enabled, route weighting, accumulation, and output writing.

## Input Contract With Dispatch

`expertOut`, `assistInfoForCombine`, and `sendCounts` describe the same compacted
row sequence. The expected assist tuple is:

```text
[source rank, source token index, topK or shared slot, logical expert id]
```

Combine uses the source rank, token index, and slot to determine the destination
window address. The logical expert id is retained as part of the shared EP
metadata contract but is not needed for reverse addressing.

`sendCounts` is the cumulative expert-major array emitted by dispatch memory.
Its last entry is the number of valid rows in `expertOut`; combine uses that
total to partition send work across vector cores. Callers constructing
combine-only inputs must preserve this cumulative convention.

Normal MoE slots are `[0, topK)`. Shared-expert slots immediately follow them.
Every active token must receive all configured slots before reduction begins.

## Window And Round Protocol

Combine shares the same peer data-region model as dispatch:

```text
peerMems[rank] + IPC_DATA_OFFSET
  state window
  two ping-pong data halves
  local receive-count workspace reservation
```

Combine uses the reserved front portion of the selected data half. Dispatch
payloads start after this area. Both operators derive the same data-half and
tail-workspace boundaries, which keeps their addresses compatible even though
combine does not use the dispatch receive-count workspace directly.

`TileXRCommNextMagic` supplies the operation epoch. Its low bit directly selects
the same ping-pong half on every rank. Combine does not flip the state inside the
kernel.

Each returned row is packed as DataAsFlag blocks:

```text
512-byte block = 480-byte payload + 32-byte ready area
```

Payload and ready state are written together by one peer-memory transfer. The
receiver polls the flags embedded in all blocks for all slots of a token. There
is no separate `SyncCollectives` ready notification and therefore no additional
communication round trip between data and readiness.

After a token is reduced, its consumed flags are cleared. This makes the same
window half reusable on a later magic epoch without treating stale payload as a
new arrival.

## Execution Flow

Send and receive progress in the same kernel launch:

1. Resolve communicator state, tensor views, expert layout, window half, row
   format, and per-core ownership.
2. Read the final cumulative send count and divide valid expert rows across
   vector cores.
3. For each assigned row, read its assist tuple, map it to the source rank's
   token/slot address, optionally quantize it, and write one packed DataAsFlag
   record to that peer window.
4. Divide source tokens across vector cores independently of send-row ownership.
5. For each active token, poll until every expected route slot has arrived.
6. Decode each slot, apply its route scale, and accumulate into FP32. Shared
   slots use scale one, and optional `sharedExpertX` is added locally.
7. Clear the token's arrival flags, convert the FP32 sum to the requested FP16
   or BF16 output type, and write `yOut`.

Inactive tokens produce zero output and do not wait for route payloads. Receive
polling provides the dependency between peer senders and local accumulation, so
the host does not insert a rank barrier between send and receive phases.

## Accumulation Contract

All route contributions are accumulated in FP32 regardless of the external
FP16/BF16 type. MoE routes are processed in topK order and use the corresponding
`expertScales[token, topK]` value when scales are supplied. Shared-expert routes
are accumulated after MoE routes with scale one. Optional `sharedExpertX` is an
additional local contribution, also with scale one.

The final FP32 sum is converted once to the output type. This order is part of
the numerical contract and must also be used by correctness golden models.

## MXFP8 Communication Contract

Combine communication quantization is independent of dispatch output
quantization:

- mode `3` selects MXFP8 E5M2;
- mode `4` selects MXFP8 E4M3;
- mode `0` sends the original FP16/BF16 row.

For MXFP8, each group of 32 hidden elements shares one E8M0 scale. The FP8
payload is padded for communication and the per-row scale count is padded to an
even number. The sender quantizes each `expertOut` row before packing it into
DataAsFlag blocks. The receiver dequantizes directly into FP32 accumulation and
applies the route's FP32 expert scale.

`expertScales` is mandatory when MXFP8 communication is selected. Quantization
does not change assist addressing, route ordering, active-mask behavior, or the
final output type.

## Shared Experts

Shared-expert rank placement follows dispatch memory: leading EP ranks are
divided evenly among the configured shared experts. Each shared route is
returned through its slot after the normal topK slots, allowing the receiver to
use one fixed token layout for both ordinary and shared contributions.

`sharedExpertX`, when supplied, is not communicated through these slots. It is a
source-rank tensor added after all communicated routes have been accumulated.

## Validation And Failure Semantics

Before consuming a magic value or launching the kernel, the host verifies:

- required pointers and stream/communicator state;
- FP16/BF16 external data type;
- supported communication quantization mode and mandatory MXFP8 scales;
- mask type and pointer agreement;
- EP rank/world agreement and `globalBs == bs * rankSize`;
- valid shared-expert grouping and divisible MoE expert placement;
- TP size one and default expert sharding;
- peer address availability for every rank;
- data-half and UB capacity for the complete row and token configuration.

Invalid configurations fail synchronously. A successful return means the kernel
was enqueued on the supplied stream; completion and asynchronous device errors
remain subject to normal stream synchronization rules.

## Verification Strategy

The design is covered at four levels:

- host validation tests for supported and rejected contracts;
- layout tests for checked arithmetic, DataAsFlag row sizing, shared-expert
  capacity, and both MXFP8 formats;
- source and target-toolchain checks for the direct launch ABI, polling, flag
  clearing, and quantized/non-quantized branches;
- multi-rank hardware correctness tests for combine-only and dispatch-combine
  execution, token masks, shared experts, repeated magic epochs, FP16/BF16, and
  MXFP8 results against an FP32 host golden model.

Performance measurements must warm up the communicator and kernel, time only
post-warmup device work, and retain rank count, shape, expert placement, type,
quantization format, and build metadata with the result.
