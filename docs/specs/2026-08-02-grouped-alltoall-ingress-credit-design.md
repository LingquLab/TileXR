# Grouped AllToAll Ingress Credit Design

## Status

The original request-plus-UDMA-credit implementation was validated through 8x8
hardware on 2026-08-02. It has since been replaced by direct receive-core
publication through a dedicated credit IPC allocation and requires renewed
hardware validation. The experimental feature remains disabled by default.

## Goal

Bound each destination rank to at most 16 concurrent payload source ranks in
the grouped AllToAll data path. Preserve independent lane progress so one slow
source does not impose a full-rank or full-group barrier.

The first implementation is experimental and disabled by default. Existing
behavior and performance remain unchanged unless ingress credits are enabled.

## Non-goals

- Limiting the number of physical port operations to 16. A source may still
  use primary and secondary routes after it receives one destination credit.
- Adding a global barrier between peer groups.
- Waiting for receive-copy completion before releasing network capacity.
- Changing topology-derived primary/secondary route weights.
- Enabling credit control for multi-pass payloads in the first hardware test.

## Existing Behavior

At `rankSize=256` and `groupWidth=16`, every rank has 16 logical peer groups.
Each group contains up to eight forward and eight backward peers. The 32 send
workers are split into 16 primary-route workers and 16 secondary-route workers.

Each worker advances to its next group immediately after its own quiet
completes. Workers and ranks do not share group progress. Consequently, a
destination may receive payloads from more than one logical group at the same
time even though each group contains no more than 16 source ranks.

## Credit Protocol

There are 16 independent destination lane chains. Group zero is initially
enabled. For group `g > 0`, a source must observe a credit from its destination
for `(invocation, g)` before either route posts payload data.

For destination rank `D`, lane `L`, and group `g`:

1. The designated receive owner waits until all payload routes expected from
   `peer(D, g, L)` have published their data-ready tokens.
2. Before receive-copy, the owner directly writes one credit token through the
   dedicated IPC mapping of `peer(D, g + 1, L)`, when that peer exists.
3. The next source observes the credit in its local dedicated credit buffer and
   may post both primary and secondary payload routes to `D`.

At most one source per destination lane can therefore be admitted. With 16
lanes, the strict payload-source bound is 16 ranks per destination.

Credit dependencies are monotonic from group `g` to `g + 1`. Group zero has no
dependency, so the protocol does not introduce a cyclic startup wait.

## Credit Storage And Tokens

Each rank allocates a dedicated 1 MiB IPC buffer when
`TILEXR_ENABLE_CREDIT_IPC=1`. A credit received by source rank `S` is indexed by
destination rank `D`; each entry occupies 512 bytes and two fixed 512 KiB
ping-pong planes support 1024 ranks:

```text
credit[pingPongSlot][destinationRank * 512]
```

The expected value encodes at least the invocation and destination group. The
receiver accepts a value greater than or equal to the expected token, matching
the existing data-ready stale-token policy. Ping-pong slots prevent adjacent
invocations from reusing a live credit location.

Primary and secondary send workers for the same peer wait on the same credit.
Credits control source-rank admission, not route admission.

## Receive Ownership

With 32 copyout workers, workers 0 through 15 and 16 through 31 can wait on the
same peer signal while copying different payload slices. Only the first slice,
kernel cores 32 through 47, owns direct credit publication. Cores 48 through 63
never publish credits.

The owner publishes its credit after both expected primary and secondary data
tokens arrive and before MTE receive-copy begins. This avoids placing copyout
latency on the send admission path.

## Credit Submission

Credit publication is a direct 8-byte GM store by the receive owner through
`creditMems[nextSource]`. It does not consume a UDMA WQE, QP, completion, or
quiet, and it does not add another producer to a shared UDMA SQ. Both send
routes poll the same local dedicated credit address through MTE.

The dedicated allocation is independent of the existing optional communication
IPC buffer, so grouped ingress credit can run with `TILEXR_ENABLE_IPC=0`. It
adds one IPC mapping per peer and process, but only 1 MiB of device memory per
rank. Host code rejects ingress-credit execution if any mapping is missing.

## Configuration

Add an experimental ingress window setting:

```text
TILEXR_DEMO_ALLTOALL_GROUP_INGRESS_WINDOW=0|1
```

- `0`: current behavior and default.
- `1`: one admitted source per lane, at most 16 payload source ranks per
  destination.

`window=1` additionally requires `TILEXR_ENABLE_CREDIT_IPC=1` before
communicator initialization.

A later `window=2` extension may trade a 32-source bound for more tolerance of
credit latency, but it is outside the first implementation.

## Validation

Host/unit tests must cover:

- 256-rank group and lane predecessor/successor mapping.
- No credit wait for group zero.
- One shared credit for primary and secondary routes.
- Single credit publisher when 32 copyout workers duplicate receive lanes.
- Correct inactive-lane behavior at the diameter and final partial group.
- Credit token separation across invocations and ping-pong slots.
- Default-disabled compatibility.

Hardware validation starts with 2x8 functional coverage, then 4x8, then 256P.
The 256P comparison uses 1 GiB/rank, multi channel, single pass, warmup 5 and
repeat 50. Evidence includes P50 and range plus per-phase `credit-wait`,
`send-quiet`, and `receive-wait` trace durations. A 1 KiB single-channel case
checks that the default-disabled path has no latency regression.

The first trace-enabled run is diagnostic only and is not compared directly
with trace-disabled performance.

The final 4x8, warmup-5/repeat-50, trace-disabled comparison passed data
validation on all 32 ranks:

```text
1 KiB/rank single: window0 P50 38.224 us, window1 P50 52.763 us
1 GiB/rank multi:  window0 P50 5329.210 us, window1 P50 5317.925 us
```

This proved the previous request/UDMA-credit protocol progressed across repeated invocations
and does not regress 1 GiB throughput in the 4x8 environment. It does not prove
the 16-source global bound because 4x8 has only two peer groups.

The 8x8 A10 run used the fixed host order `226, 223, 220, 217, 198, 195,
192, 189`, single pass, shared QP, trace disabled, and warmup-5/repeat-50. All
64 ranks passed data validation:

```text
1 KiB/rank single: window0 P50 73.067 us, window1 P50 105.930 us
1 GiB/rank multi:  window0 P50 6516.335 us, window1 P50 6336.370 us
```

Under the previous request/UDMA-credit implementation, ingress credit reduced
the 1 GiB P50 by 179.965 us (2.76%), from
approximately 153.46 GiB/s to 157.82 GiB/s. The 1 KiB case paid 32.863 us of
additional fixed control latency. This run exercises four peer groups and
therefore provides stronger repeated credit-chain evidence than 4x8, but it
still does not prove the 256P global admission bound.

## Residual Risk

Rank0-only trace cannot prove the global source-card bound. Hardware validation
needs either all-rank trace or lightweight per-destination admission counters.
The physical four-port capacity and cabinet-aware route weighting are separate
topology problems and are not solved by this credit protocol.
