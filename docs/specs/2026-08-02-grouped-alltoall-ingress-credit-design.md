# Grouped AllToAll Ingress Credit Design

## Status

Implemented and validated on 4x8 hardware on 2026-08-02. The experimental
feature remains disabled by default. The strict global admission bound still
requires 256P all-rank trace or admission-counter validation.

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
2. Before receive-copy, the owner publishes a local request for lane `L`.
3. Primary send core `L` observes that request and posts one credit token to
   `peer(D, g + 1, L)`, when that peer exists.
4. The next source observes the credit in its local registered credit plane and
   may post both primary and secondary payload routes to `D`.

At most one source per destination lane can therefore be admitted. With 16
lanes, the strict payload-source bound is 16 ranks per destination.

Credit dependencies are monotonic from group `g` to `g + 1`. Group zero has no
dependency, so the protocol does not introduce a cyclic startup wait.

## Credit Storage And Tokens

Credits use a separate registered-memory plane. A credit received by source
rank `S` is indexed by destination rank `D`, because `S` communicates with `D`
only once per invocation:

```text
credit[pingPongSlot][destinationRank]
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
kernel cores 32 through 47, owns the local credit request. Cores 48 through 63
never request credits.

The owner publishes its request after both expected primary and secondary data
tokens arrive and before MTE receive-copy begins. This avoids placing copyout
latency on the send admission path.

Each `(pingPongSlot, lane)` request occupies its own 64-byte cache line. Packing
multiple lanes into one line is incorrect: independent receive cores clean and
invalidate the line concurrently, so one core can overwrite another lane's
request with stale cache-line contents.

## Credit Submission

Credit publication is an NBI 8-byte UDMA write issued only by the corresponding
primary send core. Receive cores do not issue UDMA operations because they can
share a payload SQ with a send core, while SQ head and WQE count updates require
a single producer.

The credit path must not perform a quiet for every credit. Token source storage
remains immutable until completion through distinct lane/group slots. At the
end of the invocation, the primary send core reclaims credit completions once
per unique underlying shared SQ, identified by its WQE-count address. This
preserves single-producer SQ ownership and avoids duplicate quiet operations
when multiple peer/QP views refer to the same shared queue.

If the available UDMA interface cannot safely keep credit source data alive
without per-credit quiet, implementation pauses for a revised design rather
than adding a per-credit quiet to the critical path.

## Configuration

Add an experimental ingress window setting:

```text
TILEXR_DEMO_ALLTOALL_GROUP_INGRESS_WINDOW=0|1
```

- `0`: current behavior and default.
- `1`: one admitted source per lane, at most 16 payload source ranks per
  destination.

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

This proves the request/credit protocol progresses across repeated invocations
and does not regress 1 GiB throughput in the 4x8 environment. It does not prove
the 16-source global bound because 4x8 has only two peer groups.

## Residual Risk

Rank0-only trace cannot prove the global source-card bound. Hardware validation
needs either all-rank trace or lightweight per-destination admission counters.
The physical four-port capacity and cabinet-aware route weighting are separate
topology problems and are not solved by this credit protocol.
