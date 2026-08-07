# UDMA Registered-Memory Reliability And Configurable Multi-QP Design

## Goal

Stabilize TileXR registered-memory UDMA on Ascend950 and extend the transport
from its legacy single-QP layout to an operator-neutral, configurable number of
QP/CQ pairs per peer. The design provides generic Host and device discovery and
submission APIs, migrates in-tree callers to explicit UB WQE scratch, and gives later
operators a reusable multi-QP transport without embedding operator names or
policies in `src/comm`, public UDMA headers, or UDMA tests.

The implementation targets C++14, CANN 9.1.0, and the repository's supported
UDMA hardware. Hardware acceptance is performed on Ascend950PR hosts
`141.61.49.195` and `141.61.49.198`.

## Fixed Decisions

1. An unset multi-QP configuration preserves the single-QP transport. Same-node
   peers use topology routing; cross-node peers prefer the first deterministic
   aggregate EID and fall back to the first EID only when route data is unavailable.
2. Explicit multi-QP configuration is process-level, read during communicator
   initialization from `TILEXR_UDMA_QP_ROUTE_SPEC`.
3. The number of comma-separated route rules is the requested QP count. The
   initial implementation supports one through eight QPs per peer.
4. Explicit configuration is strict. If a requested topology or port-count
   route is unavailable, TileXR does not silently substitute a different EID or
   reduce the QP count.
5. `UDMAInfo::qpNum` is the authoritative device-side capability. No
   operator-specific `ExtraFlag` bit is added.
6. Each configured peer/QP pair owns independent QP, CQ, head, tail, WQE count,
   doorbell, and imported-QP state. QPs may select the same EID without sharing
   their queue state.
7. Queue routes, contexts, creation, exchange, and import complete during UDMA
   transport initialization. Registered-memory setup does not create or import
   QPs.
8. TileXR continues to support one active registered region per communicator.
   This change does not introduce multi-region registration.
9. Registered-memory cleanup is transactional and retryable. State for a
   resource is discarded only after the corresponding HCCP cleanup succeeds.
10. Ascend950 and Ascend950PR registered device memory uses `nonPin=1`
    regardless of the configured QP count. Registration mode is not an
    operator or route-policy decision.
11. The UDMA core API imposes no 2 MiB size limit or alignment requirement.
    Exact 2 MiB registration is an acceptance case, while operator workspace
    alignment remains a caller-level contract.
12. Communicator initialization remains best effort with respect to UDMA. A
    UDMA configuration or resource failure leaves other communicator paths
    usable and does not publish a partially initialized UDMA capability.

## Scope

### In Scope

- Correct local-memory registration flags and useful HCCP diagnostics.
- Transactional local MR registration, remote MR import, replacement,
  unregistration, and retryable cleanup.
- Deterministic RootInfo parsing without `std::regex`.
- Generic QP route-spec parsing and cross-rank configuration agreement.
- Dynamic one-through-eight per-peer QP/CQ creation and import.
- Rank-major, QP-minor device metadata layout.
- Generic per-QP device PUT, GET, deferred submission, doorbell, and quiet APIs.
- A Host API for querying the initialized QP count.
- Focused Host tests, fault-injection tests, and Ascend950 data-plane probes.
- A cross-host-capable UDMA demo barrier and validation runner.

### Non-Goals

- Any MoonEP, EP operator, planner, ReduceGrad, Python benchmark, or Torch change.
- Operator-specific route policies, capability bits, constants, or class names.
- PR #88 grouped AllToAll, SDMA copyout, grouped scheduling, or performance
  kernel changes.
- Multi-region UDMA registration.
- Dynamic route rebalancing based on measured traffic or throughput.
- Runtime QP-count reduction when explicit configuration cannot be satisfied.
- Supporting `TileXRUDMARegister` from `TileXRCommInitThread`.
- Registering `peerMems[]`; registration targets remain ordinary device memory.

## Configuration Contract

### Environment Variable

`TILEXR_UDMA_QP_ROUTE_SPEC` is read once by each UDMA context before route
construction. Missing or empty input selects the legacy single-QP path. A
non-empty value is a comma-separated list of route rules:

```text
topology
port_count:<positive-decimal-integer>
```

Examples:

```text
TILEXR_UDMA_QP_ROUTE_SPEC=topology
TILEXR_UDMA_QP_ROUTE_SPEC=port_count:6,port_count:2
TILEXR_UDMA_QP_ROUTE_SPEC=port_count:6,port_count:6,port_count:2
```

The second example creates two QPs. The third creates three QPs and proves that
multiple QPs may intentionally use the same EID selector.

Whitespace surrounding the complete value or an individual rule is ignored.
Empty rules, unknown selector names, zero port counts, non-decimal values, and
more than eight rules are configuration errors. The parser normalizes each rule
to an internal descriptor rather than retaining the original text.

### Route Semantics

For a same-node peer, every configured QP uses the existing topology edge and
port map to select the peer route. The cross-node selector controls each QP:

- `topology` uses the existing topology route resolver.
- `port_count:N` selects the first deterministic candidate EID whose RootInfo
  port array has exactly `N` entries.

Candidate EIDs are ordered by EID index before selection. Duplicate selectors
are allowed. Missing topology data or a missing exact port-count match makes
explicit configuration unavailable; no arbitrary EID fallback is allowed.

The legacy path remains separate. When the variable is unset or empty it uses
one QP. Same-node peers retain topology selection. Cross-node peers select the
lowest-index RootInfo EID whose port count is greater than one, then fall back
to the first device EID only when RootInfo cannot provide an aggregate route.

### Cross-Rank Agreement

Configuration errors must not make some ranks return while other ranks enter a
socket collective. Every rank constructs a fixed-size wire descriptor containing:

```text
version
parseStatus
qpCount
routeRule[8] = {selectorKind, selectorValue}
```

All ranks exchange this descriptor before route construction. Any local parse
error or descriptor mismatch disables UDMA consistently on every rank. Route
selection results are then exchanged as a checked `[peer][qp]` matrix.

## Public And Device Interfaces

### Host Discovery

Add the C API:

```cpp
int TileXRUDMAGetQpCount(TileXRCommPtr comm, uint32_t *qpCount);
```

The function validates both pointers. It returns success with a positive count
only when UDMA transport initialization completed. If UDMA is unavailable it
sets `*qpCount` to zero and returns the repository's not-supported status.

This query is implemented through `TileXRComm`, `TileXRUDMAContext`, and
`TileXRUDMATransport`; it does not copy `UDMAInfo` back from device memory.

### Device Discovery

Add:

```cpp
__aicore__ inline uint32_t UDMAQpCount(const __gm__ CommArgs *args);
__aicore__ inline bool UDMAQpValid(
    const __gm__ CommArgs *args, uint32_t qpIdx);
```

`UDMAQpCount` returns zero when UDMA is disabled. Device callers must treat
`qpIdx >= UDMAQpCount(args)` as invalid.

### Generic Per-QP Operations

Add generic helpers equivalent to the existing QP0 operations:

```cpp
UDMAPutNbiOnQp(args, wqeScratch, ...)
UDMAPutNbiOnQpWithFlag(args, wqeScratch, ...)
UDMAPutNbiOnQpWithFlagDeferred(args, wqeScratch, ...)
UDMAGetNbiOnQp(args, wqeScratch, ...)
UDMAFlushQpDoorbell(...)
UDMAQuietStatusOnQp(...)
```

Also add a QP-aware `UDMAGetRemoteMemInfo(udmaInfo, peer, qpIdx)` overload.
Existing no-QP overloads and `UDMAPutNbi`, `UDMAGetNbi`, and `UDMAQuiet`
continue to select QP0.

The SQE flag literal is replaced with named completion and strong-order bits.
New per-QP enqueue helpers report invalid UDMA state, invalid QP, invalid
remote registered range, and insufficient SQ capacity instead of silently
writing an invalid entry. Existing void wrappers may preserve their behavior by
discarding the new internal status for QP0.

Every enqueue helper requires a caller-owned `LocalTensor<uint8_t>` scratch of
at least 128 bytes with 32-byte alignment. The helper clears and assembles the
complete WQE in that UB scratch. It publishes a one-BB WQE as 64 bytes and a
two-BB `WRITE_WITH_NOTIFY` WQE as the complete 128 bytes through MTE3. A notify
at the physical SQ ring end is split into 64-byte tail and head transfers. No
SQ WQE field may be constructed or patched through scalar or direct-GM stores.
On device, `LocalTensor::GetPhyAddr()` is a local-buffer offset and zero is a
valid address for the first allocated UB buffer; scratch validation therefore
checks size and alignment without treating zero as a null pointer.

The generic `WithFlag` enqueue helpers require `COMPLETION`; completion-only is
accepted, while a flag without completion is rejected before the SQ is mutated.
`UDMAPollCQ` can reclaim a contiguous SQ prefix because both legacy and
configured QP creation set `jfsFlag.value = 2`, which leaves the HCCP
`outorderComp` bit clear. The SQE `STRONG_ORDER` bit (`0x02`) controls placement
ordering and is not the completion-order bit (`0x04`), so requiring `0x02` would
not establish the CQ property used here. Enabling out-of-order completion in a
future QP configuration requires either the documented completion-order bit or
explicit tracking of completed entries before contiguous reclamation.

This critical correction does not add device-side local-range validation. For
PUT, GET, and signal PUT, the caller must pass a non-empty local data range that
lies wholly inside `registry->regions[args->rank]`; using an unrelated device
allocation with the registered region's token can fail memory protection in the
UDMA data plane. Remote byte offsets remain validated against the selected peer
region. A future defensive change may add subtraction-based local bounds checks
without changing the valid registered-memory path.

### Deferred Submission Contract

A deferred PUT publishes the UB-built SQ entry through MTE3 and advances the
software head and submitted-WQE count, but does not ring the doorbell. Before writing, it
checks that the required SQ basic blocks do not collide with the reclaimed tail.
If there is insufficient capacity, it returns without modifying queue state.

The caller then follows this sequence:

```text
enqueue one or more bounded deferred PUTs
  -> flush the selected QP doorbell
  -> quiet/reclaim the selected QP when capacity or completion is required
```

`UDMAFlushQpDoorbell` rings the selected queue at its current software head.
`UDMAQuietStatusOnQp` polls through the selected QP's submitted-WQE count and
returns the completion status. With the QP configured for in-order completion,
each CQE describes the next WQE at the current SQ reclaim tail, and its `entryIdx` identifies that
WQE's final basic block. The poller validates the reported final block against
the WQE's one- or two-block size before advancing the SQ tail. Deferred
submission never permits unconsumed SQ entries to be overwritten.

Scalar-to-MTE3 and MTE3-to-scalar events bracket every WQE publish. Head and
submitted-WQE count updates occur only after the MTE3-to-scalar event confirms
the SQ write completed. Immediate submission then rings the SQ doorbell last,
and both immediate and deferred doorbells use `st_dev` exclusively.

### Device Queue Ownership

Each `(peer, qpIdx)` SQ has one device-side producer. Enqueue, flush, and quiet
operations for the same queue must not execute concurrently from multiple
AICores or control paths; callers must serialize them or assign distinct QPs.
Different `(peer, qpIdx)` queues are independent and may progress concurrently.

## Host Transport Design

### Internal Configuration

Introduce internal, operator-neutral types similar to:

```cpp
enum class UDMAQpRouteSelector : uint32_t {
    TOPOLOGY,
    PORT_COUNT,
};

struct UDMAQpRouteRule {
    UDMAQpRouteSelector selector;
    uint32_t value;
};

struct UDMAQpConfig {
    bool explicitConfig;
    std::vector<UDMAQpRouteRule> routes;
};
```

The types remain under `src/comm/udma` and are not part of the public C ABI.
Production names must not mention an operator.

### Route Matrix

The selected local and remote EIDs are stored as:

```text
localRoute[peer][qp]
remoteRoute[peer][qp]
```

`localRankSize` is passed from `TileXRComm` through the context into transport
options so same-node and cross-node peers are distinguished before selector
evaluation.

All calculations involving `rankSize * qpNum`, image entries, exchanged arrays,
and byte sizes use checked arithmetic. Invalid sizes fail before allocation.

### Queue Ownership

Configured mode creates a `PerPeerQpState` for every non-self peer and QP. Each
state owns:

- channel, CQ, local QP, and imported remote-QP handles;
- CQ and QP creation metadata;
- QP-specific TPN;
- CQ producer/consumer scalars;
- SQ producer/reclaim scalars and submitted-WQE counter;
- doorbell and atomic addresses;
- device `UDMAWQCtx` and `UDMACQCtx` images.

Resource creation order is deterministic: peer rank ascending, then QP index
ascending. Failure cleanup runs over every successfully created state and keeps
failed cleanup handles available for a later attempt.

The legacy single-QP transport may retain its existing shared-per-EID queue
implementation. Configured mode does not require rewriting legacy ownership.

### Queue Exchange And Import

Each rank exchanges fixed `[peer][qp]` QP import metadata and keys. Local QP
creation status is agreed across ranks before import begins, preventing a rank
from entering import exchange after another rank has abandoned initialization.

After import, ranks perform one more status agreement before UDMA availability
is published. Any failure rolls back imported QPs and locally created queue
resources. A partially imported transport is never exposed through `CommArgs`.

### Device Image

`BuildUDMAInfoImage` accepts an explicit `qpNum` and validates that every vector
has the same non-zero entry count and that the count is divisible by `qpNum`.
The image is rank-major and QP-minor:

```text
entryIndex = peer * qpNum + qpIdx
```

This rule applies to SQ, RQ, SCQ, RCQ, and memory metadata arrays. The existing
`UDMAInfo` binary layout remains unchanged; its existing `qpNum` field becomes
dynamic rather than always one.

### Memory Import With Multiple QPs

Local device memory is registered once for every unique local EID context used
by the route matrix. Remote MR imports are deduplicated by:

```text
(peer, localEid, remoteEid)
```

Each `[peer][qp]` memory image copies the appropriate imported MR metadata and
combines it with that QP's imported TPN. Two QPs sharing an EID may reuse the MR
import while retaining different QP-specific TPN and queue state.

## Registered-Memory Reliability

### Registration Flags

For every local registration, initialize the full relevant HCCP flag contract
before the only `RaCtxLmemRegister` call:

```text
cacheable   = 0
access      = MEM_SEG_ACCESS_DEFAULT
nonPin      = 1 on Ascend950/Ascend950PR, otherwise 0
userIova    = 0
tokenIdValid = 1
tokenPolicy = MEM_SEG_TOKEN_PLAIN_TEXT
```

Recognize both `Ascend950PR_` and `Ascend950DT_` device-name prefixes. The chip
decision is made by communicator initialization and passed into transport
options; it is independent of the QP route specification.

On failure, log the HCCP return code, EID, byte count, pointer, handle, and
pointer modulo 2 MiB. Diagnostics must not incorrectly report alignment as a
TileXR API requirement.

### Size And Ownership Contract

`TileXRUDMARegister` accepts ordinary device memory, a non-null handle output,
and any positive `size_t` byte count that the underlying HCCP implementation can
register. TileXR adds no 2 MiB upper bound and passes the exact byte count to
HCCP.

Only one active region is published per communicator. The existing handle
contract remains stable. `InitThread` registration remains unsupported, and
`peerMems[]` are not valid registration targets.

### Transactional Replacement

Registration replacement follows prepare/commit semantics:

1. Preserve the current registration, registry image, and `CommArgs` state.
2. Register the candidate region on every required local EID.
3. Exchange and import candidate remote MR metadata.
4. Build and copy the candidate device registry and UDMA image.
5. Publish the new `CommArgs` state.
6. Commit the new Host state.
7. Clean up the old registration and registry.

Failure before commit rolls back the candidate and restores the previous state.
If candidate rollback or old-state restoration fails, the context enters
cleanup-pending state instead of pretending that no registration exists.

### Retryable Cleanup

Cleanup helpers return the first error while attempting all independent
resources. A successful HCCP cleanup removes only that resource from its map.
A failed handle remains stored for retry.

The context state is:

```text
Unavailable
  -> TransportReady
  -> MemoryReady
  -> CleanupPending
  -> TransportReady
```

In `CleanupPending`:

- the device registry pointer is hidden from Host and device `CommArgs`;
- a new registration is rejected;
- unregister retries the residual resources;
- registered pointer and byte ownership remain recorded;
- successful cleanup clears the pending state and returns to `TransportReady`.

Shutdown remains best effort but logs residual cleanup failures before queue and
context destruction.

### Cleanup Coordination Contract

The application must stop issuing operations to a registered region on every
rank and complete or quiet every affected QP before any rank unregisters that
region. After the device registry is hidden, `TileXRUDMAUnregister` and cleanup
retry are local and non-collective. TileXR removes the remote imports owned by
the current rank before attempting its local MR unregister, retains failed
handles in `CleanupPending`, and does not add a socket collective to the retry
path.

This contract relies on the target HCCP implementation allowing local MR
unregister after this rank has released its own remote imports; TileXR does not
establish cross-rank ordering between peer unimport and local unregister.
Platforms that require every peer import to be destroyed first need an external
rank-wide teardown protocol and are not covered by the current local retry
path. Adding an internal rank-wide phase boundary would make unregister
collective and can block or hang applications that currently call it locally,
so that API-semantic change is documented rather than included in the critical
data-plane correction.

## RootInfo Parsing

Remove `std::regex` from the UDMA runtime. Use a small deterministic JSON cursor
that recognizes objects, arrays, strings with supported escapes, and unsigned
decimal values required by HCCL RootInfo. It must reject malformed or truncated
input without searching through string values as if they were object keys.

The default file remains `/etc/hccl_rootinfo.json`. The diagnostic and test-only
override `TILEXR_UDMA_ROOTINFO_PATH` may select another file. RootInfo parsing
tests cover whitespace, escaped strings, quoted and plain integers, arrays,
missing fields, malformed input, and port-count extraction.

## Failure Handling

Explicitly configured UDMA initialization is staged:

```text
parse and agree configuration
  -> resolve and exchange routes
  -> create local contexts and queues
  -> agree local creation status
  -> exchange and import QPs
  -> agree import status
  -> build and publish UDMA image
```

Each distributed stage is entered only after rank-wide agreement on the prior
stage. This prevents one rank from blocking in an exchange after another rank
has returned locally.

Failure disables UDMA consistently and clears its `CommArgs` pointers and flag.
The communicator may continue with IPC or other supported transports. Explicit
multi-QP mode never silently reports success with fewer QPs or substituted EIDs.

## Compatibility

- Preserve C++14 and CANN 9.1 compatibility.
- Preserve the current `UDMAInfo`, `CommArgs`, and registry binary layouts.
- Preserve current no-QP device wrapper behavior through QP0 forwarding.
- Preserve the legacy single-QP interface when the new variable is unset or empty;
  use topology for same-node peers and an aggregate EID for cross-node peers.
- Keep UDMA and SDMA best-effort capabilities independent.
- Keep active targets independent of `reference/` sources.
- Do not add CANN `${ARCH}-linux/devlib` to runtime RPATH or RUNPATH.
- Device launch syntax is outside this PR.

## Verification

### Host And Unit Tests

- Route-spec parsing for one, two, three, and eight QPs.
- Rejection of zero rules, nine rules, malformed selectors, zero port count,
  invalid integers, and checked-arithmetic overflow.
- Cross-rank descriptor match, parse failure, and mismatch behavior.
- Same-node topology selection and cross-node exact port-count selection.
- Repeated selectors mapping distinct QPs to the same EID.
- `BuildUDMAInfoImage` indexing for `qpNum` 1, 2, and 3.
- QP-aware remote-memory metadata and QP-specific TPN placement.
- Host query behavior for unavailable, single-QP, and multi-QP transports.
- Device helper source/compile checks and QP0 compatibility wrappers.
- RootInfo parser success and malformed-input coverage without `std::regex`.
- Internal HCCP fault injection for partial queue creation, QP import, local MR
  registration, remote MR import, first unregister failure, and cleanup retry.
- Existing registry, transport layout, demo source, EP, and SDMA isolation tests.

Source guards may enforce invariants such as a unique local registration call
and complete flag initialization, but behavioral fault-injection tests are the
primary evidence for lifecycle correctness.

### Build Verification

- Configure, build, and install from a clean `origin/main` worktree with CANN
  9.1 and the normal optional UDMA test targets.
- Run `git diff --check` and focused CTest targets.
- Confirm built artifacts are identical on both hardware hosts.
- Confirm `libtile-comm.so` does not contain `std::regex` symbols introduced by
  UDMA and has no CANN `devlib` runtime path.

### Hardware Verification

Use `141.61.49.195` and `141.61.49.198` with IPC and SDMA disabled for the UDMA
data-plane cases. The cross-host runner must pass a reachable rank-0 address to
the demo barrier rather than hard-coding loopback.

Required cases:

1. Legacy single-QP registration and PUT with the route variable unset.
2. Exact `2,097,152` byte registration, PUT, quiet, data validation, and
   unregister without error `528101`.
3. Two-QP configuration `port_count:6,port_count:2`, with independent transfers
   and completion on both QPs.
4. Three-QP configuration `port_count:6,port_count:6,port_count:2`, with a
   distinct data slice sent and validated through QP0, QP1, and QP2. The first
   two QPs must have independent SQ/CQ state even though they share an EID.
5. Missing route and mismatched-rank configuration failures without hangs or
   publication of a partial UDMA capability.
6. Successful unregister followed by a second registration to prove reusable
   lifecycle state.

Every success case requires registration success on every rank, kernel/stream
completion, bit-exact data validation, successful unregister, no fallback, no
`528101`, and bounded process completion.

The three-QP case proves that production logic is not fixed to the two-QP
example used by current operator work. Host/simulator tests alone do not prove
UDMA data-plane behavior.

## Delivery Boundaries

The implementation should remain reviewable through buildable commits:

1. Correct Ascend950 registration flags and diagnostics.
2. Make MR lifecycle and cleanup transactional and retryable.
3. Replace RootInfo regex parsing and add deterministic parser tests.
4. Add generic route-spec parsing and cross-rank agreement.
5. Add dynamic per-peer N-QP transport, layout, and resource rollback.
6. Add generic Host discovery and device per-QP APIs.
7. Add cross-host probe support, hardware validation, and this design document.

No spec-only commit is created. The design document is committed with the
implementation after its approval.
