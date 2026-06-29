# UDMA Multi-EID Route Design

## Scope

Port the multi-EID direct UDMA P2P route capability from PR 45 commit range
`06cd19c42846eb79ab75b5a075b82d9b656c2852..07afc489467e7e9a369eac6679913de208d169b1`
into the current `codex/udma-bigdata-isolated` branch.

The port is intentionally limited to the UDMA route/layout/device wrapper and
UDMA-specific tests. It must not merge unrelated PR 45 checker, collectives,
documentation, or demo restructuring changes.

## Goals

- Keep the existing single-route behavior as the default fallback.
- Add an opt-in multi-route policy for direct UDMA P2P.
- Discover and diagnose multiple local EID routes per peer from HCCL
  root/topology data.
- Allow explicit EID selection and ordering for diagnostics.
- Expand queue metadata so one peer can use multiple EID routes concurrently.
- Weight per-QP transfer slices by route link capacity, enabling mixed
  6-port and 2-port routes to be used concurrently with proportional work.

## Non-Goals

- Do not replace the current bigdata alltoall work.
- Do not merge the full PR 45 branch.
- Do not change public host APIs.
- Do not require UDMA on unsupported hardware; graceful fallback remains.

## Route Selection

`TileXRUDMATransport::BuildRoutes()` keeps the current single-route resolution
when no multi-route policy is requested.

When `TILEXR_UDMA_ROUTE_POLICY=all`, the transport:

1. Parses HCCL root/topology data.
2. Resolves all local EID candidates for each peer from topology ports.
3. Falls back to aggregate local EIDs when peer-specific edge data is missing.
4. Applies `TILEXR_UDMA_ROUTE_EIDS` if provided, preserving the requested order
   but only accepting valid candidates.
5. Caps the route count with `TILEXR_UDMA_MAX_EIDS_PER_PEER`.
6. Exchanges local route lists so each rank knows the peer's matching remote
   EIDs.

The first selected EID remains populated in the existing `peerLocalEid_` and
`peerRemoteEid_` maps for compatibility with fallback paths.

## Queue and Memory Model

The transport stores per-peer local and remote EID vectors. It creates RA
contexts for all selected local EIDs, then creates `qpsPerRoute_` queues per
selected route. `qpNum_` becomes the maximum expanded QP route count across
peers.

Memory registration remains local-address based, but registration/import is
tracked per EID. Remote memory handles are stored per peer and per route so
cleanup can unimport each handle through the matching local EID context.

## Device Layout

`UDMAInfo` gains `qpWeightPtr`, pointing to a device-visible `uint32_t` array
with the same `(rank, qpIdx)` indexing as SQ/RQ/CQ/memory metadata.

`BuildUDMAInfoImage()` gains an overload that accepts:

- explicit `qpNum`
- SQ/RQ/SCQ/RCQ vectors
- memory vector
- QP weight vector

The existing overload remains and defaults all weights to `1`, preserving
single-route and older test behavior.

## Weighted Device Slicing

Device code adds `UDMAGetQpWeight()`. If no weight pointer is present, or a
weight entry is zero, the effective weight is `1`.

The UDMA P2P perf kernels replace equal WQE slicing with weighted slicing:

- Sum all QP weights for the peer.
- Compute each QP's byte range by weight proportion.
- Align slice boundaries to `BLOCK_UNIT_BYTE`.
- Assign any rounding remainder to the final QP.

This allows a 6-port route and a 2-port route to both post concurrently while
the 6-port route receives proportionally more payload.

## Tests

Update the UDMA transport layout unit test to cover:

- `qpWeightPtr` placement and serialized values.
- Multi-route QP-to-EID expansion.
- Route weight expansion from topology-derived port counts.
- Explicit EID selection parsing, filtering, de-duplication, and ordering.
- Existing mismatched-array rejection and large-transfer chunk behavior.

Build/runtime validation depends on the existing repository environment. Host
layout tests can run without Ascend hardware. Direct UDMA P2P runtime validation
still requires supported A5 / Ascend950-class hardware.

