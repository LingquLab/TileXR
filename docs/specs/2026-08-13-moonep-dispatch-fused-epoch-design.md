# MoonEP Dispatch Fused Epoch Design

## Goal

Execute the Hidden payload and optional FP32 route-weight payload of one
`TileXRMoonEpDispatchV2` call in one registered AICore Kernel launch and one UDMA
communication epoch. Hidden-only calls remain one launch. The public C, Torch,
MindSpeed, asynchronous-event, and zero-copy contracts do not change.

## Direct-Launch Contract

The Host validates the mandatory Hidden descriptors and the optional paired
route-weight descriptors, builds one disjoint registered-workspace layout,
obtains one communicator magic, and calls `rtKernelLaunchWithFlagV2` once. The
Kernel ABI carries both payload pointer pairs, both active layouts, separate
diagnostic offsets, one plan status, and one communication configuration.

Standalone route-weight Dispatch is not added. The mandatory primary payload
remains FP16 or BF16 Hidden `[S,H] -> [NvS,H]`; the optional pair remains FP32
`[S,K] -> [NvS]`.

## Workspace

The registered region contains, in this order:

1. Hidden source `[S,H]` and two Hidden receive scratch slots `[NvS,H]`;
2. Weight source `[S,K]` and two Weight receive scratch slots `[NvS]`;
3. shared completion flags and signal source;
4. separate Hidden and Weight Profile arrays;
5. separate Hidden and Weight DFX arrays;
6. shared Kernel status.

All offsets use checked 64-bit arithmetic and 64-byte internal alignment. The
whole registration is rounded to 2 MiB. Binding a larger registered allocation
moves the common tail while preserving disjoint active regions. For
`S=128,K=16,H=3584,NvS=2048`, the result remains 30 MiB after registration
alignment.

## Kernel Flow

1. Validate uniform scalar, pointer, layout, route, transport, and core-count
   arguments without partially entering the communication protocol.
2. Cooperatively stage Hidden source and, when present, Weight source into their
   disjoint registered regions; converge through one `SyncAll`.
3. Load and select the route plan once. For every selected route and logical QP,
   append the Hidden WQE followed by the optional Weight WQE. Hidden uses
   `sourceRow=route/K` and `H*2` bytes; Weight uses `sourceRow=route` and four
   bytes. Both target the same decoded rank and slot in different scratch
   regions.
4. Append the ordered completion WQE after both payload WQEs for that peer/QP.
   Build every WQE in UB, publish complete batches to SQ through MTE3, then ring
   doorbells only with `st_dev` after MTE3 completion.
5. Use one completion-flag exchange, grouped credit progression, CQ recovery,
   and final quiet for the fused epoch. Errors after protocol entry continue the
   bounded convergence path so peers are not stranded.
6. After all receive completions and local-core convergence, zero-fill and copy
   Hidden output, then optional Weight output, from their respective scratch
   slot. Output-copy UB is reused only after explicit pipeline completion/reset.
7. Preserve sticky first-error publication in `plan.status` and write complete
   per-payload diagnostics plus shared Kernel status.

## Diagnostics

Hidden and Weight retain separate Profile and DFX records for compatibility and
payload-specific byte/output counters. A new diagnostic feature bit identifies
a fused epoch. Shared route-selection, flag-wait, credit, CQ, and quiet work is
owned by the final active payload record: Hidden for hidden-only calls and
RouteWeight for paired calls. The non-owning Hidden paired record reports zero
for shared-stage durations instead of duplicating time. Both records carry the
same magic and failure context. Paired shared Kernel status keeps
`payloadMode=RouteWeight`; hidden-only keeps `payloadMode=Hidden`.

## Non-Goals

- Weight-only public Dispatch.
- `WRITE_WITH_NOTIFY` as a weight carrier.
- Multi-SGE or packed Hidden/Weight WQEs.
- Public ABI or MindSpeed adapter changes.
- Removing required local barriers or weakening timeout/error convergence.

## Verification

Host and layout tests prove disjoint ranges, checked arithmetic, one magic, and
one launch. WQE helper tests prove paired counts, address/length selection, QP
split, signal ordering, batching, wrap, and CQ accounting. Source guards retain
UB-only WQE construction, MTE3 SQ publication, and `st_dev` doorbells. Python
tests preserve the one-FFI-call contract and parse fused diagnostics.

The target CANN 9.1 build must compile Host and Kernel. Hardware validation on
`141.61.49.195` first reuses the retained one-time matching official HCCL Test
result, creating it only when that environment has no prior record, and then
covers single-rank, two-rank, and full-host Hidden/paired exactness, repeated
rounds, alternating plans, grouped/group-credit/shared-QP configurations where
supported, and profiling-off/on `pair` A/B evidence.
