# Grouped AllToAll SIMT Send Design

## Goal

Migrate the validated grouped AllToAll SIMT send path from `b8f0ed4` into the
current `a36e91d` architecture without replacing newer receive/copyout,
credit, synchronization, SDMA, shared-QP, multi-region, or trace behavior.

The reference path was validated with SIMT, SDMA, credit, and multi-region
enabled on 32 ranks with a 1 GiB payload per rank.

## Scope

- Add an opt-in `TILEXR_DEMO_ALLTOALL_GROUP_SIMT=0/1` demo switch. The default
  remains the current non-SIMT sender.
- Use one physical send AIV core with 32 SIMT threads when SIMT is enabled.
  Keep the existing 32 send workers when it is disabled.
- Port the `b8f0ed4` SIMT task construction, SQ reservation, WQE construction,
  doorbell, completion polling, and multi-region payload/signal ordering.
- Resolve the registry through `CommArgs::udmaRegistryPtr`; do not add another
  device registry allocation or ownership path.
- Extend all four current grouped kernel variants (normal, batch, credit, and
  batch-credit) with the same SIMT mode ABI.
- Continue launching the embedded kernel with `rtKernelLaunchWithFlagV2`.
  SIMT launches reserve 64 KiB for DCache and assign the remaining per-vector
  core UBuf as dynamic local memory.
- Preserve current grouped trace events and add only the SIMT events needed to
  diagnose build, post, quiet, or completion failures.

## Preserved Behavior

- Ready-driven single-core SDMA copyout and 32-core MTE fallback.
- Optional ingress credit and the terminal credit/`SyncAll` protocol.
- Optional SQ-page prewarming.
- Region-aware shared QP reuse; SIMT must not increase host-created QP count.
- Single/multi route selection, group width, pass handling, quiet-batch
  selection, kernel specializations, timeout/error recording, and default
  behavior when SIMT is disabled.
- Existing dirty changes in `src/include/tilexr_sdma_a5.h` and the trace parser
  remain outside this migration unless compilation requires a narrowly scoped
  compatibility adjustment.

## Data Flow

For each group and pass, the single SIMT send core uses 32 threads to prepare
the active primary/secondary tasks. Tasks reserve their shared SQ slots,
construct WQEs, publish the queue head/doorbell, and poll the expected
completion count.

When payload and signal resolve to different registered regions, payload WQEs
are submitted and completed first, followed by ordered signal WQEs. This keeps
the receive flag from becoming visible before payload completion. A single
region may use the combined proven path.

The receive side is unchanged: current copyout workers wait for the existing
tokens, publish credits at the existing points, perform MTE or SDMA copyout,
and participate in the current terminal synchronization.

## Host And Kernel ABI

The host reads the SIMT switch, derives `sendWorkers` as 1 or 32, and computes
`blockDim = sendWorkers + copyoutWorkers`. Kernel-side receive worker indexes
must use this runtime send-worker boundary rather than the legacy constant.

The explicit launcher adds `simtMode` to both credit and non-credit argument
structures and keeps their offsets guarded by static assertions. For SIMT,
the launcher queries `ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE`, rejects an invalid or
too-small value, and sets dynamic local memory to `totalUbSize - 64 KiB` in the
`rtKernelLaunchWithFlagV2` task configuration.

## Compatibility And Failure Handling

- `SIMT=0` must retain the current ABI-selected kernel behavior and results.
- `SIMT=1` requires a valid UDMA registry and supported UBuf size; invalid
  configuration fails before launch or records the current grouped config
  error instead of silently falling back.
- SIMT is initially supported for the same group widths, routes, pass counts,
  quiet batches, credit mode, and SDMA modes already accepted by the current
  grouped implementation.
- No new QP, registered-memory region, or IPC allocation is introduced.

## Verification

1. Build the grouped demo and run its layout/source-contract unit tests.
2. Verify kernel argument sizes/offsets, SIMT/non-SIMT block dimensions, and
   the 64 KiB DCache reservation path.
3. Compile with the target b131 CANN headers and inspect that the explicit
   launch uses `rtKernelLaunchWithFlagV2` with the intended local-memory size.
4. On A10 2x8, compare SIMT off/on for 1 KiB single, 256 MiB multi, and 1 GiB
   multi, using single pass and the same warmup/repeat parameters.
5. Require every rank to return zero, print the grouped success marker, and
   pass correctness checks. Capture trace only after trace-off functionality
   succeeds.

## Non-Goals

- Replacing the current SDMA copyout scheduler or terminal synchronization.
- Importing unrelated history from the `b8f0ed4` branch.
- Changing QP provisioning, region allocation, route weights, or registered
  memory capacity.
- Enabling SIMT by default.
