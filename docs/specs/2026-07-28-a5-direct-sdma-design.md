# A5 Direct SDMA Design

Date: 2026-07-28
Status: Approved on 2026-07-28
Updated: 2026-07-29 (strided batch submission approved for implementation)

## Goal

Add a production Ascend950/A5 backend for TileXR's opt-in local SDMA API. The
backend obtains STARS queue metadata through the CANN system AICPU, maps the
RTSQ doorbells on Host, and lets AIV write and submit A5 SDMA SQEs directly.

The implementation must:

- support CANN 9.1.0 and NPU driver `25.1.rc1` or later;
- preserve the existing A2/A3 PTO backend and existing API behavior;
- add an A5 strided batch device API without changing Host or workspace ABI;
- remain best-effort when any A5 capability is unavailable;
- ship no TileXR OPP package and require no custom OPP environment variable;
- keep `reference/` comparison-only.

The validated baseline instance is driver `25.1.rc1.b188` on
`Ascend950PR_9589`. `25.1.rc1` is the project-wide minimum driver version, not
an exception specific to SDMA.

## Delivery Boundary

TileXR ships the A5 Host backend and AIV submission implementation in its
normal library and installed headers. It does not build, install, or load a
TileXR custom AICPU SO or OPP package.

The backend dynamically resolves the public CANN opapi entry points for the
built-in `ShmemSdmaStarsQuery` operator. That operator and its system AICPU
kernel remain CANN-provided runtime dependencies. No active TileXR target
includes or links source from `reference/`, modifies the CANN installation, or
sets `ASCEND_CUSTOM_OPP_PATH`.

An embedded `.aicpu` SO is not a substitute for the built-in query. CANN
registers binaries loaded through `aclrtBinaryLoadFromData` as custom AICPU
kernels. Hardware probing on the baseline host showed that this process cannot
see a Host-created STARS SQ: head, tail, depth, SQE size, and base queries all
returned status 3. The system AICPU query can see those resources.

## Public Compatibility

The following contracts remain unchanged:

- opt-in enablement through `TILEXR_ENABLE_SDMA=1`;
- `CommArgs::sdmaWorkspacePtr` and `ExtraFlag::SDMA`;
- `TileXRSDMAAvailable` and `TileXRGetSDMAWorkspaceDev`;
- device calls `SDMACopyNbi` and `SDMAWait`;
- event handle 0 as the unavailable or invalid no-op result.

The installed device header additionally exposes:

```cpp
uint64_t SDMACopyStridedNbi(
    const __gm__ CommArgs* args,
    __gm__ uint8_t* dst,
    __gm__ uint8_t* src,
    uint64_t bytes,
    uint32_t copyCount,
    uint64_t dstStrideBytes,
    uint64_t srcStrideBytes,
    uint32_t channelGroupIdx = TILEXR_SDMA_AUTO_CHANNEL_GROUP);
```

For copy `i`, the source and destination are `src + i * srcStrideBytes` and
`dst + i * dstStrideBytes`. `copyCount == 1` delegates to `SDMACopyNbi` and
preserves its behavior. On A5, `copyCount > 1` submits one ordered batch. On
A2/A3, multi-copy batch submission is unavailable and returns event 0; the
existing PTO single-copy path is unchanged.

For `copyCount > 1`, both strides must be at least `bytes`, every individual
length must fit the A5 SQE, address arithmetic must not overflow, and the
channel must have room for `copyCount + 1` SQEs. These conservative rules keep
all source and destination slices distinct. The API does not promise broadcast
or overlapping-copy semantics.

Host selects the backend by runtime SoC:

- A2/A3 continues to use PTO `SdmaWorkspaceManager` and PTO device intrinsics;
- Ascend950 uses the TileXR A5 backend described here;
- unsupported or failed initialization leaves SDMA unavailable without failing
  communicator initialization.

## Host Initialization

### Owned resources

One A5 transport owns, for the communicator lifetime:

- 48 `ACL_STREAM_DEVICE_USE_ONLY` STARS streams;
- the stream, SQ, CQ, logical-CQ, and physical-die identifiers;
- one `PROCESS_CP1 + RES_ADDR_TYPE_STARS_RTSQ` mapping per SQ;
- the final TileXR A5 workspace in ordinary device memory;
- temporary query buffers and isolated query contexts used only during init.

The CQ identifiers are validation and diagnostic fields. The data plane uses
`wrCqe=0` and completion-copy SQEs, so it does not consume a CQ base address.

### Capability-driven query

Host first presents all 48 streams to one built-in query in an isolated ACL
context.

1. If launch and stream synchronization succeed, Host validates every channel
   and takes the complete result as the fast path.
2. On driver `25.1.rc1`, the query fills the first channel and then fails while
   requesting unsupported `DRV_SQCQ_PROP_SQ_REG_BASE`. Host may accept this
   partial result only when synchronization returns exactly `507018`, the
   register-base field is zero, and all earlier fields are complete.
3. After a partial batch result, Host queries every remaining stream separately
   in a fresh isolated context. Each query writes into its own workspace slot.
4. After every expected partial failure, Host destroys the failed context,
   restores the communicator context, and performs a lightweight runtime health
   operation before continuing.

The compatibility path is based on observed behavior, not on version-string
branching. A four-channel probe on `25.1.rc1.b188` initialized four simultaneously
live SQs successfully. The production acceptance gate extends this to all 48.

### Validation and RTSQ mapping

For each channel, Host requires:

- successful local-device, head, tail, depth, SQE-size, and SQ-base statuses;
- matching stream/SQ/CQ/logical-CQ/device identifiers;
- a nonzero SQ base, 64-byte SQE size, and depth of at least three;
- head and tail inside the reported depth;
- Host HAL tail and SQE-size cross-checks matching the AICPU result;
- a nonzero RTSQ mapping of at least four bytes.

The Host mapping replaces the unsupported query result for `sq_reg_base`.
Partial data is never accepted after an unexpected status, malformed field,
context restore failure, or failed health check.

After all channels validate, Host copies a TileXR-owned workspace to GM and
publishes its address through `CommArgs`. Query intermediates are then freed;
STARS streams and RTSQ mappings remain alive until transport shutdown.

## A5 Workspace ABI

The A5 workspace is independent of PTO and uses fixed-width, 64-byte-aligned
structures shared by Host and AIV. It contains:

- a header with magic, ABI version, backend kind, channel count, and SQE size;
- 48 channel records with SQ base, RTSQ address, depth, current tail, IDs,
  generation, and outstanding state;
- one 64-byte completion payload and one 64-byte completion record per channel;
- reserved space for ABI-compatible diagnostics.

Host and device compilation enforce sizes and key offsets with static
assertions. Every pointer stored in the workspace is a device-visible address.
The ABI version changes whenever an existing field's meaning or layout changes.

Each channel allows one outstanding event. Different channels are independent
and may progress concurrently. The default channel group resolves from the AIV
block index and must be less than 48; explicit channel groups follow the same
range rule.

## AIV Submission

`SDMACopyNbi` on A5 performs these steps:

1. Validate the workspace ABI, pointers, byte count, and channel group.
2. Atomically claim the channel's outstanding state and advance its nonzero
   generation. A busy channel returns event 0.
3. Write the generation into the channel's completion payload. The completion
   record is not cleared because generations are monotonic and the completion
   SQE overwrites the full record before it can match the new generation.
4. Build two consecutive 64-byte A5 SQEs at the current tail: one data copy and
   one 64-byte copy from the payload to the completion record.
5. Set SQE type 11, `wrCqe=0`, kernel credit 254, and the validated A5 address,
   substream, length, and task fields.
6. Clean the written payload and SQE cache lines, execute the required
   store-ordering barrier, then write the new tail to the mapped RTSQ.
7. Return an event encoding the channel and generation.

The tail advances modulo the queried depth. Completion of the second SQE makes
the channel safe for reuse. Transfer length handling must respect the A5 SQE
field width; unsupported lengths return event 0 instead of truncating.

`SDMAWait` rejects malformed or stale events, polls the channel completion
generation through `ReadGmByPassDCache`, then releases the outstanding state.
A zero event remains an immediate successful no-op.

### Strided batch submission

`SDMACopyStridedNbi` uses the same channel claim, event encoding, completion
record, and wait path. After all validation succeeds, it:

1. reserves `copyCount + 1` entries while leaving one SQ slot unused;
2. writes `copyCount` ordered data SQEs with consecutive task IDs;
3. appends one 64-byte completion-copy SQE;
4. cleans the payload and every written SQE cache line;
5. advances the tail and task ID by `copyCount + 1` modulo their hardware
   widths;
6. rings one doorbell and returns one event for the complete batch.

No validation failure after the channel claim is allowed. All pointer, stride,
length, count, queue-capacity, and workspace checks therefore occur before the
atomic claim. `SDMAWait` releases the channel only after the final completion
SQE, so every data SQE in the batch is complete when the event becomes visible.

The returned event represents the whole batch. Reported per-copy latency is
total submit-through-wait time divided by `copyCount`; callers cannot observe
or wait for an individual copy through this event.

## Batch Performance Evidence

The batch decision was validated on `Ascend950PR_9589`, CANN 9.1.0, driver
`25.1.rc1.b188`, physical device 5. The benchmark used AIV
`GetSystemCycle()` at 1000 cycles/us, a rotating 64 MiB source/destination
working set with the same allocation policy for single and batch runs, device
warmup before sampling, ten samples, and full working-set byte comparison.
Host launch, initialization, allocation, H2D/D2H, and stream synchronization
were outside the timed region. SQE construction, cache maintenance, doorbell,
device completion, and channel release were included.

| Bytes per copy | Single completion | Batch 16 amortized | Speedup |
| ---: | ---: | ---: | ---: |
| 4 KiB | 3.351 us | 0.954 us | 3.51x |
| 8 KiB | 3.372 us | 0.975 us | 3.46x |
| 16 KiB | 3.415 us | 0.972 us | 3.51x |
| 32 KiB | 4.133 us | 1.031 us | 4.01x |
| 64 KiB | 3.453 us | 1.012 us | 3.41x |
| 1 MiB | 4.314 us | 1.962 us | 2.20x |
| 4 MiB | 8.752 us | 6.313 us | 1.39x |

For 4-64 KiB, callers should batch naturally available copies because fixed
submission and completion costs dominate. From 64 KiB through 1 MiB, batching
still materially improves throughput but callers must not delay an otherwise
ready copy merely to fill a batch. Above 1 MiB, batching is optional: use it
for already-available parallel work, while latency-sensitive isolated copies
should remain single submissions. For example, 16 copies at an amortized
0.954 us still expose one batch completion after roughly 15.3 us.

## Failure and Cleanup

All A5 initialization failures are capability failures, not communicator
failures. TileXR logs the first failed boundary, releases only resources it
owns, leaves `sdmaWorkspacePtr` null, clears `ExtraFlag::SDMA`, and continues
with existing communication paths.

Shutdown restores the owning device/context as needed, unmaps RTSQ mappings,
destroys STARS streams, frees the workspace, and is idempotent after partial
initialization. It never resets a device or destroys an application-owned
stream or context.

Device submission failures are contained to the SDMA call. They must not write
a doorbell after failed validation or expose a nonzero event before every SQE
in the submission is ready.

## Driver Baseline Migration

Repository requirements, architecture diagrams, build verification, and CI
provisioning change from `25.5.0 or later` to `25.1.rc1 or later`.

The shared CI version comparator must recognize Huawei release-candidate forms:

- accept `25.1.rc1`, `25.1.RC1`, and build-suffixed forms such as
  `25.1.rc1.b188`;
- accept later RCs, final `25.1.x` releases, and later major/minor releases;
- reject versions earlier than RC1 and malformed or incomplete strings.

Occurrences of `25.5.0` that record an actual 910B validation environment remain
historical facts. Only minimum-version claims and enforcement are changed.

## Verification

Implementation acceptance requires:

- C++14 build with CANN 9.1.0 for the core library and Ascend950 demo kernel;
- unit coverage for A5 ABI layout, backend selection, partial-query validation,
  cleanup, invalid events, busy channels, and best-effort fallback;
- CI tests for the `25.1.rc1` version-ordering matrix and provisioning messages;
- no TileXR OPP artifact, vendor install directory, or custom OPP environment;
- 48 distinct, simultaneously live, validated channels on the baseline host;
- direct copies of 64 B, 4 KiB, and 1 MiB on channel 0 and channel 47;
- strided batches with counts 1, 2, 4, 8, 16, and 32, including SQ/tail/task-ID
  wraparound and full comparison of every destination slice;
- representative 4 KiB through 4 MiB batch performance using equal backing
  allocation and a rotating working set;
- a concurrent multi-block run using distinct channels, followed by full byte
  comparisons and matching completion generations;
- repeated init/use/shutdown loops with a healthy communicator context;
- real driver `libascend_hal.so` resolution and no CANN `devlib` RPATH/RUNPATH.

Host/source tests and simulator runs do not prove the A5 data plane. Hardware
claims must name the exercised SoC, driver, device, channel count, sizes, and
concurrency level.

## Repository Cleanup

The implementation change replaces the PoC-only material with this design and
the production backend. It removes:

- `tests/sdma/a5_aicpu_probe/`;
- the A5 PoC and obsolete adaptation plans under `docs/plans/`;
- the temporary Ascend950 preflight that disables the PTO path;
- documentation that describes the production A5 path as pending or requires
  driver 25.5.0.

The existing SDMA demo becomes the single data-plane acceptance entry point for
both A2/A3 and A5.
