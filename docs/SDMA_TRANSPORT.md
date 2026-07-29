# TileXR SDMA Transport

TileXR SDMA is an opt-in, same-device GM-to-GM copy transport. It is separate
from UDMA: SDMA moves local device memory, while UDMA accesses registered remote
memory on supported A5 / Ascend950 systems.

TileXR supports two runtime-selected SDMA backends:

- Ascend 910A/910B-class A2/A3 devices use the existing PTO SDMA backend.
- Ascend950/A5 devices use TileXR's direct STARS SQ submission backend.

Both backends preserve the same Host and AIV APIs. SDMA remains best-effort: a
backend initialization failure leaves the communicator usable and reports SDMA
as unavailable.

## Requirements And Enablement

The project baseline is CANN 9.1.0 with NPU driver `25.1.rc1` or later. Enable
SDMA before communicator creation:

```bash
export TILEXR_ENABLE_SDMA=1
```

When disabled, `TileXRComm` does not create SDMA resources,
`TileXRSDMAAvailable` returns `false`, and `CommArgs::extraFlag` does not contain
`ExtraFlag::SDMA`.

## Host API

```cpp
bool available = false;
GM_ADDR workspace = nullptr;
TileXRSDMAAvailable(comm, &available);
TileXRGetSDMAWorkspaceDev(comm, &workspace);
```

The workspace is owned by `TileXRComm`; callers must not free it. A successful
query with `available == false` and `workspace == nullptr` means SDMA was
disabled or unavailable for that communicator.

Callers must synchronize all streams that can execute TileXR device APIs before
`TileXRCommDestroy`. Destroy does not cancel an in-flight AIV kernel or SDMA
event; it releases the communicator-owned workspace, STARS streams, and RTSQ
mappings.

## Device API

```cpp
#include "tilexr_sdma.h"

uint64_t event = TileXR::SDMACopyNbi(args, dst, src, bytes, channel);
bool ok = TileXR::SDMAWait(args, event, channel);
```

The API accepts same-device GM pointers. It does not register memory or validate
buffer ownership. Event 0 is the unavailable or invalid no-op result, and
`SDMAWait(args, 0, channel)` succeeds immediately.

`TILEXR_SDMA_AUTO_CHANNEL_GROUP` resolves to the current AIV block index. A5
provides 48 channels numbered 0 through 47. Each channel permits one outstanding
event; different channels can progress concurrently.

## A2/A3 PTO Backend

On A2/A3, `TileXRSDMATransport` owns a PTO
`pto::comm::sdma::SdmaWorkspaceManager` and publishes its device workspace in
`CommArgs::sdmaWorkspacePtr`. Device wrappers create a PTO `SdmaSession`, post
`__sdma_put_async`, and wait through the PTO event helper. Compatibility details
are isolated in `tilexr_sdma_compat.h`.

## A5 Direct Backend

On Ascend950/A5, TileXR owns the Host initializer and AIV submission path. It
does not ship a TileXR AICPU binary or operator package.

During communicator initialization, the Host backend:

1. Dynamically resolves CANN runtime and opapi entry points from `libruntime.so`
   and `libopapi.so`.
2. Creates 48 device-only STARS streams.
3. Calls CANN's built-in `ShmemSdmaStarsQuery` system operator to obtain SQ/CQ
   metadata. Driver `25.1.rc1` may return the known status `507018` after filling
   all fields except the unsupported SQ register-base field; TileXR accepts only
   that exact validated partial result and isolates each follow-up query in a
   disposable context.
4. Cross-checks SQ state with the Host HAL and maps each RTSQ doorbell through
   `halResAddrMap`.
5. Uploads a versioned TileXR-owned 48-channel workspace and publishes it through
   `CommArgs::sdmaWorkspacePtr`.

The AIV backend atomically claims a channel, writes a data-copy SQE followed by a
64-byte completion-copy SQE, cleans the touched cache lines, orders the stores,
and updates the mapped RTSQ tail. `SDMAWait` validates the event generation,
polls the completion record with a fixed upper bound, and releases the channel
only after completion. A timed-out channel remains busy so later submissions
cannot reuse an uncertain queue.

No TileXR target builds or installs a custom OPP, and this feature needs no
custom OPP environment setting. `reference/` remains comparison-only and is not
an include, source, or link dependency.

## Build

Build the core and SDMA tests against CANN 9.1.0:

```bash
bash tests/sdma/build.sh /path/to/cann                 # Ascend910B demo target
bash tests/sdma/build.sh /path/to/cann Ascend950       # A5 demo target
bash tests/sdma/run_tests.sh /path/to/cann
```

The supported demo targets are `Ascend910B` and `Ascend950`. The A5 device kernel
must be compiled with optimization enabled; the repository build uses `-O2`.
Repeated-channel submissions are not reliable with the Bisheng default
low-optimization code generation used in the investigated CANN 9.1.0 setup.

The A5 device kernel does not include PTO headers and does not link
`libnnopbase.so`. The shared Host library retains its PTO dependency so the same
installation can continue to initialize the A2/A3 backend.

The demo bounds Host stream synchronization at 60 seconds. The A5 busy-channel
check is reported only on A5; the PTO build reports that check as skipped.

## Hardware Validation

The runner's second argument is the physical device selected through
`ASCEND_RT_VISIBLE_DEVICES`. On the named A5 server, pass `5`, not logical device
`0`:

```bash
TILEXR_SDMA_DEMO_CHANNEL=0 \
TILEXR_SDMA_DEMO_ITERATIONS=3 \
bash tests/sdma/demo/run_tilexr_sdma_demo.sh /path/to/cann 5 64 4096 1048576

TILEXR_SDMA_DEMO_CHANNEL=47 \
TILEXR_SDMA_DEMO_ITERATIONS=3 \
bash tests/sdma/demo/run_tilexr_sdma_demo.sh /path/to/cann 5 64 4096 1048576

TILEXR_SDMA_DEMO_CHANNEL=0 \
TILEXR_SDMA_DEMO_BLOCKS=4 \
TILEXR_SDMA_DEMO_ITERATIONS=3 \
bash tests/sdma/demo/run_tilexr_sdma_demo.sh /path/to/cann 5 4096

TILEXR_SDMA_DEMO_CHANNEL=1 \
TILEXR_SDMA_DEMO_ITERATIONS=3 \
TILEXR_SDMA_DEMO_REPEATS=3 \
bash tests/sdma/demo/run_tilexr_sdma_demo.sh /path/to/cann 5 4096
```

Every run must report a nonzero event, a successful wait, the expected
generation, and a full byte-for-byte destination comparison. A skipped or
unavailable backend is not an A5 data-plane pass.

## Validation Record

Production A5 acceptance was exercised on:

- hardware: `Ascend950PR_9589` / Ascend950PR;
- CANN: 9.1.0 at `/home/pkg/b061/cann-9.1.T560`;
- driver: `25.1.rc1.b188`;
- physical device: 5;
- initialization: all 48 distinct STARS channels live simultaneously;
- channel 0 and channel 47: 64 B, 4 KiB, and 1 MiB, three iterations each;
- channels 0 through 3: four concurrent AIV blocks at 4 KiB, three iterations;
- channel 1: three init/use/destroy repetitions at 4 KiB, three iterations each;
- busy-channel behavior: a second outstanding submission returned event 0;
- validation: every successful run compared the full source and destination.

The A2/A3 PTO path was previously exercised on Ascend 910B3 with CANN 9.0.0 and
9.1.0 and driver `25.5.0`. That driver value records the tested machine; it is
not TileXR's minimum supported driver.

## Runtime Dependency Checks

Runtime must resolve the real driver HAL, typically:

```text
/usr/local/Ascend/driver/lib64/driver/libascend_hal.so
```

Do not put `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` in runtime RPATH or
RUNPATH. Validate the installed artifacts with:

```bash
ldd install/lib/libtile-comm.so | grep libascend_hal
readelf -d install/lib/libtile-comm.so | grep -E 'RPATH|RUNPATH' || true
readelf -d tests/sdma/install/bin/tilexr_sdma_demo | grep -E 'RPATH|RUNPATH' || true
```

The HAL path must not contain `devlib`. No artifact may depend on a TileXR custom
OPP; the A5 device kernel must not have a `libnnopbase.so` dependency.

## Failure Semantics

Any missing symbol, malformed query result, context-health failure, HAL
cross-check mismatch, RTSQ mapping failure, allocation failure, or upload failure
disables SDMA for that communicator. Cleanup releases only TileXR-owned streams,
mappings, and workspace memory. It does not reset a device or destroy an
application-owned context or stream.
