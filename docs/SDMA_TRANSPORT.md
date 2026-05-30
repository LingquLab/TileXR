# TileXR SDMA Transport

TileXR SDMA transport provides a first-class local on-card GM-to-GM copy path.
It is separate from UDMA: SDMA is local to one device, while UDMA targets
registered remote memory on supported A5/Ascend950 systems.

## Enablement

SDMA is disabled by default. Enable it explicitly:

```bash
export TILEXR_ENABLE_SDMA=1
```

When disabled, `TileXRComm` initialization does not create STARS streams and
`TileXRSDMAAvailable` reports `false`. Enabled initialization is best-effort:
if PTO SDMA headers or runtime resources are unavailable, normal communicator
initialization continues without setting `ExtraFlag::SDMA`.

## Host API

```cpp
bool available = false;
GM_ADDR workspace = nullptr;
TileXRSDMAAvailable(comm, &available);
TileXRGetSDMAWorkspaceDev(comm, &workspace);
```

The workspace pointer is owned by `TileXRComm`; callers must not free it. A
successful query with `available=false` and `workspace=nullptr` means SDMA was
disabled or unavailable for that communicator.

## Device API

```cpp
#include "tilexr_sdma.h"

uint64_t event = TileXR::SDMACopyNbi(args, dst, src, bytes, 0);
bool ok = TileXR::SDMAWait(args, event, 0);
```

The API accepts raw same-device GM pointers. It does not register memory and
does not validate whether pointers belong to TileXR communication buffers.

`SDMACopyNbi` returns event handle `0` when SDMA is disabled or arguments are
invalid. `SDMAWait(args, 0, channelGroup)` returns `true`, matching the no-op
completion path. The default channel group is
`TILEXR_SDMA_AUTO_CHANNEL_GROUP`, which resolves to the current AI Core block
index; the demo passes channel group `0` explicitly.

## Implementation Notes

`TileXRComm::InitSDMA()` owns a `TileXRSDMATransport` beside the existing UDMA
transport. When `TILEXR_ENABLE_SDMA=1`, the transport creates a PTO
`pto::comm::sdma::SdmaWorkspaceManager`, obtains its device workspace address,
stores that address in `CommArgs::sdmaWorkspacePtr`, and sets
`ExtraFlag::SDMA`.

Device wrappers build a PTO `SdmaSession` from `args->sdmaWorkspacePtr`, post
`__sdma_put_async`, and wait with PTO's SDMA event wait helper. TileXR uses a
256-byte scratch tile and current defaults of one queue,
1 MiB block bytes, and communication block offset `0`.

## CANN Compatibility

The implementation targets CANN 9.0.0 and CANN 9.1.0. TileXR isolates PTO SDMA
header differences in `tilexr_sdma_compat.h`.

Runtime must load `libascend_hal.so` from the driver path, typically:

```text
/usr/local/Ascend/driver/lib64/driver/libascend_hal.so
```

Do not put `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` into runtime RPATH. The
devlib HAL can cause `aclInit` failures such as `500000` with `init soc version
failed`.

## Build And Test

Build tests against a selected CANN install:

```bash
bash tests/sdma/build.sh /path/to/cann
bash tests/sdma/run_tests.sh /path/to/cann
```

Run data-plane demo:

```bash
bash tests/sdma/demo/run_tilexr_sdma_demo.sh /path/to/cann 0 64 4096 1048576
```

Expected demo success line:

```text
PASS TileXR SDMA copied <bytes> bytes correctly
```

## Acceptance

For release validation, run the unit tests and demo against both CANN versions:

```bash
bash tests/sdma/build.sh "${TILEXR_CANN_90_HOME}"
bash tests/sdma/run_tests.sh "${TILEXR_CANN_90_HOME}"
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_90_HOME}" 0 64 4096 1048576

bash tests/sdma/build.sh "${TILEXR_CANN_91_HOME}"
bash tests/sdma/run_tests.sh "${TILEXR_CANN_91_HOME}"
bash tests/sdma/demo/run_tilexr_sdma_demo.sh "${TILEXR_CANN_91_HOME}" 0 64 4096 1048576
```

## Current Validation Status

Local validation in this branch:

- CANN 9.1.0 build passed with `TILEXR_HAVE_PTO_SDMA: ON`.
- CANN 9.1.0 SDMA unit tests passed.
- Local demo binary and kernel built.
- Local demo runtime is blocked at `aclInit ret=500000` because this environment
  has no usable driver HAL/device runtime.

Pending validation:

- CANN 9.0.0 build and unit tests; no local CANN 9.0 install is present in this
  workspace.
- Actual SDMA data-plane copy on hardware. Run this on `blue` or another target
  Ascend host before claiming demo copy success.
- Two-version release acceptance, including CANN 9.0.0 and CANN 9.1.0 unit
  tests plus demo copy success.

## Deferred Stress And Performance Scope

Deferred validation scope includes multi-block channel-group assignment,
multi-stream concurrency, long-loop stability, parameter matrix tests for queue
settings, and MTE/SDMA bandwidth and latency comparisons.

## TODO: C And Performance Parameter Testing

Future validation should cover:

- C-facing API and parameter shape, if SDMA is exposed beyond the current C++
  device wrapper and host query API.
- `queue_num`, `block_bytes`, and `channelGroup` matrix coverage.
- Multiple transfer sizes, stream counts, loop counts, and concurrency levels.
- Runtime perf counters and DFX logs sufficient to distinguish SDMA issue,
  wait, and copy completion costs.
- Bandwidth and latency comparisons against MTE `DataCopy`.
- Two-version CANN acceptance for CANN 9.0.0 and CANN 9.1.0.
