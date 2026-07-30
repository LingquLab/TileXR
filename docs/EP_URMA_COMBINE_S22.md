# EP URMA Combine for Ascend 950

This change adds a TileXR-native Mixture-of-Experts combine operator backed by
URMA/UDMA. The production implementation is fixed to the validated S22 layout:

- 42 Pack/Receive AIVs and 22 Send AIVs;
- 22 independent UDMA queue pairs;
- one WQE per doorbell;
- QDC-v3 quant/dequant;
- round-robin receive scheduling with sticky ready state;
- parallel release publication and deferred same-parity credit checks;
- first-launch start gate after a stream synchronization;
- BiSheng `-O2`, with profiling disabled by default.

The fixed configuration avoids carrying experiment-only variant switches into
the production build.

## Build

Configure and build through CMake. No variant or wrapper script is required.

```bash
cmake -S . -B build-s22 \
  -DTILEXR_BUILD_EP=ON \
  -DTILEXR_EP_SOC_TYPE=ascend950 \
  -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/install-s22"
cmake --build build-s22 -j --target tilexr-ep
cmake --install build-s22
```

`ASCEND_HOME_PATH` and `ASCEND_DRIVER_PATH` must point to the active CANN
toolkit and driver before configuration. A profiling build is available only
when explicitly requested with `-DTILEXR_EP_ENABLE_PROFILING=ON`; it is not the
production default.

## API

The public header `src/include/tilexr_ep.h` provides:

- `TileXRMoeEpCombineUrmaGetWorkspaceSize` for the registered workspace size;
- `TileXRMoeEpCombineUrma` for the production launch;
- `TileXRMoeEpCombineUrmaProfile` for an explicitly profiled launch.

The workspace must be aligned and registered with TileXR UDMA before launch.
The operator supports an eight-rank Ascend 950 deployment and uses the existing
TileXR communicator for rank information and registered-memory exchange.

## Performance Evidence

The production comparison metric is profiling-free `strictKernelCycles`,
aggregated as max core per rank, max rank per launch, then median across 100
launches. For BS128, H=5120, top-k=6, rank-size=8 and enqueue-window=1, S22
measured 94,889 cycles (94.889 us using 1000 cycles/us).

The single retained detailed report is
[BS128 S22 profile](performance/tilexr_ep_urma_combine_s22_bs128.html). The
default proportional timeline excludes the start-gate bar and rebases the axis
to steady-state work; start-gate measurements remain in report metadata.
