# TileXR UDMA Communication Demo

This demo shows TileXR-initialized UDMA communication with verbose diagnostics. UDMA means UnifiedBus DMA and this runtime path currently targets A5 / Ascend950 / 950 hardware. The host demo uses TileXR public APIs, registers ordinary `aclrtMalloc` memory with `TileXRUDMARegister`, and the AICore kernel uses `tilexr_udma.h`.

## Build

```bash
cd /path/to/TileXR/tests/udma
bash build.sh
```

The demo target requires `bisheng`. If `bisheng` is not available, `build.sh` still builds the existing UDMA tests and reports that `tilexr_udma_demo` was skipped.

## Run

```bash
cd /path/to/TileXR/tests/udma
bash demo/run_tilexr_udma_demo.sh 0 2 16 2 0
bash demo/run_tilexr_udma_demo.sh 1 2 16 2 0
```

Arguments:

```text
run_tilexr_udma_demo.sh <test_type> <rank_size> <elements_per_rank> <npu_count> <first_npu>
```

- `test_type=0`: all-gather style UDMA put.
- `test_type=1`: UDMA put with signal.
- `rank_size`: number of local ranks to launch.
- `elements_per_rank`: `int32_t` elements in each rank segment.
- `npu_count`: number of NPUs available to this run.
- `first_npu`: first physical NPU id to use.

Each run writes per-rank logs under `tests/udma/logs/tilexr_udma_demo_*`.

For a true cross-node data-channel probe, build and install the demo on every host, then launch all ranks as one MPI job:

```bash
bash demo/run_tilexr_udma_data_channel_probe_mpi.sh \
  --hosts 141.62.19.156:4,141.62.19.108:4 \
  --rank-size 8 \
  --comm-id 141.62.19.156:10067 \
  --devices 0,1,2,3 \
  --npu-count 4 \
  --first-npu 0 \
  --test-type 0 \
  --elements 16
```

Run again with `--test-type 1` to validate UDMA put-signal. Add `--require-sdma` if the test should also fail when TileXR SDMA is unavailable. This probe validates the cross-node UDMA/SDMA data-channel prerequisites; it does not mean EP dispatch itself has a cross-node backend.

Run this demo only on A5 / Ascend950 / 950 hardware. Builds or smoke tests on other Ascend chips are not valid UDMA runtime validation.

## What To Check

Each rank prints:

- process id, rank, selected device, `TILEXR_COMM_ID`, and `LD_LIBRARY_PATH`
- TileXR `CommArgs` host/device pointers
- `extraFlag`, UDMA enable bit, and `udmaInfoPtr`
- `peerMems[]` IPC pointers for comparison
- registered data and signal buffer addresses
- kernel debug words
- result samples and signal values

The host demo process does not include `shmem.h` or call shmem APIs directly. Its process-level synchronization uses a TCP barrier derived from `TILEXR_COMM_ID` with a demo-only port offset. Set `TILEXR_DEMO_BARRIER_ADDR=<ip[:port]>` when ranks on other hosts must connect to the rank-0 barrier server.

The demo exits with an error if TileXR initializes without UDMA. In that case, verify that the machine is A5 / Ascend950 / 950 hardware, the Ascend driver and CANN runtime are configured, and the TileXR runtime libraries in `LD_LIBRARY_PATH` match the build under test.

UDMA buffers are allocated with ordinary `aclrtMalloc` and registered through `TileXRUDMARegister`; TileXR IPC `peerMems[]` are printed for diagnostics but are not used as UDMA transfer targets.
