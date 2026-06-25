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
bash demo/run_tilexr_udma_p2p_perf.sh 0 1 4096 16777216 2 20 5 0
bash demo/run_tilexr_udma_p2p_concurrency_sweep.sh 4096 16777216 2 20 5 0 1 direct_urma,memory,memory_visible_ack,data_as_flag unidir,bidir 1,2,4,8
```

Arguments:

```text
run_tilexr_udma_demo.sh <test_type> <rank_size> <elements_per_rank> <npu_count> <first_npu>
```

- `test_type=0`: all-gather style UDMA put.
- `test_type=1`: UDMA put with signal.
- `test_type=4`: directed 2-card P2P performance mode. Use
  `demo/run_tilexr_udma_p2p_perf.sh` instead of calling the binary directly.
- `rank_size`: number of local ranks to launch.
- `elements_per_rank`: `int32_t` elements in each rank segment.
- `npu_count`: number of NPUs available to this run.
- `first_npu`: first physical NPU id to use.

Each run writes per-rank logs under `tests/udma/logs/tilexr_udma_demo_*`.
P2P performance runs write logs under `tests/udma/logs/tilexr_udma_p2p_perf_*`
and append CSV rows to `p2p_perf.csv`.

The P2P performance scripts expose four user-facing transports:
`direct_urma`, `memory`, `memory_visible_ack`, and `data_as_flag`. `direct_urma`
uses the current parallel multi-jetty implementation internally; `block_dim=1`
with one QP matches the previous single-QP direct URMA baseline, while
`block_dim=N` with `TILEXR_UDMA_QP_NUM=N` uses up to `N` QPs/jettys in parallel.
`memory_visible_ack` extends `memory` with a receiver-observed tail flag so the
timed window includes peer visibility acknowledgement without copying the
payload back into a separate destination buffer.

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

The host demo process does not include `shmem.h` or call shmem APIs directly. Its process-level synchronization uses a local TCP barrier on `127.0.0.1`, derived from `TILEXR_COMM_ID` with a demo-only port offset.

The demo exits with an error if TileXR initializes without UDMA. In that case, verify that the machine is A5 / Ascend950 / 950 hardware, the Ascend driver and CANN runtime are configured, and the TileXR runtime libraries in `LD_LIBRARY_PATH` match the build under test.

UDMA buffers are allocated with ordinary `aclrtMalloc` and registered through `TileXRUDMARegister`; TileXR IPC `peerMems[]` are printed for diagnostics but are not used as UDMA transfer targets.
