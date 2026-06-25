# UDMA All-to-All Demo Design

## Goal

Add an all-to-all UDMA operator demo under `tests/udma/demo`.

The demo validates the common all-to-all layout:

- Each rank owns `rank_size` equal input slices.
- Input slice `dst_rank` from rank `src_rank` is sent to rank `dst_rank`.
- Each destination rank writes output slices ordered by source rank.

For rank `i`, input is laid out as `[to0, to1, ..., toN-1]`. For rank `j`, output is laid out as `[from0, from1, ..., fromN-1]`.

## Approach

Extend the existing `tilexr_udma_demo` binary instead of creating a separate demo.

The current demo already handles:

- multi-process local rank launch
- TileXR communicator initialization
- UDMA capability checks
- ordinary `aclrtMalloc` memory registration through `TileXRUDMARegister`
- local TCP barriers for demo synchronization
- per-rank logs and result validation

The all-to-all path will be selected with `test_type=2`. Existing `test_type=0` all-gather and `test_type=1` put-signal behavior must remain unchanged.

## Host Data Layout

For `test_type=2`, the host allocates one registered device-memory payload containing:

- input buffer: `rank_size * elements_per_peer` `int32_t` values
- output buffer: `rank_size * elements_per_peer` `int32_t` values
- signal/debug space if needed by the shared demo structure

The registered allocation remains rounded up to the existing 2 MiB UDMA registration alignment.

Input initialization for rank `src`:

```text
input[dst][elem] = 100000 + src * 1000 + dst
```

Expected output for rank `dst`:

```text
output[src][elem] = 100000 + src * 1000 + dst
```

This makes source and destination rank mistakes visible in validation logs.

## Kernel Behavior

Add a new AICore kernel and launch wrapper in `tilexr_udma_demo_kernel.cpp`.

The kernel reads `rank`, `rankSize`, and UDMA registry state from `CommArgs`.

For each peer:

- If `peer == rank`, copy the local input slice `input[rank]` into local output slice `output[rank]`.
- Otherwise, issue `TileXR::UDMAPutNbi<int32_t>` to write local `input[peer]` into the remote rank's registered output slice for this source rank.
- Call `TileXR::UDMAQuiet(args, peer)` after posting to each remote peer.

The remote byte offset is computed against the peer rank's registered base:

```text
output_offset + rank * elements_per_peer * sizeof(int32_t)
```

The local source pointer is:

```text
input + peer * elements_per_peer
```

## Synchronization

The demo keeps the existing host-side synchronization:

1. Each rank initializes and registers its buffers.
2. Host barrier ensures every rank's registered-memory metadata is visible.
3. Each rank launches the all-to-all kernel and synchronizes its stream.
4. Host barrier ensures all ranks have completed UDMA writes.
5. Host copies output back and validates.

The kernel does not add device-side inter-rank polling beyond `UDMAQuiet`.

## Build And Run

The existing `tests/udma/CMakeLists.txt` continues to build one demo kernel shared object and one `tilexr_udma_demo` executable.

Update the run script and README so:

```bash
bash demo/run_tilexr_udma_demo.sh 2 2 16 2 0
```

runs the all-to-all path.

## Verification

Local checks:

- Build metadata remains scoped to `tests/udma`.
- Existing all-gather and put-signal source paths remain intact.

Remote hardware validation:

- Create a new directory under `/home/aiv-perf/` on `root@141.61.95.18`.
- Copy or sync the repository into that directory.
- Build TileXR core and `tests/udma`.
- Run `bash demo/run_tilexr_udma_demo.sh 2 2 16 2 0`.
- If resources permit, also run a wider case such as `rank_size=4`.

Success requires every rank to print `TileXR UDMA demo success` and all output segments to match the expected all-to-all pattern.

## Out Of Scope

- Refactoring the demo runtime into shared helper classes.
- Adding a production all-to-all collective API.
- Optimizing multi-peer posting or batching.
- Supporting non-`int32_t` element types in this demo.
