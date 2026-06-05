# TileXR Standalone Collectives Design

## Context

`src/collectives` currently provides the optional `libtilexr-collectives.so` library with two standalone collective APIs:

- `TileXRAllGather`
- `TileXRAllToAll` for equal per-peer counts

The implementation is adapted from the lcal sources under `reference/ascend-transformer-boost/src/kernels/lcal`. TileXR intentionally keeps this library separate from `libtile-comm.so`; communicator setup, shared peer memory, `CommArgs`, and low-level runtime state remain owned by `src/comm`.

The current copied kernel layer was trimmed to AllGather and AllToAll. The lcal reference also contains standalone implementations for AllReduce, ReduceScatter, and Broadcast. This design extends TileXR with those three standard standalone collectives only.

## Goals

- Add public APIs for `TileXRAllReduce`, `TileXRReduceScatter`, and `TileXRBroadcast`.
- Keep the new APIs in `libtilexr-collectives.so` and `tilexr_collectives.h`, not in the core `tilexr_api.h`.
- Reuse the existing host launch flow: validation, `PrepareHostLaunchContext`, blockDim selection, magic allocation, kernel registration, and `rtKernelLaunchWithFlagV2`.
- Import only the lcal kernel families required by the three new standalone collectives.
- Preserve existing `TileXRAllGather` and `TileXRAllToAll` behavior.
- Add unit, integration, and perf-smoke coverage for the new APIs.
- Finish verification on the Ascend server reachable through `ssh blue`.

## Non-Goals

- Do not add fused COC or MC2-style operators.
- Do not add variable-count collectives such as AllToAllV.
- Do not move collective APIs into `libtile-comm.so`.
- Do not introduce hcomm, HCCL, shmem, or ops-transformer runtime dependencies into TileXR-owned source.
- Do not implement quantized AllReduce variants in the first pass.

## Selected Approach

Add the standard standalone collectives from lcal: AllReduce, ReduceScatter, and Broadcast. Keep the public shape close to lcal and HCCL expectations, but use TileXR-owned datatypes and communicator handles.

The first implementation supports SUM only for reduction collectives. The public enum can include future operations, but host validation rejects every reduction op except `TILEXR_REDUCE_SUM` until additional operations are proven by code inspection and hardware tests.

## Public API

Add these declarations to `src/include/tilexr_collectives.h`:

```cpp
int TileXRAllReduce(void *sendBuf, void *recvBuf, int64_t count,
                    TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                    TileXRCommPtr comm, aclrtStream stream);

int TileXRReduceScatter(void *sendBuf, void *recvBuf, int64_t recvCount,
                        TileXR::TileXRDataType dataType, TileXR::TileXRReduceOp op,
                        TileXRCommPtr comm, aclrtStream stream);

int TileXRBroadcast(void *buf, int64_t count,
                    TileXR::TileXRDataType dataType, int root,
                    TileXRCommPtr comm, aclrtStream stream);
```

Add `TileXRReduceOp` to `src/include/tilexr_types.h`.

Initial values:

- `TILEXR_REDUCE_SUM`
- `TILEXR_REDUCE_MAX`
- `TILEXR_REDUCE_MIN`
- `TILEXR_REDUCE_PROD`
- `TILEXR_REDUCE_RESERVED`

Initial support is deliberately narrower than lcal's public signature:

- `TILEXR_REDUCE_SUM` is supported.
- `TILEXR_REDUCE_MAX`, `TILEXR_REDUCE_MIN`, `TILEXR_REDUCE_PROD`, and `TILEXR_REDUCE_RESERVED` are rejected.
- Future work can enable MAX or MIN only after code inspection and hardware correctness tests prove support.

Count semantics:

- `TileXRAllReduce`: `count` elements are read from `sendBuf` and written to `recvBuf`.
- `TileXRReduceScatter`: `recvCount` elements are written to `recvBuf`; `sendBuf` contains `recvCount * rankSize` input elements.
- `TileXRBroadcast`: `buf` is in-place. The `root` rank provides the source data, and every rank observes the root's data after completion.

## Host Dispatch

Extend `src/collectives/host/tilexr_collectives.cpp` with wrappers for the three APIs.

Common behavior:

- Validate non-null communicator and stream-compatible buffers.
- Validate `count` or `recvCount` is positive.
- Validate dtype through the existing supported dtype helper.
- Validate byte-size conversions and multiplication overflow.
- Call `PrepareHostLaunchContext`.
- Use `TileXRCommNextMagic` through the existing kernel launch path.
- Return TileXR error codes consistently with existing AllGather and AllToAll behavior.

Operation-specific behavior:

- `TileXRAllReduce`
  - Requires non-null `sendBuf` and `recvBuf`.
  - Single-rank case copies `sendBuf` to `recvBuf`.
  - Multi-rank case launches `TileXRType::ALL_REDUCE`.

- `TileXRReduceScatter`
  - Requires non-null `sendBuf` and `recvBuf`.
  - Checks `recvCount * rankSize` overflow for input sizing.
  - Single-rank case copies `sendBuf` to `recvBuf` for `recvCount` elements.
  - Multi-rank case launches `TileXRType::REDUCE_SCATTER`.

- `TileXRBroadcast`
  - Requires non-null `buf`.
  - Checks `0 <= root < rankSize`.
  - Single-rank case returns success.
  - Multi-rank case rejects rank sizes above lcal's broadcast support limit of 8.
  - Multi-rank case launches `TileXRType::BROADCAST` with `input == output == buf`.

## BlockDim Selection

Extend `src/collectives/host/collective_utils.{h,cpp}` with:

- `GetAllReduceBlockNum`
- `GetReduceScatterBlockNum`
- `GetBroadcastBlockNum`

Rules should mirror the lcal reference logic used by `Lccl::GetBlockNum`:

- AllReduce:
  - PCIE topology uses `rankSize * 2`.
  - 910B2C large rank path follows lcal's small-data and big-data split.
  - Deterministic and 910_93 rules follow the lcal blockDim selection.
  - Default small-data and big-data rules follow lcal.

- ReduceScatter:
  - PCIE topology uses the big-data write path.
  - 910_93 double-ring and four-step conditions follow lcal.
  - Default small-data and big-data rules follow lcal.

- Broadcast:
  - Supported multi-rank blockDim is `rankSize`.
  - Invalid rank sizes return `0` so the host wrapper fails before launch.

The utility functions should return `0` for invalid or unsupported launch configurations.

## Kernel Integration

Import only the missing lcal kernel families needed by the new standalone collectives into `src/collectives/kernels`.

AllReduce imports:

- `allreduce_one_shot.h`
- `allreduce_two_shot.h`
- `allreduce_big_data.h`
- `91093/allreduce_big_data_sio.h`
- `91093/allreduce_hierarchy_double_ring.h`
- `kernels/lcal_allreduce_2npu_read.cce`
- `kernels/lcal_allreduce_2npu_write.cce`
- `kernels/lcal_allreduce_2npu_big_write.cce`
- `kernels/lcal_allreduce_two_shot.cce`
- `kernels/lcal_allreduce_big_data.cce`
- `kernels/lcal_allreduce_two_shot_910B2C.cce`
- `kernels/lcal_allreduce_big_data_910B2C.cce`
- `kernels/lcal_allreduce_deterministic.cce`
- `kernels/lcal_allreduce_deterministic_big_data.cce`

ReduceScatter imports:

- `reduce_scatter.h`
- `91093/reduce_scatter_big_data_91093_4step.h`
- `91093/reduce_scatter_hierarchy_double_ring.h`
- `kernels/lcal_reduce_scatter.cce`
- `kernels/lcal_reduce_scatter_big_data.cce`
- `kernels/lcal_reduce_scatter_write.cce`
- `kernels/lcal_reduce_scatter_big_data_write.cce`

Broadcast imports:

- `kernels/lcal_broadcast_write.cce`
- `kernels/lcal_broadcast_big_data.cce`

Adapt the imported code consistently with the existing copied lcal files:

- `Lcal` namespace and symbol prefixes become `TileXR`.
- `LCAL` constants and guards become `TILEXR` where applicable.
- Shared buffer constants, `CommArgs`, and `ExtraFlag` references use TileXR-owned headers.
- The imported files must not include from `reference/ascend-transformer-boost`.

Extend `src/collectives/kernels/lccl_op.h` and `src/collectives/kernels/tilexr_lccl_op.cpp` to generate:

- `TileXRAllReduce_<type>`
- `TileXRReduceScatter_<type>`
- `TileXRBroadcast`

Keep the CCE build target as the owner of the embedded collectives binary.

## Kernel Registration and Launch Arguments

Extend `src/collectives/host/collective_kernel.cpp`:

- Register `TileXRType::ALL_REDUCE`, `TileXRType::REDUCE_SCATTER`, and `TileXRType::BROADCAST` in addition to existing AllGather and AllToAll.
- Register typed symbols for AllReduce and ReduceScatter using the existing dtype list.
- Register Broadcast using the symbol naming generated by the adapted lcal macro.
- Generalize launch validation so the accepted collective types are the five standalone collectives.
- Pass reduce op and broadcast root into `AscendCCLKernelArgs`.

`AscendCCLKernelArgs` already has `op` and `root` fields, so no public ABI change is needed.

## Tests

Add or extend unit tests:

- Header compile test checks `TileXRAllReduce`, `TileXRReduceScatter`, `TileXRBroadcast`, and `TileXRReduceOp`.
- API ownership test confirms the new symbols are in `tilexr_collectives.h`, not `tilexr_api.h`.
- Stub and uninitialized-comm tests cover null buffers, invalid counts, invalid dtypes, invalid roots, unsupported reduce ops, and uninitialized communicator behavior.
- Host utility tests cover AllReduce, ReduceScatter, and Broadcast blockDim rules.
- Kernel ownership tests verify imported files live under `src/collectives/kernels` and do not include `reference/ascend-transformer-boost`.

Extend integration tests:

- `tests/collectives/integration/test_tilexr_collectives_correctness.cpp` accepts `--op allreduce|reducescatter|broadcast|allgather|alltoall|both`.
- Correctness initially validates:
  - INT32 SUM AllReduce
  - INT32 SUM ReduceScatter
  - INT32 Broadcast for multiple root choices
  - existing AllGather
  - existing AllToAll

Extend perf tooling:

- `tests/collectives/tilexr-tests/tilexr_collective_perf.cpp` accepts the new ops.
- Message-size reporting uses:
  - AllReduce: `count * dtype_size`
  - ReduceScatter: `recvCount * rankSize * dtype_size`
  - Broadcast: `count * dtype_size`
- Add perf smoke coverage for the three new ops.

## Verification

Local verification:

```bash
source scripts/common_env.sh
cmake -S . -B build -DTILEXR_BUILD_COLLECTIVES=ON -DCMAKE_INSTALL_PREFIX=install
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Additional local checks:

- Confirm `libtilexr-collectives.so` exports all five standalone APIs.
- Confirm kernel registration includes all five standalone collective types.
- Confirm existing AllGather and AllToAll tests still pass.

Required remote hardware verification:

```bash
ssh blue
cd /home/TileXR
source scripts/common_env.sh
```

On `blue`, verify CANN and NPU visibility before running tests. Then build or deploy the branch and run correctness for:

- `allreduce`
- `reducescatter`
- `broadcast`
- `allgather`
- `alltoall`

Run at least one perf smoke test for each new op on `blue`.

If `ssh blue` is unreachable or the NPU environment is unavailable, the work is not fully verified. The final implementation report must state that blocker explicitly.

## Risks and Constraints

- lcal's public API accepts multiple reduce ops, but comments in related lcal COC code indicate only SUM may be supported in some paths. The first TileXR implementation therefore rejects MAX, MIN, and PROD.
- Broadcast support is limited to rank size 8 in the lcal reference.
- Importing kernel files may expose hidden dependencies on lcal names or constants. Unit tests should fail on reference-tree includes and stale `Lcal` public symbols.
- CCE object size may increase. If the padded binary size is insufficient, `TILEXR_COLLECTIVES_1OP_BIN_SIZE` must be adjusted deliberately.
- All remote hardware results depend on `ssh blue` having a matching workspace, CANN environment, and available NPU devices.
