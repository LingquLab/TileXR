# TileXR MoonEP Torch and Peer-Memory Planner Implementation Plan

## Goal

Integrate the temporary Ascend A5 direct-launch Planner, expose a stable
versioned C ABI and Torch ctypes facade for the five MoonEP stages, and provide
a deterministic benchmark that executes the complete forward and backward
protocol without HCCL. Planning is native; Dispatch, PrefetchWeight, Combine,
and ReduceGrad are explicitly reported native stubs until their real kernels
replace the implementations.

The approved architecture and acceptance boundary are in
`docs/specs/2026-08-02-moonep-torch-peer-memory-design.md`.

## Fixed Contracts

- Host code remains C++14 and targets CANN 9.1.0.
- Planner uses direct kernel launch and A5 peer-window GM semantics only.
- Cross-rank Planner metadata never calls, registers, or falls back to UDMA.
- Dispatched capacity is exactly `S * K`.
- `cu_seqlens` contains exact cumulative route counts. Empty groups repeat
  the previous end.
- There is no logical route alignment parameter or padding cleanup output.
  Internal DMA/UB/workspace alignment remains an implementation detail.
- Backward reuses the forward Plan and does not rerun Planning or
  PrefetchWeight.
- Stub success is not transport-performance evidence.

## Task 1: Integrate and Simplify the Planner

**Objective and role:** Bring the historical A5 Planner into the active tree,
remove its single-node restriction, remove logical route padding, and make peer
wait failure bounded and observable.

**Modification scope:**

- `src/include/tilexr_moonep_planner.h`
- `src/moonep/planner/**`
- root `CMakeLists.txt`
- `tests/moonep_planner/**`

**Implementation requirements:**

- Remove the route-alignment argument and cleanup-range pointer from every
  Host/Kernel/CPU-oracle call.
- Compute `routeCount = dispatchedCapacity = S * K` with checked arithmetic.
- In `BuildExpertLayout`, advance by the real count and write the real end to
  `cu_seqlens`.
- Delete the Host cross-node rejection while retaining A5, locality, peer
  pointer, workspace, and capacity validation.
- Publish TPE at `peerMems[rank] + IPC_DATA_OFFSET`; use the existing magic
  flag area and direct MTE reads for every peer.
- Replace the infinite ready wait in this Planner with a bounded poll that
  stores a deterministic status. All AIV blocks stop later phases after a
  timeout.
- Keep active targets independent of `reference/` and `src/ep`.

**Acceptance and verification:**

- Host layout/oracle tests cover balanced, biased, local, remote, duplicate,
  empty-group, overflow, and exact-`S*K` cases.
- Source guards prove no UDMA calls and no single-node rejection are present.
- The target toolchain compiles Host and kernel artifacts on the A5 server.

## Task 2: Add the Stable MoonEP Native ABI and Stubs

**Objective and role:** Give future real stage implementations a stable,
Torch-independent replacement boundary.

**Modification scope:**

- `src/include/tilexr_moonep.h`
- `src/moonep/CMakeLists.txt`
- `src/moonep/host/**`
- `tests/moonep/**`

**Implementation requirements:**

- Build/install `libtilexr-moonep.so.1` behind `TILEXR_BUILD_MOONEP`.
- Use C-compatible V1 structures with `structSize`, `abiVersion`, fixed-width
  fields, raw device pointers, checked byte counts, `TileXRCommPtr`, and
  caller-owned stream.
- Export capability/version queries and V1 entry points for Planning,
  Dispatch, PrefetchWeight, Combine, and ReduceGrad.
- Planning delegates to the real Planner. The other stages enqueue only
  bounded local memset/copy work and advertise their stub capabilities.
- Calls are asynchronous and never synchronize the caller stream internally.
- Install RPATH is `$ORIGIN`; runtime paths never contain CANN `devlib`.

**Acceptance and verification:**

- C header compile, ABI layout, invalid-input, overflow, capability, and stub
  enqueue tests pass.
- Install-prefix checks resolve the Planner, kernel, communication, and MoonEP
  libraries from the install tree.

## Task 3: Add the Torch ctypes Facade

**Objective and role:** Make the native contract usable from Torch-NPU while
preserving current device, stream, tensor, communicator, and Plan lifetimes.

**Modification scope:**

- `integrations/moonep_torch/tilexr_moonep/**`
- `tests/moonep/python/**`

**Implementation requirements:**

- Load `libtile-comm.so` globally, then Planner and MoonEP libraries.
- Initialize TileXR directly from rank/world/bootstrap inputs; do not import or
  initialize `torch.distributed`.
- Validate NPU placement, dtype, contiguity, shape, storage size, and current
  stream before passing `data_ptr()`.
- `MoonEPPlan` owns `dst`, `cu_seqlens`, `experts_to_copy`,
  `remote_stats`, status, workspace lifetime boundary, topology metadata, and
  a runtime reference. Its `dispatched_capacity` property returns `S * K`.
- Provide explicit forward and backward orchestration. Do not publish a
  `torch.autograd.Function` while four stages are stubs.

**Acceptance and verification:**

- Fake-runtime tests verify exact native call order, stream forwarding, Plan
  reuse, cleanup, error propagation, and absence of HCCL dependencies.

## Task 4: Add the Complete Benchmark and Launcher

**Objective and role:** Exercise the intended training protocol and retain
honest, comparable artifacts while some stages are stubs.

**Modification scope:**

- `tools/moonep/**`
- benchmark source tests under `tests/moonep/python/**`

**Implementation requirements:**

- JSON and CLI cases cover `S/K/E/H`, dtype, seed, warmup, measured rounds,
  correctness, stage timing, and artifact directory.
- Forward is Planning -> Dispatch -> PrefetchWeight -> expert forward ->
  Combine.
- Backward is Dispatch(saved Plan) -> expert backward -> Combine(saved Plan) ->
  ReduceGrad.
- Reports contain per-rank samples, cross-rank maxima, percentiles, native/stub
  capabilities, physical/logical device counts, and validity flags.
- Each rank synchronizes its local NPU stream and joins a host TCP completion
  rendezvous before communicator destruction. HMAC-authenticated release
  requires arrivals from every rank with local quiescence proven; failures are
  propagated to every rank, while an unquiesced worker holds its communicator
  for launcher-coordinated termination even if artifact writes fail.
- The launcher maps `logical_rank % physical_device_count` and supports
  `--ranks-per-device 2`; a 16-rank run on 8 devices is marked
  oversubscribed and is not reported as 16-device performance.

**Acceptance and verification:**

- Parser/report/launcher unit tests and a deterministic fake-provider
  forward/backward smoke pass without NPU hardware.

## Task 5: Target Build and Remote Validation

**Objective and role:** Establish target-toolchain and bounded A5 runtime
evidence without modifying the existing deployment.

**Remote scope:**

- Create a new directory under `/home/d00520898` on
  `root@141.61.49.192`.
- Do not overwrite `/home/d00520898/TileXR-MoonEP-Planner`.

**Validation order:**

1. Configure/build/install communication, Planner, MoonEP, and focused tests.
2. Run Host layout/oracle/ABI/source tests.
3. Inspect SONAME, dependencies, RPATH/RUNPATH, and absence of HCCL linkage.
4. Run bounded Planner correctness on 1, 8, and 16 logical ranks. The 16-rank
   case maps two processes to each of the 8 physical devices.
5. Run Torch forward/backward only if a compatible Torch/Torch-NPU environment
   already exists. Installing packages is outside this plan without separate
   authority.

**Evidence boundary:**

- The 16-rank oversubscribed case proves logical-rank call paths and indexing,
  not 16-card throughput.
- A single host cannot validate cross-node peer-memory ordering.
- Driver `25.1.rc1.b188` evidence is development-only because it is below the
  repository's supported 25.5.0 baseline.

## Final Verification

- Run focused C++ and Python tests from the final source state.
- Run `git diff --check`, inspect all new public symbols and final diff, and
  search active code for forbidden HCCL/UDMA Planner dependencies and removed
  logical-padding fields.
- Report host/static, A5 compile, A5 runtime, Torch-NPU, and cross-node evidence
  separately; unavailable layers remain explicit residual risks.
