## Context

The host side now creates two comm objects per MC2 op on single-server 910B:
- `commLevel0` (SDMA P2P) — fills `windowsIn[]` / `windowsOut[]` in `HcclA2CombineOpParam`
- `commLevel0Rdma` (ibverbs) — fills the `data` pointer via `SetDevIbverbsData()`

`HcclA2CombineOpParam` is passed to the kernel as `contextGM` and cast to `winContext_` inside `AllGatherMatmulFullMesh::Init()`. The existing code already prints `tileXrContext` under `GetBlockIdx() == 0`. We extend this block to also dump P2P and ibverbs fields.

## Goals / Non-Goals

**Goals:**
- Print `rankId`, `rankNum`, `winSize` from `winContext_` to confirm basic context integrity
- Print `windowsIn[i]` and `windowsOut[i]` for all `rankNum` ranks to verify P2P window mapping
- Print `data` and `dataSize` to verify ibverbs path was populated by `SetCommResource`
- All output uses `[cwh]` prefix, visible via `plog_grep.sh cwh`

**Non-Goals:**
- Dereferencing `data` to inspect ibverbs QP internals
- Printing `aiRMAInfo` contents
- Any functional change to comm behavior
- Permanent production logging (these are temporary debug prints)

## Decisions

### Print location: `Init()` inside `GetBlockIdx() == 0` guard
`Init()` is called once per kernel launch. The `GetBlockIdx() == 0` guard ensures exactly one AICore prints, avoiding log flooding. `winContext_` is already set at this point (`winContext_ = (__gm__ HcclA2CombineOpParam *)contextGM`), so no extra setup is needed.

Alternative considered: printing in `Process()` → rejected, `winContext_` would need to be re-read and the timing is after comm is already underway.

### Loop `windowsIn/Out` up to `rankNum`
`HCCL_MAX_RANK_NUM = 32` but the test environment has 2 ranks. Looping up to `winContext_->rankNum` avoids printing 30 zero entries. Kernel-side loops are fine in debug mode.

### Print `data` as pointer only (no dereference)
`data` is `__gm__ AscendC::IbVerbsData*`. Dereferencing it in the kernel to inspect QP fields is risky and unnecessary — a non-null, non-zero value is sufficient to confirm ibverbs fill. `dataSize` provides additional confirmation of the payload size.

## Risks / Trade-offs

- [Risk] Kernel printf overhead slows first iteration slightly → Mitigation: guarded by `GetBlockIdx() == 0`, runs once, acceptable for debug builds
- [Risk] `winContext_->rankNum` could be 0 if context not yet initialized → Mitigation: the existing `tileXrContext` null-check pattern shows context is valid before this block executes; `rankNum` is set during comm creation so will be non-zero

## Migration Plan

Single-file change to `all_gather_matmul_full_mesh.h`. No migration needed. To remove: delete the added `printf` lines.
