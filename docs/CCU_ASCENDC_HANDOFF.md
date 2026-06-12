# AscendC CCU Context Handoff

This note is a context-free handoff for continuing the CCU direct-communication
investigation on a machine with Ascend hardware and CANN 9.1.0. It summarizes
what was found from static CANN/package inspection, where the matching source
trees are, and what should be tested next.

## Goal

Determine whether TileXR can use the AscendC HCCL CCU server path from AIV
kernels, so that kernels can communicate through CCU without TileXR manually
mapping raw CCU registers.

The target kernel-side shape is the AscendC API path:

```cpp
AscendC::Hccl<AscendC::HCCL_SERVER_TYPE_CCU> hccl;
hccl.Init(contextGM, tilingVersion);
```

The open question is how `contextGM` is produced for the CCU server path, and
whether TileXR can obtain or carry that context safely.

## Environment Assumptions

- CANN version: 9.1.0.
- In the local analysis environment, CANN was already installed at:
  `/home/TileXR/env/cann/cann-9.1.0`.
- `scripts/common_env.sh` should be sourced before building or probing:

```bash
source scripts/common_env.sh
```

If CANN 9.1.0 is absent on the test machine, install it through the repository
scripts instead of assuming `/usr/local/Ascend`:

```bash
bash scripts/cann_download_install.sh
```

Important runtime guardrail: CANN `devlib` is for link-time fallback only. Do
not put `${ASCEND_HOME_PATH}/${ARCH}-linux/devlib` into runtime RPATH/RUNPATH.
Runtime must resolve the real driver HAL, normally from
`/usr/local/Ascend/driver/lib64/driver`.

## Static Scan Summary

The static scan covered every `.so` under the local CANN 9.1.0 tree:

```bash
mkdir -p /tmp/tilexr_cann_so_scan
find "$ASCEND_HOME_PATH" -type f -name '*.so*' \
  > /tmp/tilexr_cann_so_scan/so_files.txt
wc -l /tmp/tilexr_cann_so_scan/so_files.txt
```

The local scan found 887 shared objects.

Useful string/symbol scan anchors:

```bash
PATTERN='HcclCombineOpParam|HcclCombinOpParam|xnOffset|ckeOffset|xnAddr|ckeAddr|CCUMsg|CcuSendMsg|HCCL_SERVER_TYPE_CCU|HcclAllocComResourceByTiling|HcclEngineCtx(Create|Get)|AllocOpResCtx|OpResCtx|Mc2InitTiling|Mc2CcTiling|CcuTaskParam'

while read -r so; do
  hits=$(strings -a "$so" | rg "$PATTERN" || true)
  if [ -n "$hits" ]; then
    printf '### %s\n%s\n' "$so" "$hits"
  fi
done < /tmp/tilexr_cann_so_scan/so_files.txt \
  > /tmp/tilexr_cann_so_scan/strings_hits.txt

while read -r so; do
  hits=$(nm -D --defined-only "$so" 2>/dev/null | c++filt | rg "$PATTERN" || true)
  if [ -n "$hits" ]; then
    printf '### %s\n%s\n' "$so" "$hits"
  fi
done < /tmp/tilexr_cann_so_scan/so_files.txt \
  > /tmp/tilexr_cann_so_scan/nm_defined_hits.txt
```

The exact new struct name `HcclCombineOpParam` did not appear as an exported
ABI symbol. That is expected because it is a device-side/header-visible POD used
by inline AscendC code. The shared objects expose the resource-allocation and
context-manager functions instead.

## Three Similar Context Types

Keep these three structures separate.

### New AscendC CCU Context: `HcclCombineOpParam`

Defined in CANN AscendC headers:

- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/impl/adv_api/detail/hccl/common/hccl_inner_def.h`
- Mirror in `reference/asc-devkit/impl/adv_api/detail/hccl/common/hccl_inner_def.h`

Relevant fields for 3510/950-class builds:

```cpp
struct HcclCombineOpParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;
    uint32_t rankNum;
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MAX_RANK_NUM_V310];
    uint64_t windowsOut[HCCL_MAX_RANK_NUM_V310];
    GM_ADDR xnOffset;
    GM_ADDR ckeOffset;
    ...
};
```

The CCU implementation casts the kernel context pointer to this type:

- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/impl/adv_api/detail/hccl/impl/platform_v310/hccl_ccu_v0.h`
- Mirror in `reference/asc-devkit/impl/adv_api/detail/hccl/impl/platform_v310/hccl_ccu_v0.h`

Important behavior:

- `InitInner()` casts `context` to `__gm__ HcclCombineOpParam *`.
- It reads `workSpace`, `rankId`, `rankNum`, `xnOffset`, and `ckeOffset`.
- `CcuSendMsg(resourceId)` computes:
  `xnOffset + resourceId * CCU_MSG_XN_NUM * CCU_XN_DATA_SIZE`.
- CKE commit/wait addresses are derived from `ckeOffset`.

This path is the one TileXR would want if it wants AscendC kernels to call the
CCU HCCL client directly.

### AICPU/MC2 Context: `OpResCtx`

The `libmc2_client.so` path found during static analysis allocates an `OpResCtx`,
not obviously `HcclCombineOpParam`.

Headers and source:

- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/include/adv_api/hccl/internal/hccl_msg.h`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/impl/adv_api/detail/hccl/cc/src/common/hccl_alloc_ctx_res.h`
- `reference/asc-devkit/impl/adv_api/detail/hccl/cc/src/common/hccl_alloc_ctx_res.h`

Key functions:

- `HcclAllocComResourceByTiling(...)`
- `HcclAllocOpResCtx(...)`
- `HcclEngineCtxCreate(...)`
- `HcclEngineCtxGet(...)`

The source allocates device memory for `OpParam`, workspace, and `OpResCtx`, then
copies `OpResCtx` into device memory. This is likely the AICPU MC2 path and must
not be assumed to be the CCU `HcclCombineOpParam` context without runtime proof.

### Legacy hcomm Context: `HcclCombinOpParam`

The legacy structure name is missing the `e` in `Combine` and uses `xnAddr` /
`ckeAddr`, not `xnOffset` / `ckeOffset`.

Source:

- `3rdparty/hcomm/src/legacy/common/types/mc2_type.h`
- `3rdparty/hcomm/src/legacy/framework/entrance/op_base/op_base_v2.cc`
- `3rdparty/hcomm/src/legacy/framework/ccu/ccu_mc2/mc2_compont.cpp`
This is the path that shows up clearly in `libhccl_v2.so` strings:

```text
hcclCombinOpParam info: workSpace = ..., xnAddr = ..., ckeAddr = ...
```

This is useful evidence for historical CCU resource allocation, but it is not
the same ABI as the new AscendC `HcclCombineOpParam` with `xnOffset` and
`ckeOffset`.

## Shared Object to Source Mapping

Static inspection points to this mapping:

| Shared object | Evidence | Likely source tree |
| --- | --- | --- |
| `libnnopbase.so` | `DoHcclAllocComResourceByTiling`; calls MC2/HCCL allocation | CANN nnopbase/op executor side |
| `libmc2_client.so` | `HcclAllocComResourceByTiling`, `HcclAllocOpResCtx`, `HcclCreateOpResCtx`; log string says `asc-devkit common HcclAllocComResourceByTiling` | `reference/asc-devkit/impl/adv_api/detail/hccl/cc/src/common` |
| `libhcomm.so` | `HcclEngineCtxCreate`, `HcclEngineCtxGet`, `HcclAllocComResourceByTiling`, `HcclCreateOpResCtxInner` | `3rdparty/hcomm/src/framework/communicator` and related hcomm sources |
| `libhccl_v2.so` | `HcclAllocComResourceByTilingV2`, `HcclCombinOpParam`, `xnAddr`, `ckeAddr` strings | `3rdparty/hcomm/src/legacy` |
| `devlib/device/libccl_kernel.so` | Similar HCCL symbols in device/devlib copy | Do not use as runtime host evidence |
| `opp/built-in/op_impl/aicpu/kernel/libaicpu_custom.so` | AICPU-side `HcclCombinOpParam` hits | AICPU kernel side, not host allocation API |

## Current Interpretation

The supported-looking path is still HCCL/MC2 context-based, not raw TileXR
register mapping.

The strongest static evidence for the CCU AscendC kernel path is:

- `HCCL_SERVER_TYPE_CCU` exists in AscendC HCCL headers.
- The CCU implementation consumes a prebuilt `HcclCombineOpParam` context.
- That context contains `xnOffset` and `ckeOffset`, and kernel code derives CCU
  message/register addresses from those fields.

The missing piece is the producer of that exact `HcclCombineOpParam` context.
The public-looking host allocation path found in `libmc2_client.so` constructs
`OpResCtx`, so runtime validation must prove whether there is a conversion or a
different code path for `NNOPBASE_HCCL_SERVER_TYPE_CCU`.

## Next Tests for a Hardware-Enabled Agent

Run these on an Ascend 950/3510-capable machine with CANN 9.1.0 and working HCCL.

### 1. Confirm Platform and CANN Runtime

```bash
source scripts/common_env.sh
npu-smi info
echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"
echo "ARCH=$ARCH"
find "$ASCEND_HOME_PATH" -maxdepth 3 -name 'libmc2_client.so' -o -name 'libhcomm.so' -o -name 'libhccl_v2.so'
```

Check that runtime library resolution does not pick CANN `devlib` first:

```bash
printenv LD_LIBRARY_PATH | tr ':' '\n' | nl -ba
```

### 2. Reproduce the Static Scan

Re-run the scan commands from "Static Scan Summary" and confirm the same key
libraries appear. Differences may indicate a different CANN patch level or
packaging layout.

### 3. Build a Context Dump Probe

Create a small host probe that initializes an HCCL communicator and calls:

```cpp
HcclAllocComResourceByTiling(comm, stream, mc2Tiling, &context);
```

Then copy the returned device `context` to host and interpret the first bytes in
two ways:

1. As `HcclCombineOpParam`.
2. As `OpResCtx`.

Acceptance criteria for `HcclCombineOpParam` interpretation:

- `rankId < rankNum`.
- `rankNum` matches the communicator rank size.
- `workSpace` is non-zero and suitably aligned.
- `xnOffset` is non-zero and 64-byte aligned.
- `ckeOffset` is non-zero.
- `windowsIn/windowsOut` values are plausible device addresses.

If the buffer clearly matches `OpResCtx` instead, do not pass it to the CCU
AscendC kernel as a `HcclCombineOpParam`.

### 4. Hook the Allocation Path

Use `LD_PRELOAD` or direct `dlsym` wrappers to log:

- `HcclAllocComResourceByTiling`
- `HcclAllocComResourceByTilingV2`
- `HcclEngineCtxCreate`
- `HcclEngineCtxGet`
- `HcclCreateOpResCtxInner`

For each call, record:

- library that resolved the symbol,
- `comm`,
- `stream`,
- `mc2Tiling`,
- returned context pointer,
- context size when available,
- first 256 to 8192 bytes copied back from device memory.

This should identify whether the actual runtime path goes through
`libmc2_client.so`, `libhcomm.so`, `libhccl_v2.so`, or a graph/nnopbase wrapper.

### 5. Trace the CCU Server Selection Path

Search these anchors in the installed CANN tree and repo sources:

```bash
rg -n "NNOPBASE_HCCL_SERVER_TYPE_CCU|HCCL_SERVER_TYPE_CCU|contextAddr|CcuTaskInfo|CreateCcuTask|HcclGetCcuTaskInfo|HcclCcuKernel" \
  "$ASCEND_HOME_PATH" reference 3rdparty examples src
```

The key question is whether `NNOPBASE_HCCL_SERVER_TYPE_CCU` causes nnopbase/GE
to allocate or transform a CCU `HcclCombineOpParam` context before kernel launch.

### 6. Compile a Minimal AscendC CCU Kernel

After a valid `HcclCombineOpParam` context is found, compile a tiny kernel that:

- includes AscendC HCCL headers,
- instantiates `Hccl<HCCL_SERVER_TYPE_CCU>`,
- calls `Init(contextGM, tilingVersion)`,
- performs the smallest possible supported CCU operation or message path,
- writes a status code to output memory.

Keep the first probe minimal. Do not start by integrating it into TileXR
collectives.

### 7. TileXR Integration Sketch if the Probe Works

If the context is valid and the CCU kernel probe succeeds:

- Add a TileXR host-side optional CCU context acquisition path.
- Store the returned context pointer in `CommArgs` as a device pointer, not as
  raw CCU register addresses.
- Gate it behind a capability check and an env var, for example
  `TILEXR_ENABLE_CCU=1`.
- On kernel side, call the AscendC HCCL CCU API only when the context pointer is
  non-null and the architecture supports it.
- Keep existing TileXR IPC/UDMA/SDMA paths as fallback.

Do not manually synthesize `xnOffset` or `ckeOffset` from guessed physical
addresses unless Huawei documentation explicitly guarantees that interface.

## Stop Conditions

Stop the CCU direct path and report "not currently safe to integrate" if any of
the following are true:

- No public or stable host API can produce a valid `HcclCombineOpParam`.
- The only available context is `OpResCtx`.
- `xnOffset` or `ckeOffset` is zero, unaligned, or not readable as a valid device
  address by the CCU kernel.
- The path requires runtime loading from CANN `devlib`.
- The path only works through private graph-engine side effects that TileXR
  cannot reproduce in a standalone runtime.

## Useful Files

Installed CANN paths:

- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/include/adv_api/hccl/hccl.h`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/impl/adv_api/detail/hccl/common/hccl_inner_def.h`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/impl/adv_api/detail/hccl/impl/platform_v310/hccl_ccu_v0.h`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/asc/impl/adv_api/detail/hccl/cc/src/common/hccl_alloc_ctx_res.h`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/libmc2_client.so`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/libhcomm.so`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/libhccl_v2.so`
- `${ASCEND_HOME_PATH}/${ARCH}-linux/lib64/libnnopbase.so`

Repository paths:

- `reference/asc-devkit/impl/adv_api/detail/hccl/common/hccl_inner_def.h`
- `reference/asc-devkit/impl/adv_api/detail/hccl/impl/platform_v310/hccl_ccu_v0.h`
- `reference/asc-devkit/impl/adv_api/detail/hccl/cc/src/common/hccl_mc2.cc`
- `reference/asc-devkit/impl/adv_api/detail/hccl/cc/src/common/hccl_alloc_ctx_res.h`
- `3rdparty/hcomm/src/framework/communicator/impl/independent_op/hccl_independent_op_ctx.cc`
- `3rdparty/hcomm/src/legacy/common/types/mc2_type.h`
- `3rdparty/hcomm/src/legacy/framework/entrance/op_base/op_base_v2.cc`
- `3rdparty/hcomm/src/legacy/framework/ccu/ccu_mc2/mc2_compont.cpp`
