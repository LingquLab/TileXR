# Reference Code

This directory contains helper scripts for reference-only source trees.

Downloaded code under `reference/<repo>/` is intentionally ignored by git. It is kept for historical comparison, experiments, local documentation lookup, and upstream review.

Use:

```bash
bash reference/download_cann_repos.sh
```

`reference/download_cann_repos.sh` downloads the CANN-related open-source
repositories that are useful when working on TileXR:

- `asc-devkit`
- `runtime`
- `driver`
- `asc-tools`
- `oam-tools`
- `ascend-transformer-boost`
- `ops-transformer`
- `shmem`

By default it downloads all repositories into `reference/<repo>/` from
`https://gitcode.com/cann/<repo>.git` on the `master` branch.

```bash
bash reference/download_cann_repos.sh
```

To inspect what would be downloaded without cloning:

```bash
bash reference/download_cann_repos.sh --dry-run
```

To download or update only selected repositories:

```bash
bash reference/download_cann_repos.sh asc-devkit runtime ops-transformer shmem
```

Branch and mirror settings can be overridden with environment variables:

```bash
CANN_REPO_BRANCH=master bash reference/download_cann_repos.sh
CANN_RUNTIME_BRANCH=release-test bash reference/download_cann_repos.sh runtime
CANN_GITCODE_BASE_URL=https://gitcode.com/cann bash reference/download_cann_repos.sh
SHMEM_BRANCH=release-test bash reference/download_cann_repos.sh shmem
```

The `asc-devkit` checkout is the expected local source for Ascend C API
documentation and examples used by the `.claude/skills/ascendc-docs-search`
workflow.

## Repository Guide

| Repository | What It Contains | Read It When |
| --- | --- | --- |
| `asc-devkit` | Ascend C language, API headers, implementations, CMake support, docs, and examples. Important areas include `docs/`, `examples/`, `include/`, and `impl/`. | You need Ascend C API behavior, API variants, kernel examples, SIMT/SIMD migration notes, tiling references, or the source behind `.claude/skills/ascendc-docs-search`. |
| `runtime` | CANN runtime and DFX components: ACL/aclrt public headers, runtime core, memory/stream/event/task management, profiling, dump, logging, examples, and runtime coding guides. | You are debugging `aclInit`, `aclrtMalloc`, stream/event/memory APIs, runtime include paths, DFX behavior, or TileXR host-side interaction with CANN runtime. |
| `driver` | CANN driver stack source and docs, including DCMI, HAL, SDK-driver, RoCE, SVM, NDA/RDMA extension docs, device management examples, and driver/HAL packaging structure. | You need to understand driver/HAL boundaries, `libascend_hal` behavior, device management APIs, RoCE/RDMA/NDA support, SVM, or low-level device/driver diagnostics. |
| `asc-tools` | Ascend C companion tools: CPU debug, NPU check, `msobjdump`, `show_kernel_debug_data`, docs, examples, and tool implementation sources. | You are validating or debugging Ascend C kernels without immediately running full hardware flows, inspecting compiled kernel objects, parsing kernel debug data, or checking kernel implementation issues. |
| `oam-tools` | Operations, administration, and maintenance tools such as `asys`, `msaicerr`, `msprof`, and `hccl_test`, plus docs and examples for environment collection, AI Core error analysis, and profiling. | You are diagnosing hardware/software environment issues, collecting failure bundles, analyzing AI Core errors, profiling workloads, or comparing TileXR troubleshooting scripts with upstream CANN tools. |
| `ascend-transformer-boost` | ATB acceleration library for Transformer workloads, including fused ops, graph/plugin mechanisms, PyTorch integration, examples, and LCAL collective kernels under `src/kernels/lcal`. | You are working on TileXR collective kernels, EP/Transformer acceleration ideas, LCAL-derived code, AllGather/AllReduce/ReduceScatter/All2All references, or ATB integration behavior. |
| `ops-transformer` | Upstream fused communication and computation operators, host tiling code, kernels, examples, and tests. | You need historical MC2 implementation details or want to compare TileXR-owned collectives and EP code with upstream operator implementations. |
| `shmem` | Ascend SHMEM communication library with host/device APIs, symmetric memory heap, team/sync/RMA/Signal interfaces, RDMA/SDMA/UDMA/MTE/xDMA transports, docs, examples, and tests. | You are comparing TileXR registered-memory UDMA/SDMA work with upstream SHMEM, studying RMA and signal semantics, checking AICore direct-drive communication examples, or revisiting historical TileXR UDMA experiments. |
