# Full-Mesh Isolated Task Profiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and run wait-free per-task microbenchmarks for the physical full-mesh kernel.

**Architecture:** Host selects one isolated task and passes it as a dedicated kernel argument. The kernel dispatches to wait-free task helpers before the normal pipeline loop and records existing trace phases.

**Tech Stack:** C++14, Ascend C/Bisheng, Python trace conversion, physical Ascend950 2x8.

## Global Constraints

- Preserve normal mode when task is zero.
- Use `/home/pkg/b101/cann` for build and runtime.
- Keep the 16:0 split.
- Never execute a wait primitive in isolated mode.
- Commit and deploy through a verified Git bundle.

---

### Task 1: Host And Kernel Dispatch

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo.cpp`
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp`

**Interfaces:**
- Consumes: `TILEXR_DEMO_BIGDATA_ISOLATED_TASK` in range 0..11.
- Produces: kernel argument `uint32_t isolatedTask`.

- [ ] Add failing source checks for the environment, argument, range guard,
  validation skip, and isolated dispatch.
- [ ] Verify the layout test fails on both hosts.
- [ ] Implement Host parsing and kernel argument plumbing.
- [ ] Verify source tests pass.

### Task 2: Wait-Free Task Helpers

**Files:**
- Modify: `tests/udma/demo/tilexr_udma_demo_kernel.cpp`
- Modify: `tests/udma/unit/test_tilexr_udma_alltoall_layout.cpp`

**Interfaces:**
- Consumes: existing layout, copy, UDMA, trace, and control-slot helpers.
- Produces: `BigDataRunIsolatedTask` for task IDs 1..11.

- [ ] Add failing checks for all task IDs and absence of wait calls in the
  isolated helper slice.
- [ ] Implement each task with existing trace phase names.
- [ ] Build the Host and Bisheng kernel with b101 and run all unit tests.
- [ ] Commit, create a complete bundle, and deploy to both hosts.

### Task 3: Physical Task Matrix

**Files:**
- Create: `tmp/fullmesh_isolated_b101_2x8/` artifacts.

**Interfaces:**
- Consumes: task IDs 1..11 and full-mesh raw tracing.
- Produces: per-task repeat50 timing and iteration49 summaries.

- [ ] Run each task on physical 2x8 with 128 MiB/rank and repeat50.
- [ ] Verify 16 successful ranks and sixteen 8 MiB traces for every task.
- [ ] Verify no wait phase appears in any isolated trace.
- [ ] Summarize per-operation iteration49 mean, minimum, and maximum.
