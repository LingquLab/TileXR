# MoE EP-card Compare Example

This example keeps a small op-simulator harness for comparing the
`ProcessMoeExpertsLoop() -> ProcessMoeExpert()` path of MoE combine kernels.
It models EP as card count while the total number of experts is fixed at 128:

- `EP=32`: 4 experts per rank.
- `EP=64`: 2 experts per rank.

The default kernel is loop-only: dispatch flag polling is skipped so that the
trace focuses on the local expert processing loop.

## Build

On a CANN 9.1.0 machine:

```bash
export ASCEND_HOME_PATH=/path/to/cann
cmake -S op-simulator/examples/moe_epcard_compare \
  -B /tmp/tilexr_moe_epcard_compare_build
cmake --build /tmp/tilexr_moe_epcard_compare_build -j
```

The linked AI Core objects are emitted under:

```text
op-simulator/examples/moe_epcard_compare/op/<kernel>/my_kernel.o
```

Build the host runner:

```bash
RUNNER_BIN=/tmp/tilexr_moe_epcard_compare_runner \
  bash op-simulator/examples/moe_epcard_compare/build_runner.sh
```

## Run

Copy or stage the example directory to the simulator machine, then run the
matrix. The scripts use environment variables so the same harness can be reused
outside the original test server.

```bash
export ASCEND_HOME_PATH=/path/to/cann
export BASE_DIR=/path/to/moe_epcard_compare
export RUNNER_BIN=/tmp/tilexr_moe_epcard_compare_runner
export RESULTS_DIR=/tmp/tilexr_moe_epcard_compare_results

bash "$BASE_DIR/run_matrix.sh"
```

Useful knobs:

```bash
EP_LIST="32 64" BS_LIST="32 128 256" bash "$BASE_DIR/run_matrix.sh"
CASE_TIMEOUT=1800s SOC_VERSION=Ascend950 bash "$BASE_DIR/run_matrix.sh"
SIMULATOR_ARCH=dav_3510 bash "$BASE_DIR/run_matrix.sh"
CASE_LIST=$'32 32 baseline\n32 32 final' bash "$BASE_DIR/run_priority_matrix.sh"
```

The CANN 9.1.0 defaults use headers from `aarch64-linux/pkg_inc`, the AI Core
linker at `aarch64-linux/bin/ld.lld`, and simulator libraries under
`aarch64-linux/simulator/dav_3510`.

Generate comparison JSON and Markdown:

```bash
python op-simulator/scripts/summarize_matrix.py "$RESULTS_DIR"
```

The summary script also parses `report/trace_core0.json` and reports barrier
counts, span ticks, top pipeline category durations, and top category gaps.
