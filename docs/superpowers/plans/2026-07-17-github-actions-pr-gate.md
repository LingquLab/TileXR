# GitHub Actions PR Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and activate a mandatory TileXR pull-request gate that runs fast Ubuntu 20.04 host checks, then queues the exact merge commit for CANN 9.1 build and complete single-host 8x910B3 validation on `blue`.

**Architecture:** A public `pull_request` workflow runs cheap source and host tests on a GitHub-hosted runner, then calls a reusable workflow pinned to `main`. The reusable workflow is the only workflow allowed to use a dedicated organization runner group; it invokes an administrator-owned `v1` controller on `blue`, builds the merge commit, waits for all eight NPUs, executes the hardware manifest, cleans only CI-owned processes, and uploads evidence. Repository rules require the final `PR Gate` check only after a successful trial run proves that the check exists.

**Tech Stack:** GitHub Actions, GitHub rulesets and REST API, Bash, Python 3 standard library, Ruby Psych for YAML validation, CMake/CTest, systemd, `npu-smi`, CANN 9.1, MPICH, Ascend 910B3.

---

## Scope And Execution Boundaries

- Implement repository code and tests before changing GitHub or `blue` state.
- Use a feature branch or isolated worktree when execution starts; do not continue implementation directly on `main`.
- Keep all workflow permissions read-only. Do not add repository or environment secrets.
- Do not use `pull_request_target`.
- Treat external-fork approval as authorization to execute reviewed code on the shared host.
- Do not terminate NPU processes owned by another account.
- Do not enable the required status check until a real `PR Gate` check has succeeded.
- Do not claim A5 / Ascend950 UDMA or multi-host coverage.
- The repository portion and the server/GitHub portion remain one plan because the latter cannot be validated until the former is merged to `main`.

## File Structure

### Repository orchestration

- `.github/CODEOWNERS`: assigns the CI boundary to `@LingquLab/ci-maintainers`.
- `.github/workflows/pr-ci.yml`: pull-request trigger, host job, reusable NPU call, and stable `PR Gate` aggregator.
- `.github/workflows/npu-ci.yml`: selected reusable workflow allowed to target `TileXR-NPU`.
- `scripts/ci/host_checks.sh`: complete Ubuntu-compatible fast test entrypoint.
- `scripts/common_env.sh`: permits an administrator-provided CANN location without changing normal local behavior.

### Sealed controller source

- `scripts/ci/control/VERSION`: contains exactly `v1`.
- `scripts/ci/control/npu_state.py`: parses live `npu-smi` process and health output and implements the six-hour idle waiter.
- `scripts/ci/control/gate.py`: verifies merge identity, enforces process policies and timeouts, acquires the lock, runs phases, cleans up, and classifies failures.
- `scripts/ci/control/build_blue.sh`: clean CANN build and all non-hardware validation.
- `scripts/ci/control/run_hardware.sh`: exact 8-card hardware manifest.
- `scripts/ci/control/collect_artifacts.sh`: gathers bounded logs and reports under one upload directory.
- `scripts/ci/control/job_completed.sh`: removes the finished runner workspace after Actions post-job steps.

### Provisioning

- `scripts/ci/provision/common.sh`: dry-run-aware privileged command helper and shared constants.
- `scripts/ci/provision/account.sh`: idempotent `tilexr-ci` account and directory setup.
- `scripts/ci/provision/cann.sh`: user-local CANN 9.1 Toolkit and 910B Ops installation and sealing.
- `scripts/ci/provision/control.sh`: installs the reviewed `v1` controller read-only.
- `scripts/ci/provision/runner.sh`: installs, configures, and starts the current ARM64 Actions runner.
- `scripts/ci/provision/verify.sh`: read-only server acceptance check.
- `docs/CI.md`: operator runbook, rollout, diagnostics, and rollback.

### Tests

- `tests/ci/CMakeLists.txt`: standalone and top-level CTest wiring.
- `tests/ci/fixtures/npu_smi_busy.txt`: representative `blue` process table.
- `tests/ci/fixtures/npu_smi_idle.txt`: representative process-free output.
- `tests/ci/fixtures/npu_smi_health_ok.txt`: representative healthy-device output.
- `tests/ci/test_common_env_override.sh`: CANN path override behavior.
- `tests/ci/test_npu_state.py`: parser, health, stable-idle, and timeout tests.
- `tests/ci/test_gate.py`: merge verification, policy, termination, and classification tests.
- `tests/ci/test_control_contract.py`: exact build and hardware manifest requirements.
- `tests/ci/test_provision_dry_run.sh`: idempotent provisioning command contract.
- `tests/ci/test_workflows.rb`: structured workflow and CODEOWNERS validation.

## Task 1: Add The CI Test Surface And CANN Override

**Files:**

- Create: `tests/ci/CMakeLists.txt`
- Create: `tests/ci/test_common_env_override.sh`
- Modify: `CMakeLists.txt`
- Modify: `scripts/common_env.sh`

**Interface:** `TILEXR_CANN_HOME` may be set before sourcing `scripts/common_env.sh`. If unset, the current `${TILEXR_HOME}/env/cann` default remains unchanged.

- [ ] **Step 1: Write the failing override test**

Create `tests/ci/test_common_env_override.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

mkdir -p "${TMP_DIR}/cann"
actual="$({
    TILEXR_CANN_HOME="${TMP_DIR}/cann"
    export TILEXR_CANN_HOME
    source "${ROOT_DIR}/scripts/common_env.sh"
    printf '%s\n' "${TILEXR_CANN_HOME}"
} | tail -n 1)"

if [[ "${actual}" != "${TMP_DIR}/cann" ]]; then
    echo "expected TILEXR_CANN_HOME=${TMP_DIR}/cann, got ${actual}" >&2
    exit 1
fi
```

- [ ] **Step 2: Run it and confirm the current overwrite fails**

Run:

```bash
bash tests/ci/test_common_env_override.sh
```

Expected: nonzero exit with the test's expected/actual CANN path mismatch because
`common_env.sh` currently overwrites the caller value.

- [ ] **Step 3: Preserve an explicit toolchain path**

Replace the unconditional CANN assignment in `scripts/common_env.sh` with:

```bash
export TILEXR_CANN_HOME="${TILEXR_CANN_HOME:-${TILEXR_ENV_HOME}/cann}"
```

Do not change `TILEXR_ENV_HOME`, temporary directories, utility paths, or the default local layout.

- [ ] **Step 4: Add standalone CTest wiring**

Create `tests/ci/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(TileXR_CI_Tests NONE)

enable_testing()
find_program(BASH_EXECUTABLE bash)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_program(RUBY_EXECUTABLE ruby)

if(NOT BASH_EXECUTABLE)
    message(FATAL_ERROR "bash is required for TileXR CI tests")
endif()

add_test(
    NAME test_tilexr_ci_common_env_override
    COMMAND ${BASH_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_common_env_override.sh
)
```

Ruby is intentionally optional in this standalone CMake file because workflow
YAML is validated in the required GitHub-hosted stage, where Ruby is installed
explicitly. Do not silently omit any Python or Bash controller test.

Add this next to the existing data-as-flag test wiring in root `CMakeLists.txt`:

```cmake
if(BUILD_TESTING OR TILEXR_BUILD_TESTS)
    add_subdirectory(tests/ci)
endif()
```

- [ ] **Step 5: Verify and commit**

Run:

```bash
bash tests/ci/test_common_env_override.sh
cmake -S tests/ci -B tests/ci/build
ctest --test-dir tests/ci/build --output-on-failure
git diff --check
```

Expected: the shell test passes; CTest reports `100% tests passed`; `git diff --check` prints nothing.

Commit:

```bash
git add CMakeLists.txt scripts/common_env.sh tests/ci
git commit -m "build: allow a fixed CANN toolchain path"
```

## Task 2: Implement And Test NPU State Detection

**Files:**

- Create: `scripts/ci/control/VERSION`
- Create: `scripts/ci/control/npu_state.py`
- Create: `tests/ci/fixtures/npu_smi_busy.txt`
- Create: `tests/ci/fixtures/npu_smi_idle.txt`
- Create: `tests/ci/fixtures/npu_smi_health_ok.txt`
- Create: `tests/ci/test_npu_state.py`
- Modify: `tests/ci/CMakeLists.txt`

**Interfaces:**

```python
parse_process_table(text: str) -> List[NpuProcess]
parse_health(text: str) -> str
wait_for_idle(read_snapshot, sleep, now, emit, max_wait_seconds, poll_seconds, stable_samples) -> Snapshot
```

- [ ] **Step 1: Add realistic fixtures**

Copy a sanitized `npu-smi info` process section into `npu_smi_busy.txt` with two rows:

```text
| NPU     Chip              | Process id    | Process name             | Process memory(MB)      |
| 0       0                 | 3558608       | python3.11                | 6064                    |
| 7       0                 | 3558615       | python3.11                | 6064                    |
```

Create `npu_smi_idle.txt` with the same header and no numeric process rows. Create `npu_smi_health_ok.txt`:

```text
    NPU ID                         : 0
    Chip Count                     : 1
    Health                         : OK
    Chip ID                        : 0
```

- [ ] **Step 2: Write failing parser and waiter tests**

Create `tests/ci/test_npu_state.py` with these cases:

```python
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/ci/control"))
import npu_state as MODULE


class NpuStateTest(unittest.TestCase):
    def fixture(self, name):
        return (ROOT / "tests/ci/fixtures" / name).read_text()

    def test_process_table_extracts_device_and_pid(self):
        rows = MODULE.parse_process_table(self.fixture("npu_smi_busy.txt"))
        self.assertEqual([(row.device, row.pid) for row in rows], [(0, 3558608), (7, 3558615)])

    def test_idle_table_has_no_processes(self):
        self.assertEqual(MODULE.parse_process_table(self.fixture("npu_smi_idle.txt")), [])

    def test_health_parser_uses_first_health_field(self):
        self.assertEqual(MODULE.parse_health(self.fixture("npu_smi_health_ok.txt")), "OK")

    def test_waiter_requires_two_consecutive_idle_snapshots(self):
        snapshots = iter([
            MODULE.Snapshot(False, ()),
            MODULE.Snapshot(True, ()),
            MODULE.Snapshot(True, ()),
        ])
        clock = [0]
        def sleep(seconds):
            clock[0] += seconds
        result = MODULE.wait_for_idle(
            read_snapshot=lambda: next(snapshots),
            sleep=sleep,
            now=lambda: clock[0],
            emit=lambda message: None,
            max_wait_seconds=21600,
            poll_seconds=60,
            stable_samples=2,
        )
        self.assertTrue(result.healthy)

    def test_waiter_raises_resource_timeout(self):
        clock = [0]
        def sleep(seconds):
            clock[0] += seconds
        with self.assertRaises(MODULE.ResourceTimeout):
            MODULE.wait_for_idle(
                read_snapshot=lambda: MODULE.Snapshot(False, ()),
                sleep=sleep,
                now=lambda: clock[0],
                emit=lambda message: None,
                max_wait_seconds=21600,
                poll_seconds=60,
                stable_samples=2,
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Verify the test fails because the module is absent**

Run:

```bash
python3 tests/ci/test_npu_state.py -v
```

Expected: import failure for `scripts/ci/control/npu_state.py`.

- [ ] **Step 4: Implement the minimal parser and waiter**

Create `scripts/ci/control/VERSION` containing exactly:

```text
v1
```

Implement `npu_state.py` with frozen `NpuProcess` and `Snapshot` dataclasses. The process parser must enter process-table mode only after a line containing both `Process id` and `Process name`, split pipe-delimited rows, split the `NPU Chip` cell, and accept a row only when device and PID are decimal integers. `read_snapshot()` must run `npu-smi info` once, run `npu-smi info -t health -i DEVICE_ID` for each ID from 0 through 7, resolve process owners through `/proc/PID` and `pwd.getpwuid`, and return unhealthy if any command fails or any device is not `OK`.

Use these definitions exactly:

```python
class ResourceTimeout(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class NpuProcess:
    device: int
    pid: int
    name: str
    owner: str = "unknown"


@dataclasses.dataclass(frozen=True)
class Snapshot:
    healthy: bool
    processes: Tuple[NpuProcess, ...]

    @property
    def idle(self):
        return self.healthy and not self.processes
```

`wait_for_idle()` must reset its stable counter after any busy or unhealthy sample, emit one status line per sample, return only after two idle samples, and raise `ResourceTimeout` once the six-hour deadline is reached.

- [ ] **Step 5: Add the test to CTest and verify**

Append to `tests/ci/CMakeLists.txt`:

```cmake
add_test(
    NAME test_tilexr_ci_npu_state
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_npu_state.py -v
)
```

Run:

```bash
python3 tests/ci/test_npu_state.py -v
ctest --test-dir tests/ci/build --output-on-failure
```

Expected: five Python cases pass and CTest remains at `100% tests passed`.

- [ ] **Step 6: Commit**

```bash
git add scripts/ci/control tests/ci
git commit -m "feat: add NPU resource state detection"
```

## Task 3: Implement The Trusted Gate Controller

**Files:**

- Create: `scripts/ci/control/gate.py`
- Create: `tests/ci/test_gate.py`
- Modify: `tests/ci/CMakeLists.txt`

**Interfaces:**

```text
gate.py --source PATH --artifacts PATH --expected-merge-sha SHA --repository LingquLab/TileXR --pr-number NUMBER
```

Exit codes are stable:

- `20`: code or test failure
- `21`: resource timeout
- `22`: resource collision
- `23`: runner or toolchain failure
- `130`: cancelled or obsolete

- [ ] **Step 1: Write controller tests first**

Create `tests/ci/test_gate.py` using `unittest.mock`. Cover these exact behaviors:

```python
def test_verify_merge_sha_rejects_wrong_checkout(self):
    with self.assertRaises(self.module.ObsoleteRun):
        self.module.verify_merge_sha("/source", "expected", run_git=lambda path: "actual")

def test_build_policy_rejects_ci_owned_npu_process(self):
    snapshot = self.state.Snapshot(True, (
        self.state.NpuProcess(0, 111, "test", "tilexr-ci"),
    ))
    self.assertEqual(self.module.policy_violation("build", snapshot, "tilexr-ci"), "ci-npu-before-lock")

def test_hardware_policy_rejects_foreign_process(self):
    snapshot = self.state.Snapshot(True, (
        self.state.NpuProcess(7, 222, "python", "other-user"),
    ))
    self.assertEqual(self.module.policy_violation("hardware", snapshot, "tilexr-ci"), "foreign-npu-process")

def test_child_environment_drops_github_tokens(self):
    env = self.module.child_environment({
        "GITHUB_TOKEN": "one",
        "TILEXR_CI_GITHUB_TOKEN": "two",
        "GITHUB_ENV": "/tmp/env",
        "GITHUB_PATH": "/tmp/path",
        "GITHUB_OUTPUT": "/tmp/output",
        "GITHUB_STEP_SUMMARY": "/tmp/summary",
        "PATH": "/bin",
    })
    self.assertEqual(env, {"PATH": "/bin", "PYTHONDONTWRITEBYTECODE": "1"})
```

Also test that `terminate_group()` sends `SIGTERM`, waits up to ten seconds, then sends `SIGKILL` only to the child process group supplied by the controller.
Test that the final Markdown report contains pull-request identity, all phase
outcomes, wait and execution durations, test counts, failed case names, and the
two explicit coverage exclusions even when a phase fails.

- [ ] **Step 2: Run and observe the missing-module failure**

```bash
python3 tests/ci/test_gate.py -v
```

Expected: import failure for `scripts/ci/control/gate.py`.

- [ ] **Step 3: Implement process and identity control**

Implement these controller units in `gate.py`:

- `verify_merge_sha()` compares `git -C SOURCE rev-parse HEAD` with the event SHA.
- `fetch_current_merge_sha()` calls `https://api.github.com/repos/LingquLab/TileXR/pulls/PR_NUMBER` with the read-only token and extracts `merge_commit_sha` using `json` and `urllib.request`.
- `child_environment()` removes `GITHUB_TOKEN`, `TILEXR_CI_GITHUB_TOKEN`,
  `ACTIONS_RUNTIME_TOKEN`, `ACTIONS_ID_TOKEN_REQUEST_TOKEN`,
  `ACTIONS_ID_TOKEN_REQUEST_URL`, `GITHUB_ENV`, `GITHUB_PATH`, `GITHUB_OUTPUT`,
  `GITHUB_STATE`, and `GITHUB_STEP_SUMMARY`, then sets
  `PYTHONDONTWRITEBYTECODE=1`. Capture the trusted summary path before creating
  the sanitized child environment.
- before launching child phases, the controller adds `PR_NUMBER`,
  `TILEXR_CI_SOURCE_DIR`, `TILEXR_CI_ARTIFACT_DIR`, and
  `TILEXR_CANN_HOME=/home/tilexr-ci/toolchains/cann/9.1.0` to the sanitized child
  environment;
- `run_phase()` uses
  `subprocess.Popen(command, cwd=cwd, env=env, start_new_session=True)`, polls at
  ten-second intervals, applies the phase policy to
  `npu_state.read_snapshot()`, enforces the phase timeout, and terminates only
  the child process group.
- signal handlers translate `SIGINT` and `SIGTERM` to exit 130 after child cleanup.
- a file lock at `/home/tilexr-ci/locks/npu8.lock` wraps idle waiting and hardware execution.
- the build phase has a two-hour timeout and forbids any process owned by `tilexr-ci` from appearing on an NPU.
- the resource waiter uses 21600 seconds, 60-second polls, and two stable samples.
- immediately before hardware execution, the controller fetches the current merge SHA and rejects an obsolete run.
- the hardware phase has a 7200-second timeout and aborts when any NPU PID is owned by an account other than `tilexr-ci`.
- the controller reads `cases.tsv` files emitted by the build and hardware
  manifests, writes `summary.md` under the artifact directory, and appends the
  same Markdown to the captured `GITHUB_STEP_SUMMARY` path when present;
- a `finally` block terminates remaining child processes, verifies no NPU
  process owned by `tilexr-ci` remains, releases the lock, determines one of the
  five stable failure classes, writes the summary, and invokes artifact
  collection.

The controller must call sibling scripts by resolving `Path(__file__).resolve().parent`; it must not call a control script from the pull-request checkout.

- [ ] **Step 4: Run focused tests**

```bash
python3 tests/ci/test_gate.py -v
```

Expected: all identity, policy, environment, and process-group tests pass.

- [ ] **Step 5: Add CTest wiring and commit**

Append:

```cmake
add_test(
    NAME test_tilexr_ci_gate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_gate.py -v
)
```

Run the complete CI unit slice, then commit:

```bash
cmake -S tests/ci -B tests/ci/build
ctest --test-dir tests/ci/build --output-on-failure
git add scripts/ci/control/gate.py tests/ci
git commit -m "feat: add trusted CI gate controller"
```

## Task 4: Define The Blue Build And Hardware Manifests

**Files:**

- Create: `scripts/ci/control/build_blue.sh`
- Create: `scripts/ci/control/run_hardware.sh`
- Create: `scripts/ci/control/collect_artifacts.sh`
- Create: `scripts/ci/control/job_completed.sh`
- Create: `tests/ci/test_control_contract.py`
- Modify: `tests/ci/CMakeLists.txt`

- [ ] **Step 1: Write failing source-contract tests**

Create `test_control_contract.py`. Read the three scripts as text and assert all required boundaries:

```python
def test_build_manifest_enables_all_optional_components(self):
    text = self.read("scripts/ci/control/build_blue.sh")
    for token in [
        "-DTILEXR_BUILD_COLLECTIVES=ON",
        "-DTILEXR_BUILD_EP=ON",
        "-DTILEXR_BUILD_CHECKER=ON",
        "-DTILEXR_BUILD_TESTS=ON",
        "-DBUILD_TESTING=ON",
        "ctest --test-dir",
        "test_tilexr_udma_transport_layout",
        "test_tilexr_udma_registry",
        "test_tilexr_udma_source_guard",
        "libascend_hal.so",
        "devlib",
    ]:
        self.assertIn(token, text)

def test_hardware_manifest_has_required_8_card_coverage(self):
    text = self.read("scripts/ci/control/run_hardware.sh")
    for token in [
        "run_tilexr_memory_demo.sh 8",
        "run_tilexr_ep_dispatch_demo.sh 8",
        "allgather alltoall allreduce reducescatter",
        "broadcast --root 0",
        "broadcast --root 7",
        "64 4096 1048576",
        "TILEXR_TEST_DEVICES=0,1,2,3,4,5,6,7",
    ]:
        self.assertIn(token, text)

def test_hardware_manifest_does_not_claim_udma_data_plane(self):
    text = self.read("scripts/ci/control/run_hardware.sh")
    self.assertNotIn("run_tilexr_udma_demo.sh", text)
    self.assertIn("UDMA data plane: out of scope on 910B3", text)

def test_job_hook_uses_fixed_workspace_and_absolute_tools(self):
    text = self.read("scripts/ci/control/job_completed.sh")
    self.assertIn("/home/tilexr-ci/actions-runner/_work/TileXR/TileXR", text)
    self.assertIn("/usr/bin/find", text)
    self.assertIn("/usr/bin/rm", text)
```

Run and confirm failure because the scripts are absent.

- [ ] **Step 2: Implement the build manifest**

`build_blue.sh` accepts `SOURCE_DIR` and `ARTIFACT_DIR`, exports the sealed CANN path, creates clean build directories, and executes this order:

```bash
export TILEXR_CANN_HOME=/home/tilexr-ci/toolchains/cann/9.1.0
source "${SOURCE_DIR}/scripts/common_env.sh"
export ARCH="${TILEXR_OS_ARCH}"

cmake -S "${SOURCE_DIR}" -B "${SOURCE_DIR}/build-ci" \
    -DCMAKE_INSTALL_PREFIX="${SOURCE_DIR}/install" \
    -DTILEXR_BUILD_COLLECTIVES=ON \
    -DTILEXR_BUILD_EP=ON \
    -DTILEXR_BUILD_CHECKER=ON \
    -DTILEXR_BUILD_TESTS=ON \
    -DBUILD_TESTING=ON
cmake --build "${SOURCE_DIR}/build-ci" --target install -j"$(nproc)"
ctest --test-dir "${SOURCE_DIR}/build-ci" --output-on-failure \
    --output-junit "${ARTIFACT_DIR}/ctest-top-level.xml"

bash "${SOURCE_DIR}/tests/comm/build.sh"
"${SOURCE_DIR}/tests/comm/install/bin/test_tilexr_log"
"${SOURCE_DIR}/tests/comm/install/bin/test_tilexr_log_spdlog_compile"
"${SOURCE_DIR}/tests/comm/install/bin/test_tilexr_source_guards"

BUILD_TILEXR_UDMA_DEMO=OFF bash "${SOURCE_DIR}/tests/udma/build.sh"
"${SOURCE_DIR}/tests/udma/install/bin/test_tilexr_udma_transport_layout"
"${SOURCE_DIR}/tests/udma/install/bin/test_tilexr_udma_registry"
"${SOURCE_DIR}/tests/udma/install/bin/test_tilexr_udma_source_guard"

bash "${SOURCE_DIR}/tests/sdma/build.sh" "${ASCEND_HOME_PATH}"
bash "${SOURCE_DIR}/tests/ep/build.sh" full

cmake -S "${SOURCE_DIR}/tests/memory" -B "${SOURCE_DIR}/tests/memory/build" \
    -DCMAKE_INSTALL_PREFIX="${SOURCE_DIR}/tests/memory/install" \
    -DBUILD_TILEXR_MEMORY_DEMO=ON \
    -DTILEXR_MEMORY_DEMO_SOC_TYPE=Ascend910B
cmake --build "${SOURCE_DIR}/tests/memory/build" --target install -j"$(nproc)"
```

After building, require the EP, memory, SDMA, collective correctness, and collective perf binaries with `test -x`. Run `ldd` on `install/lib/libtile-comm.so` or `install/lib64/libtile-comm.so`; fail if `libascend_hal.so` is unresolved or resolves through a path containing `devlib`. Record `readelf -d` output and fail if RPATH or RUNPATH contains `devlib`.

- [ ] **Step 3: Implement the hardware manifest**

`run_hardware.sh` accepts the source and artifact directories, sets a PR-derived local port, and runs:

```bash
export TILEXR_TEST_DEVICES=0,1,2,3,4,5,6,7
export TILEXR_AVAILABLE_NPUS=8
export TILEXR_SKIP_IF_INSUFFICIENT_NPUS=0
export TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=600
export TILEXR_COMM_ID="127.0.0.1:$((20000 + PR_NUMBER % 20000))"

RANK=0 RANK_SIZE=1 "${SOURCE_DIR}/tests/udma/install/bin/test_tilexr_udma"
mpirun -n 8 "${SOURCE_DIR}/tests/udma/install/bin/test_tilexr_udma"

bash "${SOURCE_DIR}/tests/memory/demo/run_tilexr_memory_demo.sh" 8 1024 8 0
bash "${SOURCE_DIR}/tests/ep/demo/run_tilexr_ep_dispatch_demo.sh" 8 8 0

for op in allgather alltoall allreduce reducescatter; do
    bash "${SOURCE_DIR}/tests/collectives/run_collectives_correctness.sh" \
        8 1024 0 "${SOURCE_DIR}/build-ci/tests/collectives" "${op}"
done
bash "${SOURCE_DIR}/tests/collectives/run_collectives_correctness.sh" \
    8 1024 0 "${SOURCE_DIR}/build-ci/tests/collectives" broadcast --root 0
bash "${SOURCE_DIR}/tests/collectives/run_collectives_correctness.sh" \
    8 1024 0 "${SOURCE_DIR}/build-ci/tests/collectives" broadcast --root 7

for op in allgather alltoall allreduce reducescatter broadcast; do
    bash "${SOURCE_DIR}/tools/collectives/run_collective_perf.sh" \
        8 0 "${SOURCE_DIR}/build-ci/tests/collectives" \
        --op "${op}" --min-bytes 4 --max-bytes 1048576 --step-factor 8 \
        --iters 3 --warmup-iters 1 --datatype int32 --check 1
done

for device in 0 1 2 3 4 5 6 7; do
    bash "${SOURCE_DIR}/tests/sdma/demo/run_tilexr_sdma_demo.sh" \
        "${ASCEND_HOME_PATH}" "${device}" 64 4096 1048576
done

echo "UDMA data plane: out of scope on 910B3"
echo "Multi-host validation: out of scope for this gate"
```

Before running tests, loop over devices 0 through 7 and require `npu-smi info -t health -i DEVICE_ID` to contain `Health` followed by `OK`. The gate controller supplies the two-hour overall timeout; individual collective scripts retain their ten-minute timeout.

Both manifest scripts use a shared `run_case NAME COMMAND...` convention that
appends one tab-separated row to `${ARTIFACT_DIR}/cases.tsv` with the case name,
result, exit code, and elapsed seconds. Every test invocation shown above has a
stable case name. A failing case is recorded before the script returns its exit
code, so the controller can report exact counts and failed names.

- [ ] **Step 4: Implement bounded artifact collection**

`collect_artifacts.sh` must always create the artifact directory, copy
environment/version output, CTest XML, dynamic dependency reports, and
discovered `*.log`, `*.xml`, `summary.csv`, and `trace.json` files while
excluding build objects, installed libraries, model files, CANN packages, and
all symbolic links. Write a `manifest.txt` containing relative path and byte
size. Refuse an individual artifact larger than 100 MiB and never follow a
pull-request-created link outside the source tree.

- [ ] **Step 5: Implement the post-job workspace hook**

`job_completed.sh` is administrator-owned when installed. It uses only absolute
tool paths, verifies that the fixed workspace is below
`/home/tilexr-ci/actions-runner/_work`, and removes every child of
`/home/tilexr-ci/actions-runner/_work/TileXR/TileXR` without following symlinks.
It leaves the workspace directory itself and the artifact directory intact so
Actions post-job processing and artifact upload have already completed before
cleanup runs.

- [ ] **Step 6: Verify and commit**

```bash
bash -n scripts/ci/control/build_blue.sh \
    scripts/ci/control/run_hardware.sh \
    scripts/ci/control/collect_artifacts.sh \
    scripts/ci/control/job_completed.sh
python3 tests/ci/test_control_contract.py -v
cmake -S tests/ci -B tests/ci/build
ctest --test-dir tests/ci/build --output-on-failure
git add scripts/ci/control tests/ci
git commit -m "ci: define blue build and hardware manifests"
```

Expected: source-contract and all earlier CI unit tests pass.

## Task 5: Add The Fast Host Check Entrypoint

**Files:**

- Create: `scripts/ci/host_checks.sh`
- Modify: `tests/ci/test_control_contract.py`

- [ ] **Step 1: Add a failing host-manifest test**

Require `host_checks.sh` to contain the CI CTest slice, comm source guard, EP source-only mode, data-as-flag CTest, all three collectives Python tests, a stable `cases.tsv` record, and an EXIT-trap summary. Run the focused test and confirm it fails because the file is absent.

- [ ] **Step 2: Implement the host script**

The script uses `set -euo pipefail`, accepts `TILEXR_CI_BUILD_ROOT`, removes only that build root, and runs:

```bash
cmake -S "${ROOT_DIR}/tests/ci" -B "${BUILD_ROOT}/ci"
(cd "${BUILD_ROOT}/ci" && ctest --output-on-failure -T Test --no-compress-output)
cp "$(find "${BUILD_ROOT}/ci/Testing" -name Test.xml -print -quit)" \
    "${ARTIFACT_DIR}/ctest-ci.xml"

cmake -S "${ROOT_DIR}/tests/comm" -B "${BUILD_ROOT}/comm" \
    -DCMAKE_INSTALL_PREFIX="${BUILD_ROOT}/comm-install"
cmake --build "${BUILD_ROOT}/comm" --target install -j"$(nproc)"
"${BUILD_ROOT}/comm-install/bin/test_tilexr_log"
"${BUILD_ROOT}/comm-install/bin/test_tilexr_log_spdlog_compile"
"${BUILD_ROOT}/comm-install/bin/test_tilexr_source_guards"

cmake -S "${ROOT_DIR}/tests/ep" -B "${BUILD_ROOT}/ep" \
    -DBUILD_TILEXR_EP_DEMO=OFF
cmake --build "${BUILD_ROOT}/ep" -j"$(nproc)"
(cd "${BUILD_ROOT}/ep" && ctest --output-on-failure -T Test --no-compress-output)
cp "$(find "${BUILD_ROOT}/ep/Testing" -name Test.xml -print -quit)" \
    "${ARTIFACT_DIR}/ctest-ep.xml"

cmake -S "${ROOT_DIR}/tests/data_as_flag" -B "${BUILD_ROOT}/data-as-flag"
cmake --build "${BUILD_ROOT}/data-as-flag" -j"$(nproc)"
(cd "${BUILD_ROOT}/data-as-flag" && ctest --output-on-failure -T Test --no-compress-output)
cp "$(find "${BUILD_ROOT}/data-as-flag/Testing" -name Test.xml -print -quit)" \
    "${ARTIFACT_DIR}/ctest-data-as-flag.xml"

python3 -m pytest -q "${ROOT_DIR}/tests/collectives/unit/test_vllm_collectives_patch.py"
python3 -m pytest -q "${ROOT_DIR}/tests/collectives/unit/test_vllm_collectives_integration_sources.py"
python3 "${ROOT_DIR}/tests/collectives/unit/test_collective_profile_report.py"
```

Before the builds, run `bash -n` over tracked shell files under `scripts/ci`, `tests/comm`, `tests/ep`, `tests/memory`, `tests/sdma`, `tests/udma`, and `tests/collectives`. Do not scan ignored generated files.

Wrap each host suite in `run_case NAME COMMAND...` so
`${ARTIFACT_DIR}/cases.tsv` records its result, exit code, and duration. An EXIT
trap writes `summary.md` and, when `GITHUB_STEP_SUMMARY` is set, appends the
same report there. Include PR, head, base, and merge identity from the
`TILEXR_CI_*` environment variables, host case counts, failed names, duration,
and the explicit NPU-only coverage note. Local runs use `local` for missing
GitHub identity values.

- [ ] **Step 3: Run the host checks and commit**

```bash
bash scripts/ci/host_checks.sh
git diff --check
git add scripts/ci/host_checks.sh tests/ci/test_control_contract.py
git commit -m "ci: add fast host validation entrypoint"
```

Expected: all host suites pass and artifacts appear under `.ci-artifacts/host`.

## Task 6: Add Workflows And CODEOWNERS

**Files:**

- Create: `.github/CODEOWNERS`
- Create: `.github/workflows/pr-ci.yml`
- Create: `.github/workflows/npu-ci.yml`
- Create: `tests/ci/test_workflows.rb`
- Modify: `tests/ci/CMakeLists.txt`

Use these immutable Action revisions verified from the official repositories on 2026-07-17:

- `actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0` (`v7.0.0`)
- `actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a` (`v7.0.1`)

- [ ] **Step 1: Write the structured workflow test**

Use Ruby `YAML.safe_load_file(path, aliases: true)` to assert:

- `pr-ci.yml` uses only `pull_request`, targets `main`, and lists `opened`, `reopened`, `synchronize`, and `ready_for_review`;
- top-level permissions are `contents: read`;
- per-PR concurrency cancels obsolete runs;
- the NPU caller uses `LingquLab/TileXR/.github/workflows/npu-ci.yml@main`;
- `npu-ci.yml` uses `workflow_call`, group `TileXR-NPU`, and labels `self-hosted`, `Linux`, `ARM64`, `tilexr`, `ascend910b`, and `npu8`;
- checkout and artifact actions use the exact SHAs above;
- checkout disables credential persistence and selects the PR merge ref;
- the host artifact upload explicitly includes the `.ci-artifacts` hidden path;
- the NPU job timeout is 660 minutes and artifact retention is 14 days;
- host, NPU, and final gate summaries contain PR, head, base, and merge identity;
- no workflow text contains `pull_request_target`, `secrets: inherit`, or a write permission;
- CODEOWNERS assigns `@LingquLab/ci-maintainers` to `.github/CODEOWNERS`,
  `.github/workflows/`, `.github/actions/`, `scripts/ci/control/`,
  `scripts/ci/provision/`, `scripts/cann_download_install.sh`,
  `scripts/cann_local_install.sh`, and `.gitmodules`.

Run it before creating workflows and expect a missing-file failure.

- [ ] **Step 2: Create `pr-ci.yml`**

Use this job graph:

```yaml
name: PR CI

"on":
  pull_request:
    branches: [main]
    types: [opened, reopened, synchronize, ready_for_review]

permissions:
  contents: read

concurrency:
  group: tilexr-pr-${{ github.event.pull_request.number }}
  cancel-in-progress: true

jobs:
  host_checks:
    name: Host Checks
    runs-on: ubuntu-24.04
    container: ubuntu:20.04
    steps:
      - name: Install host dependencies
        run: apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential cmake git python3 python3-pytest ruby
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          persist-credentials: false
          submodules: recursive
      - name: Run host checks
        env:
          TILEXR_CI_PR_NUMBER: ${{ github.event.pull_request.number }}
          TILEXR_CI_HEAD_SHA: ${{ github.event.pull_request.head.sha }}
          TILEXR_CI_BASE_SHA: ${{ github.event.pull_request.base.sha }}
          TILEXR_CI_EXPECTED_MERGE_SHA: ${{ github.event.pull_request.merge_commit_sha }}
        run: bash scripts/ci/host_checks.sh
      - name: Upload host evidence
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a
        with:
          name: host-checks-${{ github.event.pull_request.number }}-${{ github.run_attempt }}
          path: .ci-artifacts/host
          include-hidden-files: true
          retention-days: 14

  npu_gate:
    name: NPU Gate
    needs: host_checks
    if: ${{ needs.host_checks.result == 'success' && !github.event.pull_request.draft }}
    uses: LingquLab/TileXR/.github/workflows/npu-ci.yml@main
    permissions:
      contents: read

  pr_gate:
    name: PR Gate
    if: always()
    needs: [host_checks, npu_gate]
    runs-on: ubuntu-24.04
    steps:
      - name: Evaluate required jobs
        env:
          IS_DRAFT: ${{ github.event.pull_request.draft }}
          HOST_RESULT: ${{ needs.host_checks.result }}
          NPU_RESULT: ${{ needs.npu_gate.result }}
        run: |
          test "${HOST_RESULT}" = success
          if test "${IS_DRAFT}" != true; then
            test "${NPU_RESULT}" = success
          fi
      - name: Summarize gate
        if: always()
        env:
          PR_NUMBER: ${{ github.event.pull_request.number }}
          HEAD_SHA: ${{ github.event.pull_request.head.sha }}
          BASE_SHA: ${{ github.event.pull_request.base.sha }}
          MERGE_SHA: ${{ github.event.pull_request.merge_commit_sha }}
          IS_DRAFT: ${{ github.event.pull_request.draft }}
          HOST_RESULT: ${{ needs.host_checks.result }}
          NPU_RESULT: ${{ needs.npu_gate.result }}
        run: |
          {
            echo '## PR Gate'
            echo
            printf -- '- PR: #%s\n' "${PR_NUMBER}"
            printf -- '- Head SHA: `%s`\n' "${HEAD_SHA}"
            printf -- '- Base SHA: `%s`\n' "${BASE_SHA}"
            printf -- '- Merge SHA: `%s`\n' "${MERGE_SHA}"
            printf -- '- Draft: `%s`\n' "${IS_DRAFT}"
            printf -- '- Host Checks: `%s`\n' "${HOST_RESULT}"
            printf -- '- NPU Gate: `%s`\n' "${NPU_RESULT}"
          } >>"${GITHUB_STEP_SUMMARY}"
```

- [ ] **Step 3: Create `npu-ci.yml`**

```yaml
name: Trusted NPU CI

"on":
  workflow_call:

permissions:
  contents: read

jobs:
  hardware:
    name: NPU Gate
    timeout-minutes: 660
    runs-on:
      group: TileXR-NPU
      labels: [self-hosted, Linux, ARM64, tilexr, ascend910b, npu8]
    steps:
      - name: Verify sealed controller
        run: test "$(cat /home/tilexr-ci/control/current/VERSION)" = v1
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          ref: refs/pull/${{ github.event.pull_request.number }}/merge
          path: source
          clean: true
          submodules: recursive
          persist-credentials: false
      - name: Verify merge identity
        env:
          EXPECTED_MERGE_SHA: ${{ github.event.pull_request.merge_commit_sha }}
        run: test "$(git -C source rev-parse HEAD)" = "${EXPECTED_MERGE_SHA}"
      - name: Run trusted gate
        env:
          TILEXR_CI_GITHUB_TOKEN: ${{ github.token }}
          TILEXR_CI_PR_NUMBER: ${{ github.event.pull_request.number }}
          TILEXR_CI_HEAD_SHA: ${{ github.event.pull_request.head.sha }}
          TILEXR_CI_BASE_SHA: ${{ github.event.pull_request.base.sha }}
          TILEXR_CI_EXPECTED_MERGE_SHA: ${{ github.event.pull_request.merge_commit_sha }}
          TILEXR_CI_REPOSITORY: ${{ github.repository }}
          TILEXR_CI_ARTIFACT_DIR: ${{ runner.temp }}/tilexr-ci-artifacts
        run: |
          python3 /home/tilexr-ci/control/current/gate.py \
            --source "${GITHUB_WORKSPACE}/source" \
            --artifacts "${TILEXR_CI_ARTIFACT_DIR}" \
            --expected-merge-sha "${TILEXR_CI_EXPECTED_MERGE_SHA}" \
            --repository "${TILEXR_CI_REPOSITORY}" \
            --pr-number "${TILEXR_CI_PR_NUMBER}"
      - name: Upload NPU evidence
        if: always()
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a
        with:
          name: npu-gate-${{ github.event.pull_request.number }}-${{ github.run_attempt }}
          path: ${{ runner.temp }}/tilexr-ci-artifacts
          retention-days: 14
```

- [ ] **Step 4: Add CODEOWNERS and CTest wiring**

```text
/.github/CODEOWNERS @LingquLab/ci-maintainers
/.github/workflows/ @LingquLab/ci-maintainers
/.github/actions/ @LingquLab/ci-maintainers
/scripts/ci/control/ @LingquLab/ci-maintainers
/scripts/ci/provision/ @LingquLab/ci-maintainers
/scripts/cann_download_install.sh @LingquLab/ci-maintainers
/scripts/cann_local_install.sh @LingquLab/ci-maintainers
/.gitmodules @LingquLab/ci-maintainers
```

Append the structured validator to `tests/ci/CMakeLists.txt` only when Ruby is
available:

```cmake
if(RUBY_EXECUTABLE)
    add_test(
        NAME test_tilexr_ci_workflows
        COMMAND ${RUBY_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_workflows.rb
    )
endif()
```

The GitHub-hosted job installs Ruby, so this test is mandatory there; the blue
build may omit only this YAML parser test when Ruby is unavailable.

- [ ] **Step 5: Verify and commit**

```bash
ruby tests/ci/test_workflows.rb
cmake -S tests/ci -B tests/ci/build
ctest --test-dir tests/ci/build --output-on-failure
git diff --check
git add .github tests/ci
git commit -m "ci: add GitHub Actions PR gate"
```

Expected: structured validation passes and the workflow has no write permission or secret inheritance.

## Task 7: Add Idempotent Blue Provisioning

**Files:**

- Create: `scripts/ci/provision/common.sh`
- Create: `scripts/ci/provision/account.sh`
- Create: `scripts/ci/provision/cann.sh`
- Create: `scripts/ci/provision/control.sh`
- Create: `scripts/ci/provision/runner.sh`
- Create: `scripts/ci/provision/verify.sh`
- Create: `tests/ci/test_provision_dry_run.sh`
- Modify: `scripts/cann_download_install.sh`
- Modify: `tests/ci/CMakeLists.txt`

- [ ] **Step 1: Write the dry-run contract**

The test invokes every provisioning script with `--dry-run` and asserts output contains:

```text
useradd --system --create-home --home-dir /home/tilexr-ci --shell /usr/sbin/nologin
usermod -aG HwHiAiUser tilexr-ci
/home/tilexr-ci/toolchains/cann/9.1.0
Ascend-cann-toolkit_9.1.0_linux-aarch64.run
Ascend-cann-910b-ops_9.1.0_linux-aarch64.run
/home/tilexr-ci/control/v1
/home/tilexr-ci/control/current/job_completed.sh
ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/home/tilexr-ci/control/current/job_completed.sh
TileXR-NPU
blue-tilexr-npu8
tilexr,ascend910b,npu8
```

It must reject any output containing `/etc/sudoers`, `docker group`,
`/usr/local/Ascend/ascend-toolkit`, `curl -k`, or a registration token value.

- [ ] **Step 2: Implement shared dry-run handling**

`common.sh` defines exact constants and a `run()` helper:

```bash
CI_USER=tilexr-ci
CI_GROUP=HwHiAiUser
CI_HOME=/home/tilexr-ci
CANN_HOME=/home/tilexr-ci/toolchains/cann/9.1.0
CONTROL_HOME=/home/tilexr-ci/control/v1
RUNNER_HOME=/home/tilexr-ci/actions-runner

run() {
    if [[ "${DRY_RUN}" == 1 ]]; then
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}
```

Every script accepts only `--dry-run` or no option, uses `set -euo pipefail`, and is safe to rerun.

- [ ] **Step 3: Implement account and toolchain provisioning**

`account.sh` creates the locked system account only if absent, sets its shell to
`/usr/sbin/nologin`, adds it to `HwHiAiUser`, and creates runner, work, lock,
artifact, toolchain, and control parent directories under `/home/tilexr-ci`.
It locks the password, does not create `authorized_keys`, and never grants sudo
or Docker access.

`cann.sh` requires driver 25.5.0, eight visible 910B3 devices, and at least
30 GiB free on `/home` before downloading. It copies the trusted installer
scripts to a fresh
root-owned staging directory below `/home/tilexr-ci/install-work`, invokes the
existing CANN downloader there with `TILEXR_CANN_HOME=${CANN_HOME}`, verifies
Toolkit then 910B Ops, and seals the completed tree with owner
`root:HwHiAiUser`, no group/other write bits, and preserved executable bits.
Keep downloads and temporary installation files under `/home/tilexr-ci`, not
`/`.

Replace the downloader's `curl -k -C - -O` call in
`scripts/cann_download_install.sh` with TLS-verifying
`curl --fail --location --continue-at - --remote-name`. Require the downloaded
toolkit and 910B Ops files to have the exact names for 9.1.0/aarch64 and to be
nonempty before invoking the installers. The HTTPS endpoint was verified from
`blue` without `-k`; do not add an insecure fallback.

- [ ] **Step 4: Implement control and runner provisioning**

`control.sh` requires repository control version `v1`, installs it to `/home/tilexr-ci/control/v1`, applies `root:HwHiAiUser` ownership and read/execute-only permissions, and atomically updates `/home/tilexr-ci/control/current`.

`runner.sh` reads a short-lived registration token from standard input, never echoes it, downloads the latest stable ARM64 Actions runner from the official release API, verifies the release asset SHA-256 digest, configures:

```text
URL: https://github.com/LingquLab
Runner group: TileXR-NPU
Runner name: blue-tilexr-npu8
Labels: tilexr,ascend910b,npu8
Work directory: _work
```

Write this root-owned entry to `${RUNNER_HOME}/.env` before starting the
service:

```text
ACTIONS_RUNNER_HOOK_JOB_COMPLETED=/home/tilexr-ci/control/current/job_completed.sh
```

Then install and start the systemd service as `tilexr-ci`. Re-running replaces
an offline registration with the same name but refuses to replace an online
runner.

- [ ] **Step 5: Implement verification**

`verify.sh` exits nonzero unless:

- `tilexr-ci` exists, belongs to `HwHiAiUser`, has a locked password and
  `nologin` shell, has no `authorized_keys`, and has no sudo rule;
- CANN is 9.1.0 and `bisheng` resolves from the sealed path;
- all eight devices are healthy;
- the sealed controller is version `v1` and not writable by `tilexr-ci`;
- the runner service is active;
- the job-completed hook is configured, executable, and not writable by
  `tilexr-ci`;
- root and home filesystem usage is recorded;
- no NPU process owned by `tilexr-ci` remains before activation.

- [ ] **Step 6: Verify and commit**

```bash
bash tests/ci/test_provision_dry_run.sh
bash -n scripts/ci/provision/*.sh
ctest --test-dir tests/ci/build --output-on-failure
git add scripts/cann_download_install.sh scripts/ci/provision tests/ci
git commit -m "ci: add blue runner provisioning"
```

## Task 8: Add The Operator Runbook, Create The Maintainer Team, And Verify Repository Changes

**Files:**

- Create: `docs/CI.md`
- Modify: `scripts/README.md`
- Verify: all files from Tasks 1-7

- [ ] **Step 1: Document exact operations**

`docs/CI.md` must cover:

- workflow and trust model;
- account, group, CANN, controller, and runner paths;
- six-hour resource wait and two-hour hardware timeout;
- exact test matrix and explicit exclusions;
- failure class meanings;
- GitHub settings and runner-group verification commands;
- artifact locations and 14-day retention;
- runner upgrade procedure;
- required-check enable and rollback commands;
- server rollback that stops the runner without deleting the toolchain.

Add a short CI section to `scripts/README.md` linking to `docs/CI.md` and listing the host, control, and provisioning entrypoints.

- [ ] **Step 2: Run the complete repository validation**

```bash
bash scripts/ci/host_checks.sh
bash tests/ci/test_provision_dry_run.sh
ruby tests/ci/test_workflows.rb
bash -n scripts/ci/host_checks.sh scripts/ci/control/*.sh scripts/ci/provision/*.sh
git diff --check
git status --short
```

Expected: every command exits zero; only CI implementation, approved design correction, plan, and documentation files are changed.

- [ ] **Step 3: Commit documentation**

```bash
git add docs/CI.md scripts/README.md \
  docs/superpowers/specs/2026-07-17-github-actions-pr-gate-design.md \
  docs/superpowers/plans/2026-07-17-github-actions-pr-gate.md
git commit -m "docs: add TileXR CI operations guide"
```

- [ ] **Step 4: Create and verify the CODEOWNERS team**

Authorize the current GitHub CLI login for organization administration, create
the closed team if necessary, make the two requested users team maintainers,
and grant the team write access to TileXR:

```bash
gh auth refresh -h github.com -s admin:org
gh api user/memberships/orgs/LingquLab --jq '[.state,.role] | @tsv'

if gh api orgs/LingquLab/teams/ci-maintainers >/dev/null 2>&1; then
  gh api --method PATCH orgs/LingquLab/teams/ci-maintainers \
    -f name=ci-maintainers \
    -f privacy=closed \
    -f description='Maintains the TileXR CI trust boundary' >/dev/null
else
  gh api --method POST orgs/LingquLab/teams \
    -f name=ci-maintainers \
    -f privacy=closed \
    -f description='Maintains the TileXR CI trust boundary' >/dev/null
fi

for login in Kur0x chaowick; do
  gh api --method PUT \
    "orgs/LingquLab/teams/ci-maintainers/memberships/${login}" \
    -f role=maintainer >/dev/null
done

gh api --method PUT \
  orgs/LingquLab/teams/ci-maintainers/repos/LingquLab/TileXR \
  -f permission=push >/dev/null
```

Expected: the membership query reports `active` and `admin`. Verify the team
has exactly the two requested members, both have the `maintainer` team role,
and TileXR is its only repository:

```bash
for login in Kur0x chaowick; do
  gh api "orgs/LingquLab/members/${login}" >/dev/null
done

diff -u \
  <(printf '%s\n' Kur0x chaowick | sort -f) \
  <(gh api --paginate orgs/LingquLab/teams/ci-maintainers/members \
      --jq '.[].login' | sort -f)

for login in Kur0x chaowick; do
  [[ "$(gh api \
    "orgs/LingquLab/teams/ci-maintainers/memberships/${login}" \
    --jq '[.state,.role] | @tsv')" == $'active\tmaintainer' ]]
done

diff -u \
  <(printf '%s\n' LingquLab/TileXR) \
  <(gh api --paginate orgs/LingquLab/teams/ci-maintainers/repos \
      --jq '.[].full_name' | sort)

[[ "$(gh api repos/LingquLab/TileXR/teams --jq \
  '.[] | select(.slug == "ci-maintainers") | .permission')" == push ]]
```

If either `diff` finds an unexpected existing member or repository, stop for
review instead of deleting that association automatically.

- [ ] **Step 5: Push the implementation PR**

Push the implementation branch and create a non-draft PR targeting `main`. The PR body must list host-test evidence, security boundaries, external configuration still pending, and the fact that `PR Gate` is not yet required.

Obtain two distinct approvals. At least one must be a CODEOWNER approval from
`@LingquLab/ci-maintainers`; if `Kur0x` authored the PR, `chaowick` must provide
that approval, and vice versa. Merge the PR only after both approvals. Do not
configure the runner group to accept the workflow until the trusted workflow
exists on `main`.

## Task 9: Configure GitHub Actions Policy And Runner Group

**External state:** `LingquLab` organization and `LingquLab/TileXR` repository.

- [ ] **Step 1: Reconfirm organization Actions authority**

Run locally:

```bash
gh api user/memberships/orgs/LingquLab --jq '[.state,.role] | @tsv'
```

Expected: `active` and `admin`. The team operation in Task 8 has already proven
that the token has the required `admin:org` authorization; do not continue if
the organization role has changed.

- [ ] **Step 2: Restrict repository Actions policy**

```bash
gh api --method PUT repos/LingquLab/TileXR/actions/permissions \
  -F enabled=true -f allowed_actions=selected -F sha_pinning_required=true

SELECTED_ACTIONS_PAYLOAD="$(mktemp)"
trap 'rm -f "${SELECTED_ACTIONS_PAYLOAD}"' EXIT
jq -n '{
  github_owned_allowed: false,
  verified_allowed: false,
  patterns_allowed: [
    "actions/checkout@*",
    "actions/upload-artifact@*"
  ]
}' >"${SELECTED_ACTIONS_PAYLOAD}"
gh api --method PUT \
  repos/LingquLab/TileXR/actions/permissions/selected-actions \
  --input "${SELECTED_ACTIONS_PAYLOAD}"

gh api --method PUT repos/LingquLab/TileXR/actions/permissions/workflow \
  -f default_workflow_permissions=read -F can_approve_pull_request_reviews=false

gh api --method PUT repos/LingquLab/TileXR/actions/permissions/fork-pr-contributor-approval \
  -f approval_policy=all_external_contributors
```

Read all four endpoints back. Expected: Actions enabled, only repository-local
Actions plus `actions/checkout@*` and `actions/upload-artifact@*` allowed, SHA
pinning enabled, read-only token, PR approval disabled, and
`all_external_contributors` set.

- [ ] **Step 3: Create the restricted runner group**

Use an input file so booleans and arrays keep their JSON types:

```bash
REPO_ID="$(gh api repos/LingquLab/TileXR --jq .id)"
GROUP_ID="$(gh api orgs/LingquLab/actions/runner-groups --jq \
  '.runner_groups[] | select(.name == "TileXR-NPU") | .id')"
GROUP_PAYLOAD="$(mktemp)"
GROUP_UPDATE_PAYLOAD="$(mktemp)"
trap 'rm -f "${GROUP_PAYLOAD}" "${GROUP_UPDATE_PAYLOAD}"' EXIT

jq -n --argjson repo_id "${REPO_ID}" '{
  name: "TileXR-NPU",
  visibility: "selected",
  allows_public_repositories: true,
  restricted_to_workflows: true,
  selected_workflows: [
    "LingquLab/TileXR/.github/workflows/npu-ci.yml@refs/heads/main"
  ],
  selected_repository_ids: [$repo_id]
}' >"${GROUP_PAYLOAD}"
jq 'del(.selected_repository_ids)' "${GROUP_PAYLOAD}" >"${GROUP_UPDATE_PAYLOAD}"

if [[ -z "${GROUP_ID}" ]]; then
  GROUP_ID="$(gh api --method POST orgs/LingquLab/actions/runner-groups \
    --input "${GROUP_PAYLOAD}" --jq .id)"
else
  gh api --method PATCH "orgs/LingquLab/actions/runner-groups/${GROUP_ID}" \
    --input "${GROUP_UPDATE_PAYLOAD}" >/dev/null
fi

gh api "orgs/LingquLab/actions/runner-groups/${GROUP_ID}/repositories" \
  --jq '.repositories[].id' | while read -r existing_id; do
    if [[ "${existing_id}" != "${REPO_ID}" ]]; then
      gh api --method DELETE \
        "orgs/LingquLab/actions/runner-groups/${GROUP_ID}/repositories/${existing_id}"
    fi
  done
gh api --method PUT \
  "orgs/LingquLab/actions/runner-groups/${GROUP_ID}/repositories/${REPO_ID}"
```

Expected: create or update exactly one group, remove any unrelated selected
repository, and retain only TileXR.

- [ ] **Step 4: Verify the group**

```bash
gh api orgs/LingquLab/actions/runner-groups --jq \
  '.runner_groups[] | select(.name == "TileXR-NPU") | {visibility,allows_public_repositories,restricted_to_workflows,selected_workflows}'
```

Expected: selected visibility, public repository allowed, workflow restriction true, and exactly the `npu-ci.yml@refs/heads/main` workflow.

## Task 10: Provision And Register Blue

**External state:** `blue` server and the `TileXR-NPU` runner group.

- [ ] **Step 1: Check out the merged trusted code on blue**

```bash
CI_SHA="$(gh api repos/LingquLab/TileXR/commits/main --jq .sha)"
BOOTSTRAP="/home/d00520898/tilexr-ci-bootstrap-${CI_SHA}"
ssh blue "test ! -e '${BOOTSTRAP}'"
ssh blue "git clone --branch main --depth 1 https://github.com/LingquLab/TileXR.git '${BOOTSTRAP}'"
ssh blue "git -C '${BOOTSTRAP}' checkout --detach '${CI_SHA}'"
```

Expected: a new commit-addressed checkout succeeds under `/home`, its HEAD is
`CI_SHA`, and no pre-existing path is deleted.

- [ ] **Step 2: Provision the account, CANN, and controller**

```bash
CI_SHA="$(gh api repos/LingquLab/TileXR/commits/main --jq .sha)"
BOOTSTRAP="/home/d00520898/tilexr-ci-bootstrap-${CI_SHA}"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/account.sh"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/cann.sh"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/control.sh"
```

Expected: `tilexr-ci` belongs to `HwHiAiUser`, CANN 9.1 is sealed under `/home/tilexr-ci/toolchains/cann/9.1.0`, and `/home/tilexr-ci/control/current/VERSION` is `v1`.

- [ ] **Step 3: Register and start the runner without logging the token**

```bash
CI_SHA="$(gh api repos/LingquLab/TileXR/commits/main --jq .sha)"
BOOTSTRAP="/home/d00520898/tilexr-ci-bootstrap-${CI_SHA}"
gh api --method POST orgs/LingquLab/actions/runners/registration-token --jq .token | \
  ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/runner.sh"
```

Expected: systemd service active and runner `blue-tilexr-npu8` online in `TileXR-NPU`.

- [ ] **Step 4: Run server acceptance checks**

```bash
CI_SHA="$(gh api repos/LingquLab/TileXR/commits/main --jq .sha)"
BOOTSTRAP="/home/d00520898/tilexr-ci-bootstrap-${CI_SHA}"
ssh blue "cd '${BOOTSTRAP}' && sudo bash scripts/ci/provision/verify.sh"
gh api orgs/LingquLab/actions/runners --jq \
  '.runners[] | select(.name == "blue-tilexr-npu8") | {status,busy,labels:[.labels[].name]}'
```

Expected: verification exits zero; runner is `online`, not busy, and exposes all required labels.

## Task 11: Exercise Trial Pull Requests

**External state:** temporary pull requests and Actions runs in `LingquLab/TileXR`.

- [ ] **Step 1: Create a positive empty-commit PR**

From a clean `origin/main` checkout:

```bash
git switch -c ci/pr-gate-positive
git commit --allow-empty -m "test: exercise PR gate"
git push -u origin ci/pr-gate-positive
gh pr create --base main --head ci/pr-gate-positive \
  --title "test: exercise PR gate" \
  --body "Temporary PR for validating the new required CI path."
```

Record `PR_NUMBER` from `gh pr view --json number --jq .number`. Monitor until `Host Checks`, `NPU Gate`, and `PR Gate` complete. Do not merge this PR.

- [ ] **Step 2: Verify queue and resource behavior**

While the first NPU run is waiting or running, create a second empty-commit PR from `origin/main`. Verify with the Actions API that only one NPU job is `in_progress` and the other remains `queued`. Confirm `blue` has only one Actions worker job and one CI lock holder.

- [ ] **Step 3: Verify the sealed resource-timeout behavior**

Exercise the installed waiter with a synthetic foreign NPU process and a
shortened 120-second fake clock. Patch signaling to fail the probe if the
waiter ever attempts it:

```bash
ssh blue 'sudo -u tilexr-ci env PYTHONPATH=/home/tilexr-ci/control/current python3 -' <<'PY'
import os
from unittest import mock

from npu_state import NpuProcess, ResourceTimeout, Snapshot, wait_for_idle

clock = [0]
messages = []

def sleep(seconds):
    clock[0] += seconds

def read_snapshot():
    return Snapshot(True, (NpuProcess(0, 424242, "foreign-test", "other-user"),))

with mock.patch.object(os, "kill", side_effect=AssertionError("waiter attempted to signal")):
    try:
        wait_for_idle(
            read_snapshot=read_snapshot,
            sleep=sleep,
            now=lambda: clock[0],
            emit=messages.append,
            max_wait_seconds=120,
            poll_seconds=30,
            stable_samples=2,
        )
    except ResourceTimeout:
        pass
    else:
        raise AssertionError("expected ResourceTimeout")

assert 120 <= clock[0] <= 150
assert messages
print("resource-timeout validated without signaling")
PY
```

Expected: the final line is printed and the command exits zero. This proves the
installed sealed waiter times out without owning a signaling path; the live
queue check remains read-only toward foreign NPU processes.

- [ ] **Step 4: Verify obsolete-run cancellation**

Push another empty commit to the first PR. Expected: its old run becomes cancelled, the new run uses the new merge SHA, and no `tilexr-ci` NPU process survives cancellation.

- [ ] **Step 5: Verify a host failure does not schedule blue**

Create a temporary PR branch that adds `exit 91` as the first executable line after `set -euo pipefail` in `scripts/ci/host_checks.sh`. Expected: `Host Checks` fails with 91, `NPU Gate` is skipped, and `PR Gate` fails. Close the PR and delete the branch without merging.

- [ ] **Step 6: Verify hardware failure cleanup**

Create a temporary PR branch that inserts `exit 92` immediately after
`set -euo pipefail` in
`tests/memory/demo/run_tilexr_memory_demo.sh`. Shell syntax and host checks still
pass, but the sealed hardware manifest reaches that PR-controlled script and
fails. Expected: NPU Gate reports `code-or-test-failure`, uploads rank/build
evidence, releases the lock, and leaves no NPU process owned by `tilexr-ci`.
Close and delete the branch without merging.

- [ ] **Step 7: Verify external-fork approval**

Create or update a controlled personal fork branch with an empty commit and open a PR to `LingquLab/TileXR:main`. Expected: the workflow remains awaiting approval until a maintainer approves it. After approval, it follows the same gate. Close the PR after verification.

- [ ] **Step 8: Close temporary PRs and delete temporary branches**

Close all smoke PRs, delete their remote branches, and confirm the runner
returns online and idle. Verify the completed hook left the repository
workspace empty:

```bash
ssh blue 'test -z "$(/usr/bin/find /home/tilexr-ci/actions-runner/_work/TileXR/TileXR -mindepth 1 -maxdepth 1 -print -quit)"'
```

Keep the successful run ID and merge SHA for ruleset configuration evidence.

## Task 12: Enable And Verify The Required Check

**External state:** active default-branch ruleset.

- [ ] **Step 1: Resolve the actual check and GitHub Actions integration ID**

Using the successful positive PR:

```bash
PR_NUMBER="$(gh pr list --repo LingquLab/TileXR --state all \
  --head ci/pr-gate-positive --json number --jq '.[0].number')"
MERGE_SHA="$(gh api "repos/LingquLab/TileXR/pulls/${PR_NUMBER}" --jq .merge_commit_sha)"
ACTIONS_APP_ID="$(gh api "repos/LingquLab/TileXR/commits/${MERGE_SHA}/check-runs" --jq \
  '[.check_runs[] | select(.name == "PR Gate" and .conclusion == "success")] | first | .app.id')"
test -n "${ACTIONS_APP_ID}"
gh api "repos/LingquLab/TileXR/commits/${MERGE_SHA}/check-runs" --jq \
  '.check_runs[] | select(.name == "PR Gate") | [.name,.conclusion,.app.id] | @tsv'
```

Expected: one `PR Gate`, conclusion `success`, and `ACTIONS_APP_ID` is a decimal
integer belonging to the GitHub Actions App.

- [ ] **Step 2: Preserve and extend the existing ruleset**

Resolve the active ruleset ID by name `master`, fetch its complete JSON, and
generate an update payload that preserves every existing rule while replacing
only the required-status-check rule:

```bash
PR_NUMBER="$(gh pr list --repo LingquLab/TileXR --state all \
  --head ci/pr-gate-positive --json number --jq '.[0].number')"
MERGE_SHA="$(gh api "repos/LingquLab/TileXR/pulls/${PR_NUMBER}" --jq .merge_commit_sha)"
ACTIONS_APP_ID="$(gh api "repos/LingquLab/TileXR/commits/${MERGE_SHA}/check-runs" --jq \
  '[.check_runs[] | select(.name == "PR Gate" and .conclusion == "success")] | first | .app.id')"
[[ "${ACTIONS_APP_ID}" =~ ^[0-9]+$ ]]

RULESET_ID="$(gh api repos/LingquLab/TileXR/rulesets --jq \
  '.[] | select(.name == "master" and .target == "branch") | .id')"
test -n "${RULESET_ID}"
RULESET_BEFORE="$(mktemp)"
RULESET_PAYLOAD="$(mktemp)"
trap 'rm -f "${RULESET_BEFORE}" "${RULESET_PAYLOAD}"' EXIT

gh api "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" >"${RULESET_BEFORE}"
jq --argjson app_id "${ACTIONS_APP_ID}" '{
  name: .name,
  target: .target,
  enforcement: .enforcement,
  bypass_actors: (.bypass_actors // []),
  conditions: .conditions,
  rules: (
    [.rules[] | select(.type != "required_status_checks")] +
    [{
      type: "required_status_checks",
      parameters: {
        do_not_enforce_on_create: false,
        required_status_checks: [{context: "PR Gate", integration_id: $app_id}],
        strict_required_status_checks_policy: true
      }
    }]
  )
}' "${RULESET_BEFORE}" >"${RULESET_PAYLOAD}"

diff -u \
  <(jq -S '{name,target,enforcement,bypass_actors,conditions,rules}' "${RULESET_BEFORE}") \
  <(jq -S . "${RULESET_PAYLOAD}") || true

gh api --method PUT "repos/LingquLab/TileXR/rulesets/${RULESET_ID}" \
  --input "${RULESET_PAYLOAD}" >/dev/null
```

Before the PUT, inspect the diff and confirm that it contains only the new
`required_status_checks` rule.

- [ ] **Step 3: Verify the rule without bypassing it**

Read the ruleset back. Expected:

- two approvals remain required;
- code-owner review and resolved threads remain required;
- deletion and non-fast-forward rules remain;
- bypass actors remain empty;
- `PR Gate` is required from the GitHub Actions App;
- strict up-to-date policy is true.

- [ ] **Step 4: Verify merge blocking and success**

Create one final empty-commit PR. Before CI completes,
`gh pr view --json mergeStateStatus` must report a blocked state. After
`PR Gate` succeeds, obtain two distinct approvals, including a CODEOWNER
approval from the `ci-maintainers` member who did not author the PR. The merge
state must become mergeable. Close the PR without merging after the verification
unless the user explicitly chooses to retain it.

- [ ] **Step 5: Final operational verification**

Run:

```bash
gh api repos/LingquLab/TileXR/actions/permissions
gh api repos/LingquLab/TileXR/actions/permissions/selected-actions
gh api repos/LingquLab/TileXR/actions/permissions/workflow
gh api repos/LingquLab/TileXR/actions/permissions/fork-pr-contributor-approval
gh api --paginate orgs/LingquLab/teams/ci-maintainers/members --jq '.[].login'
gh api --paginate orgs/LingquLab/teams/ci-maintainers/repos --jq '.[].full_name'
gh api repos/LingquLab/TileXR/rulesets
gh api orgs/LingquLab/actions/runner-groups
gh api orgs/LingquLab/actions/runners
ssh blue 'sudo systemctl status actions.runner.LingquLab.* --no-pager'
ssh blue 'sudo -u tilexr-ci npu-smi info'
ssh blue 'pgrep -u tilexr-ci -af "tilexr|mpirun|python" || true'
```

Expected: policies match the design, runner is online and idle, all devices remain healthy, and no orphan CI test process exists.

Document the successful run URL, runner version, CANN version, driver version, wait duration, test duration, and any accepted warnings in the implementation PR or follow-up operations record.
