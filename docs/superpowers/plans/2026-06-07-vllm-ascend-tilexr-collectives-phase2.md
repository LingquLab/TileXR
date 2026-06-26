# vllm-ascend TileXR Collectives Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `tests/collectives/deploy_and_run_vllm_remote.sh` select a reproducible remote `torch_npu` Python environment and pass the existing TileXR vllm collectives shim smoke on `ssh blue`.

**Architecture:** Keep Phase 2 outside the vllm-ascend inference path. Add source-level guards first, then thread one selected remote Python through environment probes, preflight import checks, and the multi-rank smoke launcher. The remote TileXR CANN/CMake build path remains unchanged, while Python selection is explicit through environment variables.

**Tech Stack:** Bash, SSH, rsync, conda activation, Python 3, torch-npu, CMake, CTest, pytest, TileXR ctypes shim.

---

## Scope Check

This plan implements the approved Phase 2 spec in `docs/superpowers/specs/2026-06-05-vllm-ascend-tilexr-collectives-phase2-design.md`. It does not modify vllm-ascend source, does not install Python packages, and does not add inference-path backend hooks. The only runtime target is the existing `allgather int32` and `allgather fp16` shim smoke through a selected remote Python environment.

## File Structure

- Modify `tests/collectives/unit/test_vllm_collectives_integration_sources.py`
  - Add source guards for remote Python/conda selection and the smoke launcher Python override.
- Modify `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
  - Add `TILEXR_VLLM_SMOKE_PYTHON` and use it for every rank process.
- Modify `tests/collectives/deploy_and_run_vllm_remote.sh`
  - Add `TILEXR_VLLM_REMOTE_PYTHON`, `TILEXR_VLLM_REMOTE_CONDA_ENV`, and `TILEXR_VLLM_REMOTE_CONDA_SH`.
  - Select one remote Python, report package locations, run torch-npu preflight, and export `TILEXR_VLLM_SMOKE_PYTHON`.
- Modify `tests/collectives/README.md`
  - Document the new Python environment selection variables and current `tt4` validation pattern without defaulting scripts to host-specific paths.

## Task 1: Add Source Guards for Python Environment Selection

**Files:**
- Modify: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Write failing guards for remote Python selection**

Append these tests after `test_remote_script_is_isolated_and_logs_environment` in `tests/collectives/unit/test_vllm_collectives_integration_sources.py`:

```python
def test_remote_script_supports_selected_python_environment() -> None:
    source = read_rel("tests/collectives/deploy_and_run_vllm_remote.sh")
    for token in [
        "TILEXR_VLLM_REMOTE_PYTHON",
        "TILEXR_VLLM_REMOTE_CONDA_ENV",
        "TILEXR_VLLM_REMOTE_CONDA_SH",
        "select_remote_python",
        "selected_python=",
        "dump_selected_python_environment",
        "run_selected_python_preflight",
        "torch.npu.current_stream()",
        "npu_stream",
        "TILEXR_VLLM_SMOKE_PYTHON=\"\\${selected_python}\"",
    ]:
        assert token in source
    assert "python3 -m pip show torch" not in source
    assert "python3 -m pip show torch-npu" not in source


def test_smoke_launcher_supports_python_override() -> None:
    source = read_rel("integrations/vllm_ascend/run_tilexr_collectives_smoke.sh")
    for token in [
        "PYTHON_BIN=\"${TILEXR_VLLM_SMOKE_PYTHON:-python3}\"",
        "command -v \"${PYTHON_BIN}\"",
        "TileXR vllm collectives smoke",
        "\"${PYTHON_BIN}\" \"${SCRIPT_DIR}/smoke_collectives.py\"",
    ]:
        assert token in source
    assert "python3 \"${SCRIPT_DIR}/smoke_collectives.py\"" not in source
```

Update `main()` in the same file so it calls both new tests:

```python
def main() -> None:
    test_vllm_ascend_shim_files_exist()
    test_runtime_uses_tilexr_c_abi_and_not_hccl()
    test_torch_helpers_require_contiguous_npu_tensors()
    test_remote_script_is_isolated_and_logs_environment()
    test_remote_script_supports_selected_python_environment()
    test_smoke_launcher_supports_python_override()
    print("PASS vllm collectives integration source checks")
```

- [ ] **Step 2: Run the source test and verify it fails**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: fails with an assertion for `TILEXR_VLLM_REMOTE_PYTHON` or `PYTHON_BIN="${TILEXR_VLLM_SMOKE_PYTHON:-python3}"`.

- [ ] **Step 3: Commit the failing guard**

Run:

```bash
git add tests/collectives/unit/test_vllm_collectives_integration_sources.py
git commit -m "test: guard vllm remote python selection"
```

Expected: commit succeeds with only the source guard file staged.

## Task 2: Add Python Override to the Smoke Launcher

**Files:**
- Modify: `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Add the selected Python command**

In `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`, insert this block after the `DTYPE` and `TIMEOUT_SEC` assignments:

```bash
PYTHON_BIN="${TILEXR_VLLM_SMOKE_PYTHON:-python3}"
```

Then insert this block after the dtype validation block and before the `export PYTHONPATH=...` line:

```bash
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python command not found: ${PYTHON_BIN}" >&2
  exit 2
fi

echo "TileXR vllm collectives smoke"
echo "  python: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)' 2>/dev/null || printf '%s' "${PYTHON_BIN}")"
echo "  rank_size: ${RANK_SIZE}"
echo "  op: ${OP}"
echo "  dtype: ${DTYPE}"
```

Replace the rank launch command:

```bash
  python3 "${SCRIPT_DIR}/smoke_collectives.py" \
```

with:

```bash
  "${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_collectives.py" \
```

- [ ] **Step 2: Run the source test and verify the launcher guard passes while remote script guard still fails**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected: the smoke launcher assertions pass, and the test still fails on remote script Python selection tokens.

- [ ] **Step 3: Run shell syntax check for the launcher**

Run:

```bash
bash -n integrations/vllm_ascend/run_tilexr_collectives_smoke.sh
```

Expected: no output and exit code 0.

- [ ] **Step 4: Commit the launcher change**

Run:

```bash
git add integrations/vllm_ascend/run_tilexr_collectives_smoke.sh
git commit -m "feat: allow vllm smoke python override"
```

Expected: commit succeeds with only the launcher staged.

## Task 3: Select Remote Python in the Deployment Script

**Files:**
- Modify: `tests/collectives/deploy_and_run_vllm_remote.sh`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Add local environment variables**

In `tests/collectives/deploy_and_run_vllm_remote.sh`, add these assignments after `REMOTE_CMAKE_CCE_COMPILER=...`:

```bash
REMOTE_PYTHON="${TILEXR_VLLM_REMOTE_PYTHON:-}"
REMOTE_CONDA_ENV="${TILEXR_VLLM_REMOTE_CONDA_ENV:-}"
REMOTE_CONDA_SH="${TILEXR_VLLM_REMOTE_CONDA_SH:-/home/miniconda3/etc/profile.d/conda.sh}"
```

Add these status lines after `echo "  commit: ${commit}"`:

```bash
if [[ -n "${REMOTE_PYTHON}" ]]; then
  echo "  remote python: ${REMOTE_PYTHON}"
fi
if [[ -n "${REMOTE_CONDA_ENV}" ]]; then
  echo "  remote conda env: ${REMOTE_CONDA_ENV}"
fi
```

- [ ] **Step 2: Pass Python selection variables into the remote script**

Inside the `remote_script=$(cat <<EOF` block, add these quoted variable assignments after `remote_cmake_cce_compiler=...`:

```bash
remote_python=$(printf '%q' "${REMOTE_PYTHON}")
remote_conda_env=$(printf '%q' "${REMOTE_CONDA_ENV}")
remote_conda_sh=$(printf '%q' "${REMOTE_CONDA_SH}")
```

- [ ] **Step 3: Add remote Python helper functions**

Inside the remote script body, immediately after `cd $(printf '%q' "${REMOTE_REPO}")`, insert this heredoc-safe block exactly as shown. The backslashes before `$` are required because the block lives inside the existing unquoted `remote_script=$(cat <<EOF` heredoc:

```bash
select_remote_python() {
  selected_python="python3"
  if [[ -n "\${remote_python}" ]]; then
    if [[ -n "\${remote_conda_env}" ]]; then
      echo "INFO: TILEXR_VLLM_REMOTE_PYTHON is set; ignoring TILEXR_VLLM_REMOTE_CONDA_ENV=\${remote_conda_env}"
    fi
    selected_python="\${remote_python}"
  elif [[ -n "\${remote_conda_env}" ]]; then
    if [[ ! -f "\${remote_conda_sh}" ]]; then
      echo "ERROR: conda activation script not found: \${remote_conda_sh}" >&2
      return 2
    fi
    set +u
    source "\${remote_conda_sh}"
    conda activate "\${remote_conda_env}"
    set -u
    selected_python="\$(command -v python)"
  fi
  if ! command -v "\${selected_python}" >/dev/null 2>&1; then
    echo "ERROR: selected Python not found: \${selected_python}" >&2
    return 2
  fi
  selected_python="\$(command -v "\${selected_python}")"
  export TILEXR_VLLM_SELECTED_PYTHON="\${selected_python}"
}

dump_selected_python_environment() {
  echo "Selected Python: \${selected_python}"
  "\${selected_python}" --version
  "\${selected_python}" - <<'PY'
import importlib.util
import sys

print("Python executable:", sys.executable)
print("Python version:", sys.version.replace("\n", " "))
print("sys.path prefix:", sys.path[:5])
for name in ["torch", "torch_npu", "vllm", "vllm_ascend"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {spec.origin if spec else 'MISSING'}")
PY
  "\${selected_python}" -m pip show torch || true
  "\${selected_python}" -m pip show torch-npu || true
  "\${selected_python}" -m pip show vllm || true
  "\${selected_python}" -m pip show vllm-ascend || true
}

run_selected_python_preflight() {
  "\${selected_python}" - <<'PY'
import sys

missing = []
for name in ["torch", "torch_npu"]:
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {type(exc).__name__}: {exc}")
if missing:
    raise SystemExit("missing required Python packages: " + "; ".join(missing))

import torch
import torch_npu  # noqa: F401

print("torch:", getattr(torch, "__version__", "unknown"))
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
if not torch.npu.is_available():
    raise SystemExit("torch.npu.is_available() is false")
device_count = torch.npu.device_count()
print("npu device_count:", device_count)
if device_count < 2:
    raise SystemExit(f"need at least 2 visible NPUs, got {device_count}")
torch.npu.set_device(0)
stream = torch.npu.current_stream()
stream_fields = {
    "npu_stream": getattr(stream, "npu_stream", None),
    "stream": getattr(stream, "stream", None),
}
print("current stream type:", type(stream))
print("current stream fields:", stream_fields)
if stream_fields["npu_stream"] is None and stream_fields["stream"] is None:
    raise SystemExit("torch.npu.current_stream() exposes neither npu_stream nor stream")
PY
}
```

- [ ] **Step 4: Replace hard-coded Python probes with selected Python probes**

In the remote script body, replace these lines:

```bash
  python3 --version || true
  cmake --version 2>/dev/null | sed -n '1p' || true
  gcc --version 2>/dev/null | sed -n '1p' || true
  g++ --version 2>/dev/null | sed -n '1p' || true
  python3 -m pip show torch || true
  python3 -m pip show torch-npu || true
  python3 -m pip show vllm || true
  python3 -m pip show vllm-ascend || true
```

with:

```bash
  select_remote_python
  dump_selected_python_environment
  cmake --version 2>/dev/null | sed -n '1p' || true
  gcc --version 2>/dev/null | sed -n '1p' || true
  g++ --version 2>/dev/null | sed -n '1p' || true
```

- [ ] **Step 5: Run torch-npu preflight after CANN environment setup**

After the existing block:

```bash
  set +u
  source scripts/common_env.sh
  set -u
```

insert:

```bash
  run_selected_python_preflight
```

- [ ] **Step 6: Thread selected Python into the smoke launcher**

Replace the two smoke launcher commands:

```bash
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather int32
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather fp16
```

with this heredoc-safe replacement in the actual file:

```bash
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 TILEXR_VLLM_SMOKE_PYTHON="\${selected_python}" ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather int32
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 TILEXR_VLLM_SMOKE_PYTHON="\${selected_python}" ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather fp16
```

- [ ] **Step 7: Print the remote log path on success and failure**

At the end of `tests/collectives/deploy_and_run_vllm_remote.sh`, replace:

```bash
ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_script}")"

echo "Remote validation log: ${REMOTE}:${REMOTE_LOG}"
```

with:

```bash
set +e
ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_script}")"
ssh_rc="$?"
set -e

echo "Remote validation log: ${REMOTE}:${REMOTE_LOG}"
exit "${ssh_rc}"
```

- [ ] **Step 8: Run source and shell checks**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
bash -n tests/collectives/deploy_and_run_vllm_remote.sh
bash -n integrations/vllm_ascend/run_tilexr_collectives_smoke.sh
```

Expected:

```text
PASS vllm collectives integration source checks
```

The two `bash -n` commands produce no output and exit 0.

- [ ] **Step 9: Commit the deployment script change**

Run:

```bash
git add tests/collectives/deploy_and_run_vllm_remote.sh
git commit -m "feat: select remote python for vllm validation"
```

Expected: commit succeeds with only the deployment script staged.

## Task 4: Document the Phase 2 Remote Python Controls

**Files:**
- Modify: `tests/collectives/README.md`
- Test: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`

- [ ] **Step 1: Update remote validation docs**

Replace lines 141-164 in `tests/collectives/README.md` with:

````markdown
If the remote login shell does not already expose CANN Toolkit or the CCE compiler, pass the remote paths
explicitly:

```bash
TILEXR_VLLM_REMOTE_ASCEND_HOME_PATH=/path/to/remote/ascend-toolkit \
TILEXR_VLLM_REMOTE_ASCEND_DRIVER_PATH=/path/to/remote/ascend-driver \
TILEXR_VLLM_REMOTE_CMAKE_CCE_COMPILER=/path/to/remote/ascend-toolkit/bin/ccec \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

If the remote default Python does not provide `torch_npu`, select a Python environment explicitly. A direct Python
path takes precedence over a conda environment name:

```bash
TILEXR_VLLM_REMOTE_PYTHON=/path/to/remote/python \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

```bash
TILEXR_VLLM_REMOTE_CONDA_ENV=tt4 \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

Set `TILEXR_VLLM_REMOTE_CONDA_SH` when the remote conda activation script is not available at
`/home/miniconda3/etc/profile.d/conda.sh`.

The script syncs the current TileXR commit, initializes submodules from local worktrees, dumps the NPU/CANN/Python
environment, builds `tile-comm` and `tilexr-collectives`, runs the standalone 2-rank AllGather correctness check, and
runs the Python torch-npu shim AllGather smoke for `int32` and `fp16`.

Expected success lines include:

```text
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=fp16
```

If `torch` or `torch-npu` is missing in the selected Python environment, the preflight fails before the multi-rank
shim smoke. Missing `vllm` or `vllm-ascend` is recorded in the environment dump but does not fail this shim phase.
````

- [ ] **Step 2: Run docs/source guard**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py
```

Expected:

```text
PASS vllm collectives integration source checks
```

- [ ] **Step 3: Commit the docs change**

Run:

```bash
git add tests/collectives/README.md
git commit -m "docs: document vllm remote python selection"
```

Expected: commit succeeds with only the README staged.

## Task 5: Run Local Verification

**Files:**
- Verify: `tests/collectives/unit/test_vllm_collectives_integration_sources.py`
- Verify: `tests/collectives/deploy_and_run_vllm_remote.sh`
- Verify: `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`
- Verify: `integrations/vllm_ascend/smoke_collectives.py`
- Verify: `integrations/vllm_ascend/tilexr_collectives/runtime.py`
- Verify: `integrations/vllm_ascend/tilexr_collectives/torch_collectives.py`
- Verify: `integrations/vllm_ascend/tilexr_collectives/__init__.py`

- [ ] **Step 1: Run the complete local Phase 2 verification command**

Run:

```bash
python3 tests/collectives/unit/test_vllm_collectives_integration_sources.py && \
python3 -m pytest -q tests/collectives/unit/test_vllm_collectives_integration_sources.py && \
bash -n tests/collectives/deploy_and_run_vllm_remote.sh && \
bash -n integrations/vllm_ascend/run_tilexr_collectives_smoke.sh && \
python3 -m py_compile \
  integrations/vllm_ascend/tilexr_collectives/runtime.py \
  integrations/vllm_ascend/tilexr_collectives/torch_collectives.py \
  integrations/vllm_ascend/tilexr_collectives/__init__.py \
  integrations/vllm_ascend/smoke_collectives.py && \
git diff --check
```

Expected:

```text
PASS vllm collectives integration source checks
```

`pytest` reports the source tests pass, both shell syntax checks exit 0, Python compilation exits 0, and `git diff --check` exits 0.

- [ ] **Step 2: Confirm no uncommitted local changes**

Run:

```bash
git status --short
```

Expected: no output. If there are uncommitted changes, inspect them and commit only changes produced by this plan.

## Task 6: Run Remote `blue` Validation with the Selected Conda Environment

**Files:**
- Verify: `tests/collectives/deploy_and_run_vllm_remote.sh`
- Verify: `integrations/vllm_ascend/run_tilexr_collectives_smoke.sh`

- [ ] **Step 1: Confirm required remote path variables are present in the local shell**

Run:

```bash
: "${TILEXR_BLUE_REMOTE_BASE:?set TILEXR_BLUE_REMOTE_BASE to a writable remote scratch parent}"
: "${TILEXR_BLUE_ASCEND_HOME_PATH:?set TILEXR_BLUE_ASCEND_HOME_PATH to the remote CANN 9.1.0 toolkit}"
: "${TILEXR_BLUE_ASCEND_DRIVER_PATH:?set TILEXR_BLUE_ASCEND_DRIVER_PATH to the remote driver shim or driver path}"
: "${TILEXR_BLUE_CMAKE_CCE_COMPILER:?set TILEXR_BLUE_CMAKE_CCE_COMPILER to the remote ccec compiler}"
```

Expected: no output. These variables are intentionally not committed to repository docs or scripts.

- [ ] **Step 2: Run remote validation on `blue` with `tt4`**

Run:

```bash
TILEXR_VLLM_REMOTE=blue \
TILEXR_VLLM_REMOTE_BASE="${TILEXR_BLUE_REMOTE_BASE%/}/tilexr_vllm_collectives_$(date +%Y%m%d_%H%M%S)" \
TILEXR_VLLM_REMOTE_ASCEND_HOME_PATH="${TILEXR_BLUE_ASCEND_HOME_PATH}" \
TILEXR_VLLM_REMOTE_ASCEND_DRIVER_PATH="${TILEXR_BLUE_ASCEND_DRIVER_PATH}" \
TILEXR_VLLM_REMOTE_CMAKE_CCE_COMPILER="${TILEXR_BLUE_CMAKE_CCE_COMPILER}" \
TILEXR_VLLM_REMOTE_CONDA_ENV=tt4 \
bash tests/collectives/deploy_and_run_vllm_remote.sh
```

Expected remote log output includes:

```text
Selected Python:
torch:
torch_npu:
npu device_count: 8
current stream fields:
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=int32
PASS TileXR vllm collectives smoke rank_size=2 op=allgather dtype=fp16
Remote validation log: blue:
```

- [ ] **Step 3: If remote validation fails, preserve the failure evidence before fixing**

Run:

```bash
printf '%s\n' "Remote failure log path from deploy output:"
```

Expected: the deploy output has already printed `Remote validation log: blue:<path>`. Inspect that remote log and rank logs before changing code. Do not delete the scratch directory.

- [ ] **Step 4: Record final verification status**

Run:

```bash
git log --oneline -n 8
git status --short
```

Expected: `git status --short` has no output. The latest commits include the source guards, smoke launcher override, remote Python selection, docs update, and this implementation plan.
