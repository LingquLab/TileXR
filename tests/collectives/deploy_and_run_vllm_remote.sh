#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TILEXR_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

REMOTE="${TILEXR_VLLM_REMOTE:?set TILEXR_VLLM_REMOTE to the SSH target for remote vllm collectives validation}"
REMOTE_BASE="${TILEXR_VLLM_REMOTE_BASE:?set TILEXR_VLLM_REMOTE_BASE to a scratch directory on the remote host}"
REMOTE_REPO="${REMOTE_BASE}/TileXR"
REMOTE_LOG="${REMOTE_BASE}/logs/deploy.log"
REMOTE_ASCEND_HOME_PATH="${TILEXR_VLLM_REMOTE_ASCEND_HOME_PATH:-}"
REMOTE_ASCEND_DRIVER_PATH="${TILEXR_VLLM_REMOTE_ASCEND_DRIVER_PATH:-}"
REMOTE_CMAKE_CCE_COMPILER="${TILEXR_VLLM_REMOTE_CMAKE_CCE_COMPILER:-}"
REMOTE_PYTHON="${TILEXR_VLLM_REMOTE_PYTHON:-}"
REMOTE_CONDA_ENV="${TILEXR_VLLM_REMOTE_CONDA_ENV:-}"
REMOTE_CONDA_SH="${TILEXR_VLLM_REMOTE_CONDA_SH:-/home/miniconda3/etc/profile.d/conda.sh}"

branch="$(git -C "${TILEXR_ROOT}" rev-parse --abbrev-ref HEAD)"
commit="$(git -C "${TILEXR_ROOT}" rev-parse HEAD)"
staging_dir="$(mktemp -d "${TMPDIR:-/tmp}/tilexr_vllm_collectives.XXXXXX")"
trap 'rm -rf "${staging_dir}"' EXIT
staging_repo="${staging_dir}/TileXR"

echo "Deploying TileXR vllm collectives validation"
echo "  remote: ${REMOTE}"
echo "  remote base: ${REMOTE_BASE}"
echo "  branch: ${branch}"
echo "  commit: ${commit}"
if [[ -n "${REMOTE_PYTHON}" ]]; then
  echo "  remote python: ${REMOTE_PYTHON}"
fi
if [[ -n "${REMOTE_CONDA_ENV}" ]]; then
  echo "  remote conda env: ${REMOTE_CONDA_ENV}"
fi

git clone --no-hardlinks --no-checkout "${TILEXR_ROOT}" "${staging_repo}"
git -C "${staging_repo}" checkout --detach "${commit}"

sync_local_submodule() {
  local rel_path="$1"
  local src="${TILEXR_ROOT}/${rel_path}"
  local dst="${staging_repo}/${rel_path}"
  if [[ ! -d "${src}" ]]; then
    echo "ERROR: required local submodule is missing: ${rel_path}" >&2
    echo "Initialize local submodules before running this script." >&2
    exit 1
  fi
  mkdir -p "${dst}"
  rsync -a --delete --exclude='.git' "${src}/" "${dst}/"
}

sync_local_submodule "3rdparty/hcomm"
sync_local_submodule "3rdparty/ops-transformer"
sync_local_submodule "3rdparty/spdlog"

remote_prepare=$(cat <<EOF
set -euo pipefail
remote_base=$(printf '%q' "${REMOTE_BASE}")
remote_repo=$(printf '%q' "${REMOTE_REPO}")
case "\${remote_repo}" in
  "\${remote_base}"/TileXR)
    rm -rf -- "\${remote_repo}"
    mkdir -p -- "\${remote_repo}"
    mkdir -p -- "\${remote_base}/logs" "\${remote_base}/artifacts"
    ;;
  *)
    echo "Refusing to clean unexpected remote repo: \${remote_repo}" >&2
    exit 2
    ;;
esac
EOF
)

ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_prepare}")"

rsync -a --delete \
  --exclude='.worktrees' \
  --exclude='build' \
  --exclude='build_*' \
  --exclude='build-*' \
  --exclude='install' \
  --exclude='run' \
  --exclude='env/temp' \
  "${staging_repo}/" "${REMOTE}:${REMOTE_REPO}/"

remote_script=$(cat <<EOF
set -euo pipefail
remote_ascend_home_path=$(printf '%q' "${REMOTE_ASCEND_HOME_PATH}")
remote_ascend_driver_path=$(printf '%q' "${REMOTE_ASCEND_DRIVER_PATH}")
remote_cmake_cce_compiler=$(printf '%q' "${REMOTE_CMAKE_CCE_COMPILER}")
remote_python=$(printf '%q' "${REMOTE_PYTHON}")
remote_conda_env=$(printf '%q' "${REMOTE_CONDA_ENV}")
remote_conda_sh=$(printf '%q' "${REMOTE_CONDA_SH}")
cd $(printf '%q' "${REMOTE_REPO}")
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

{
  echo "Remote branch source: ${branch}"
  echo "Remote commit source: ${commit}"
  echo "Remote host: \$(hostname)"
  echo "Remote user: \$(whoami)"
  command -v npu-smi || true
  npu-smi info || true
  select_remote_python
  dump_selected_python_environment
  cmake --version 2>/dev/null | sed -n '1p' || true
  gcc --version 2>/dev/null | sed -n '1p' || true
  g++ --version 2>/dev/null | sed -n '1p' || true
  git submodule status --recursive || true
  if [[ -n "\${remote_ascend_home_path}" ]]; then
    export ASCEND_HOME_PATH="\${remote_ascend_home_path}"
    if [[ -f "\${ASCEND_HOME_PATH}/set_env.sh" ]]; then
      set +u
      source "\${ASCEND_HOME_PATH}/set_env.sh"
      set -u
    fi
  fi
  if [[ -n "\${remote_ascend_driver_path}" ]]; then
    export ASCEND_DRIVER_PATH="\${remote_ascend_driver_path}"
  fi
  if [[ -n "\${remote_cmake_cce_compiler}" ]]; then
    export CMAKE_CCE_COMPILER="\${remote_cmake_cce_compiler}"
  fi
  set +u
  source scripts/common_env.sh
  set -u
  run_selected_python_preflight
  cmake_args=(
    -DCMAKE_INSTALL_PREFIX=install
    -DTILEXR_BUILD_COLLECTIVES=ON
    -DTILEXR_BUILD_TESTS=ON
    -DBUILD_TESTING=ON
  )
  if [[ -n "\${remote_cmake_cce_compiler}" ]]; then
    cmake_args+=("-DCMAKE_CCE_COMPILER=\${remote_cmake_cce_compiler}")
  fi
  if [[ -n "\${remote_ascend_driver_path}" ]]; then
    cmake_args+=("-DASCEND_DRIVER_PATH=\${remote_ascend_driver_path}")
  fi
  cmake -S . -B build-vllm-collectives \
    "\${cmake_args[@]}"
  cmake --build build-vllm-collectives --target install test_tilexr_collectives_correctness tilexr_collective_perf -j"\$(nproc)"
  ctest --test-dir build-vllm-collectives --output-on-failure
  (
    cd tests/collectives
    TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=600 ./run_collectives_correctness.sh 2 16 0 ../../build-vllm-collectives/tests/collectives allgather
  )
  (
    cd integrations/vllm_ascend
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 TILEXR_VLLM_SMOKE_PYTHON="\${selected_python}" ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather int32
    TILEXR_VLLM_SMOKE_TIMEOUT_SEC=600 TILEXR_VLLM_SMOKE_PYTHON="\${selected_python}" ./run_tilexr_collectives_smoke.sh 2 16 0 ../../install allgather fp16
  )
} 2>&1 | tee $(printf '%q' "${REMOTE_LOG}")
EOF
)

set +e
ssh "${REMOTE}" "bash -lc $(printf '%q' "${remote_script}")"
ssh_rc="$?"
set -e

echo "Remote validation log: ${REMOTE}:${REMOTE_LOG}"
exit "${ssh_rc}"
