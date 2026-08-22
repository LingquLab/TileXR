# MindSpeed validation tools

This directory contains reusable tools developed while validating the TileXR MoonEP
backend with MindSpeed on Ascend 950 nodes.

- `tilexr_mindspeed_adapter.py`: MindSpeed backend adapter that keeps communication
  buffers owned by TileXR.
- `mindspeed_stage_barrier.py`: shared optional world barrier immediately before
  native or TileXR Dispatch and Combine calls, including backward calls.
- `mindspeed_external_comm_owner.patch`: MindSpeed patch that lets an external backend
  own the communication runtime and buffer.
- `preflight_adapter.sh`: validates the adapter and the patched MindSpeed ownership
  contract before an NPU model run.
- `grouped_urma_dispatch_oracle.py`: exact repeated grouped-URMA Dispatch/Combine
  oracle at the production `S=4096`, `K=8`, EP8 route shape.
- `run_case15_32.sh`: 32-iteration case 15 runner.
- `run_grouped_oracle.sh`: eight-rank grouped-URMA oracle runner.
- `run_model.sh`: single entry point for local single-node runs and controller-side
  multi-node SSH orchestration.
- `run_model_node.sh`: non-interactive per-node runner based on the validated
  4K/8P model command.
- `probe_idle.sh`: bounded NPU idle gate with a conservative device-node fallback
  for hosts where a concurrent management query blocks full `npu-smi info`.

The runners use the current repository by default and write results below
`run/moonep/mindspeed`. Override paths with `TILEXR_HOME`,
`TILEXR_INSTALL_PREFIX`, `TILEXR_CANN_ENV`, `TILEXR_CONDA_SH`,
`TILEXR_CONDA_ENV`, and `TILEXR_MOONEP_NATIVE_ENV`.

Every multi-node run uses one global expert-parallel group spanning all ranks for
both backends. The model keeps at least 32 experts and rounds that count up to a
multiple of the global rank count; for example, 16 nodes x 8 devices runs with
EP128 and 128 experts, while a single-node run remains EP8 with 32 experts.
Multi-node runs prefer `data0.3001` for HCCL and Gloo on the validated cluster;
set `MODEL_RUNNER_SOCKET_IFNAME` to override that interface on other deployments.

Before using the adapter, apply the MindSpeed ownership patch and run its
preflight:

```bash
git -C "${MINDSPEED_HOME}" apply \
    "${TILEXR_HOME}/tools/moonep/mindspeed/mindspeed_external_comm_owner.patch"
MINDSPEED_HOME="${MINDSPEED_HOME}" \
    bash "${TILEXR_HOME}/tools/moonep/mindspeed/preflight_adapter.sh"
```

The preflight installs `tilexr_mindspeed_adapter.py` into the checkout named by
`MINDSPEED_HOME`; use a disposable or task-owned MindSpeed checkout.

## HCCL rank-table generator

`generate_rank_table.py` builds the HCCL v1.2 table used by the validated
Ascend 950 multi-node topology. It queries only the selected interface with
`ip -j` and reads each node's HCCL RootInfo JSON; it does not start an NPU
process or run a model or HCCL Test. The validated 16-node cluster has
`/etc/hccl_rootinfo.json.bak` but no `/etc/hccl_rootinfo.json`; deployments
with the canonical file should pass `--rootinfo-path /etc/hccl_rootinfo.json`.

For the 16-node, 128-rank cluster:

```bash
python3 tools/moonep/mindspeed/generate_rank_table.py \
    --hosts /path/to/hosts.txt \
    --super-pod-ids 5,15 \
    --servers-per-super-pod 8 \
    --devices-per-server 8 \
    --interface data0.3001 \
    --rootinfo-path /etc/hccl_rootinfo.json.bak \
    --output run/moonep/mindspeed/config/rank_table_128.json
```

Host order defines rank order. Blank lines and lines beginning with `#` are
ignored. An inventory entry may be either `HOST` or `HOST:CREDENTIAL`; the
credential suffix is discarded and never stored or printed. Authentication is
owned by the system SSH client, so configure its agent, askpass, or terminal
prompt outside this tool. Add system SSH options with repeated
`--ssh-option KEY=VALUE` arguments.

The generator requires one unambiguous `0/*` EID from `net_layer=1` for every
device. The `super_device_id` values use the encoding measured on this B131
Ascend 950 deployment:

```text
super_device_id = super_pod_id * 4194304 + device_id * 262145
```

Confirm that encoding before using the tool on another hardware generation.
Generated rank tables use deterministic UTF-8/LF formatting so hashes match
across Linux and Windows, and belong under the ignored `run/` tree. Validate
an existing file without connecting to any host:

```bash
python3 tools/moonep/mindspeed/generate_rank_table.py \
    --check run/moonep/mindspeed/config/rank_table_128.json
```

## Full model runner

The first invocation asks for the node list and deployed paths, then writes the
answers to the ignored runtime file
`run/moonep/mindspeed/model_runner.env`. Later invocations reuse it without
prompting. Use `--configure` only when the saved deployment information must be
replaced. The file contains paths and hostnames, never SSH passwords.

Run a single-node TileXR model locally:

```bash
bash tools/moonep/mindspeed/run_model.sh \
    --mode single --backend tilexr
```

From the controller host, start all configured nodes concurrently over system
SSH:

```bash
bash tools/moonep/mindspeed/run_model.sh \
    --mode multi --backend tilexr
```

The controller maps node ranks in the saved host order; the first host is rank 0
and supplies `MASTER_ADDR`. It verifies that `run_model_node.sh` exists on every
node before starting any model process, records one controller log per node, and
stops peer process groups if a node fails or the controller is interrupted.
It also waits for three consecutive all-node idle samples before launch; use
`--idle-wait` to bound that wait on a shared test machine.

Useful optional modes:

```bash
# Inspect the exact local or SSH commands without launching.
bash tools/moonep/mindspeed/run_model.sh --mode multi --dry-run

# Replace the cached answers.
bash tools/moonep/mindspeed/run_model.sh --configure --mode multi --dry-run

# Enable one profiler window. The stage barrier is off by default and applies to
# both native MoonEP and TileXR MoonEP when requested.
bash tools/moonep/mindspeed/run_model.sh --mode multi --profile
bash tools/moonep/mindspeed/run_model.sh --mode multi --stage-barrier
```

SSH authentication is owned by the system client. Configure a key, agent,
terminal password prompt, or askpass outside this tool. The runner does not copy
the repository: deploy identical code and builds to every node with Mutagen
before starting a multi-node run.

## Model input replay cache

`scripts/run_moonep.sh --model-replay` captures the exact compressed TopK routes
from a TileXR MindSpeed model iteration, caches them with their execution
provenance, then replays the 55 MoonEP calls in model order. The controller prints
the cached model timings and the cascade timings together. This mode is
shape-driven; provide `S`, `K`, hidden size, expert parallel size, and rank size:

```bash
bash scripts/run_moonep.sh \
    --mode benchmark --rank-size 8 \
    --model-replay \
    --s 4096 --k 8 --hidden-size 7168 --ep 8
```

When stdin is a terminal, omitted shape values are prompted. A non-interactive
run fails before CANN or device initialization unless all five values are
present. `--model-replay-from` selects the first allowed input source. Its
default, `cache`, tries runtime cache, compatible checked-in meta, then one model
capture. `meta` skips runtime cache and tries meta before model capture; `model`
skips both persisted sources and captures new route and performance data. Use
`--model-replay-cache-dir` to relocate the ignored runtime cache and
`--model-runner-config` only when a specific saved `model_runner.env` must be
used.

The runner creates the per-run replay files automatically:

- `model_replay_case.json` in the selected output directory;
- `model_replay_result.env` in the selected output directory;
- route replay and model performance cache generations under
  `run/moonep/model_replay_cache`;
- for single-node replay only, a temporary `generated/model_runner.env` when no
  configured model runner env is found.

After a model capture it also writes a compact source bundle under
`tools/moonep/model_replay_meta/<case-id>/`:

```text
meta.json
routes.u8.zst
performance.json
```

The route blob stores globally deduplicated, range-checked uint8 expert IDs with
standard zstd level 3 compression. `meta.json` retains every rank/call reference,
source call, plan metadata, tensor descriptor, checksum, shape, and sanitized
provenance. `performance.json` retains all-rank 55-stage data. Override the source
root with `TILEXR_MOONEP_MODEL_REPLAY_META_ROOT`. Meta materialization always creates
a validated runtime generation under `run/moonep/model_replay_cache`; the cascade
consumer never reads checked-in files directly.

When the synchronized source replica intentionally omits `.git`, set
`TILEXR_MODEL_REPLAY_TILEXR_GIT_SHA` to the 7-64 digit hexadecimal source revision.
The runner rejects other values so checked-in provenance cannot silently use an
arbitrary label.

For single-node replay, the script generates a per-run `model_runner.env` under
the selected output directory unless `--model-runner-config` or
`TILEXR_MODEL_RUNNER_CONFIG` is set explicitly. This keeps cache refreshes and
misses tied to the current `TILEXR_INSTALL_PREFIX` instead of silently reusing a
stale persistent model-runner config. Multi-node replay still requires an
explicit preconfigured model runner env because node order and remote deployment
paths cannot be inferred safely.

For repeatable server-side comparison, add
`--model-replay-stage-summary-only`. The replay still runs the cascade benchmark,
but the terminal output is limited to the model-vs-replay stage table instead of
the longer per-operator report. A cache hit does not launch the model again:

```bash
base=${TILEXR_HOME:-$PWD}
install_prefix=${TILEXR_INSTALL_PREFIX:-$base/install-clean-b131}
if [[ ! -d "${install_prefix}" ]]; then
    install_prefix=$base/install
fi
out=$base/run/moonep/model-replay-$(date +%Y%m%d-%H%M%S)
cd "$base"

TILEXR_INSTALL_PREFIX=$install_prefix \
TILEXR_MOONEP_CONDA_ENV=ai_moe_test \
TILEXR_MOONEP_OUTPUT_DIR=$out \
bash scripts/run_moonep.sh \
    --mode benchmark --rank-size 8 \
    --auto-build-install \
    --model-replay --model-replay-stage-summary-only \
    --s 4096 --k 8 --hidden-size 7168 --ep 8 \
    --warmup 5 --iterations 20
```

Add `--model-replay-from model` to force recapturing and replacing the model
route, compact meta, and runtime cache before replaying the same shape. Use
`--model-replay-from meta` to ignore a suspect runtime cache while still avoiding
a model run when compatible meta exists. `--auto-build-install` is optional;
when the selected `TILEXR_INSTALL_PREFIX` already exists it is skipped, and when
it is missing the script configures, builds, and installs the MoonEP runtime into
that prefix before launching the benchmark. Override its build directory and
parallelism with `TILEXR_MOONEP_BUILD_DIR` and `TILEXR_MOONEP_BUILD_JOBS`.

The cache identity covers the shape, 55-call contract, source and adapter hashes,
model stack, backend/kernel selection, CANN, driver, SoC, topology, and device
mapping. Each capture is published as an immutable generation only after every
rank record and checksum validates. Interrupted, partial, stale, or mismatched
generations are not consumed. Full hidden and weight tensors are intentionally
regenerated from cached deterministic descriptors and seeds instead of being
stored as multi-gigabyte artifacts.

Route and performance sampling share one model launch. Route capture skips 60
forward calls, which selects the ten forward calls in profiler step 6.
Profiler-off performance capture skips 330 operators, or six complete 55-operator
iterations, before timing the next complete iteration with NPU Events. Framework
profiling is the default comparison source; pass `--no-model-replay-profile` for
the lightweight NPU Event path. Keep profiler-on and profiler-off results labeled
separately.
Model replay also enables the Dispatch/Combine stage barrier by default. Project
validation and performance comparisons keep both the framework profiler and
stage barrier enabled on the model and replay sides; explicit off modes are
diagnostic A/Bs, not the performance baseline.

For multi-node runs, node 0 is the cache publisher and distributes the completed
generation with `rsync -a --protect-args`. Followers wait for that validated
generation. Replay rank artifacts are collected back to node 0 with the same
non-destructive rsync options before aggregation; no synchronization path uses
delete semantics.
