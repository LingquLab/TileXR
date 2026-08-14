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
