# MindSpeed validation tools

This directory contains reusable tools developed while validating the TileXR MoonEP
backend with MindSpeed on Ascend 950 nodes.

- `tilexr_mindspeed_adapter.py`: MindSpeed backend adapter that keeps communication
  buffers owned by TileXR.
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

The runners use the current repository by default and write results below
`run/moonep/mindspeed`. Override paths with `TILEXR_HOME`,
`TILEXR_INSTALL_PREFIX`, `TILEXR_CANN_ENV`, `TILEXR_CONDA_SH`,
`TILEXR_CONDA_ENV`, and `TILEXR_MOONEP_NATIVE_ENV`.

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

# Enable one profiler window. The diagnostic stage barrier is off by default.
bash tools/moonep/mindspeed/run_model.sh --mode multi --profile
bash tools/moonep/mindspeed/run_model.sh --mode multi --stage-barrier
```

SSH authentication is owned by the system client. Configure a key, agent,
terminal password prompt, or askpass outside this tool. The runner does not copy
the repository: deploy identical code and builds to every node with Mutagen
before starting a multi-node run.
