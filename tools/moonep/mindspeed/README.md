# MindSpeed validation tools

This directory contains reusable tools developed while validating the TileXR MoonEP
backend with MindSpeed on a single eight-device Ascend 950 node.

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
