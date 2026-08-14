#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tilexr_home=${TILEXR_HOME:-$(cd "${script_dir}/../../.." && pwd)}
mindspeed_home=${MINDSPEED_HOME:?MINDSPEED_HOME must point to the MindSpeed checkout}
adapter_dir=${mindspeed_home}/mindspeed/core/transformer/moe
adapter=${adapter_dir}/tilexr_mindspeed_adapter.py
stage_barrier=${adapter_dir}/mindspeed_stage_barrier.py
install_prefix=${TILEXR_INSTALL_PREFIX:-${tilexr_home}/install}

if [[ ! -d "${adapter_dir}" ]]; then
    printf 'MindSpeed MoE package directory not found: %s\n' "${adapter_dir}" >&2
    exit 1
fi

install -m 0644 "${script_dir}/tilexr_mindspeed_adapter.py" "${adapter}"
install -m 0644 "${script_dir}/mindspeed_stage_barrier.py" "${stage_barrier}"
if grep -Eq 'from moonep import Buffer|aclshmem' "${adapter}"; then
    echo "forbidden SHMEM Buffer dependency in TileXR adapter" >&2
    exit 1
fi

export TILEXR_INSTALL_PREFIX=${install_prefix}
export PYTHONPATH="${mindspeed_home}:${tilexr_home}/integrations/moonep_torch:${tilexr_home}${PYTHONPATH:+:${PYTHONPATH}}"

python - <<'PY'
import inspect
from types import SimpleNamespace

import torch

from mindspeed.core.transformer.moe import tilexr_mindspeed_adapter as adapter
from mindspeed.core.transformer.moe import mindspeed_stage_barrier
from mindspeed.core.transformer.moe.moonep_model_arena import MoonEPModelArenaLayout

assert adapter.MindSpeedTileXRBuffer.__mro__[1].__module__ == "tilexr_moonep.compat"
assert callable(mindspeed_stage_barrier.create_native_barrier_backend)
assert "hidden_buffer" in inspect.signature(adapter.MindSpeedTileXRBuffer.dispatch).parameters
assert "hidden_buffer" in inspect.signature(adapter.MindSpeedTileXRBuffer.combine).parameters
assert adapter.MOONEP_NATIVE_NPU_CAPABILITY in (
    adapter.create_tilexr_moonep_backend.__mindspeed_capabilities__
)
assert (
    adapter.create_tilexr_moonep_backend.__mindspeed_external_communication_owner__
    is True
)

layout = MoonEPModelArenaLayout(S=8, H=64, K=2, E=8, R=2, B=4, F=128)
default_requests = layout.owned_requests()
tilexr_requests = layout.owned_requests(include_communication_buffer=False)
assert any(request.role == "communication" for request in default_requests)
assert all(request.role != "communication" for request in tilexr_requests)
assert [request for request in default_requests if request.role != "communication"] == list(
    tilexr_requests
)
assert adapter.upstream_moonep.Buffer._tilexr_mindspeed_guard is True

owner = object()
plan = object()
probe = SimpleNamespace(_plan_owner_token=owner, _dispatch_generation=0)
hidden = torch.empty((4, 8))
route = torch.empty((4,))
hidden_alias, route_alias = adapter.MindSpeedTileXRBuffer._bind_zero_copy_views(
    probe, plan, hidden, route
)
assert hidden_alias is not hidden and hidden_alias.data_ptr() == hidden.data_ptr()
assert route_alias is not route and route_alias.data_ptr() == route.data_ptr()
assert hidden_alias._moonep_buffer_owner is owner
assert hidden_alias._moonep_dispatch_plan is plan
assert hidden_alias._moonep_dispatch_generation == 1
assert probe._dispatch_generation == 1

try:
    adapter.upstream_moonep.Buffer.__init__(object())
except RuntimeError as exc:
    assert "forbids construction" in str(exc)
else:
    raise AssertionError("upstream SHMEM Buffer constructor guard did not fire")

print(f"adapter={adapter.__file__}")
print("tilexr_mindspeed_adapter_preflight=PASS")
PY
