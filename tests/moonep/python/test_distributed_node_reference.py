from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.moonep import distributed_node
from tools.moonep.benchmark import topology_metadata
from tools.moonep.distributed_node import (
    _require_shared_environment,
    build_parser,
    resolve_node_topology,
)
from tools.moonep.launcher import _process_command


def test_reference_topology_supports_64_and_128_ranks() -> None:
    sixty_four = resolve_node_topology(
        node_count=8,
        node_rank=0,
        local_world_size=8,
        physical_device_count=8,
        planner_block_dim=64,
        mode="reference",
    )
    one_twenty_eight = resolve_node_topology(
        node_count=16,
        node_rank=15,
        local_world_size=8,
        physical_device_count=8,
        planner_block_dim=64,
        mode="reference",
    )

    assert sixty_four["world_size"] == 64
    assert sixty_four["first_global_rank"] == 0
    assert one_twenty_eight["world_size"] == 128
    assert one_twenty_eight["first_global_rank"] == 120


def test_native_multinode_topology_allows_128_ranks_with_64_planner_blocks() -> None:
    topology = resolve_node_topology(
        node_count=16,
        node_rank=15,
        local_world_size=8,
        physical_device_count=8,
        planner_block_dim=64,
        mode="benchmark",
    )

    assert topology["world_size"] == 128
    assert topology["first_global_rank"] == 120
    assert topology["planner_block_dim"] == 64


def test_correctness_topology_supports_two_nodes_with_one_rank_per_device() -> None:
    topology = resolve_node_topology(
        node_count=2,
        node_rank=1,
        local_world_size=8,
        physical_device_count=8,
        planner_block_dim=64,
        mode="correctness",
    )

    assert topology["world_size"] == 16
    assert topology["first_global_rank"] == 8
    assert topology["local_world_size"] == 8
    assert topology["physical_device_count"] == 8


def test_native_128_rank_metadata_keeps_64_planner_blocks(monkeypatch) -> None:
    context = SimpleNamespace(
        global_rank=127,
        global_world_size=128,
        node_rank=15,
        node_count=16,
        local_rank=7,
        local_world_size=8,
        planner_group_rank=127,
        planner_group_size=128,
        lane_group_rank=0,
        lane_group_size=1,
    )
    monkeypatch.setenv("TILEXR_PHYSICAL_DEVICE_COUNT", "8")
    monkeypatch.setenv("TILEXR_RANKS_PER_DEVICE", "1")
    monkeypatch.setenv("TILEXR_MOONEP_PLANNER_BLOCK_DIM", "64")

    metadata = topology_metadata(context)

    assert metadata["planner_group_size"] == 128
    assert metadata["planner_block_dim"] == 64


def test_reference_node_builds_the_reference_worker_command() -> None:
    args = build_parser().parse_args(
        [
            "--cases",
            "cases.json",
            "--case-ids",
            "planning-128rank-single-route",
            "--output-dir",
            "output",
            "--install-prefix",
            "install",
            "--node-count",
            "16",
            "--node-rank",
            "15",
            "--mode",
            "reference",
        ]
    )

    command = _process_command(args)
    assert "tools.moonep.benchmark" in command
    assert "tools.moonep.dispatch_hot_loop" not in command
    assert command[command.index("--mode") + 1] == "reference"


def test_correctness_node_builds_the_correctness_worker_command() -> None:
    args = build_parser().parse_args(
        [
            "--cases",
            "cases.json",
            "--case-ids",
            "planning-16rank-16card-single-route",
            "--output-dir",
            "output",
            "--install-prefix",
            "install",
            "--node-count",
            "2",
            "--node-rank",
            "1",
            "--mode",
            "correctness",
        ]
    )

    command = _process_command(args)
    assert "tools.moonep.benchmark" in command
    assert "tools.moonep.dispatch_hot_loop" not in command
    assert command[command.index("--mode") + 1] == "correctness"


def test_reference_node_requires_torch_rendezvous_but_not_native_comm() -> None:
    environment = {
        "MASTER_ADDR": "141.61.55.118",
        "MASTER_PORT": "29600",
        "TILEXR_MOONEP_LAUNCH_ID": "phase25-reference",
    }
    _require_shared_environment(environment, mode="reference")

    with pytest.raises(ValueError, match="MASTER_PORT"):
        _require_shared_environment(
            {name: value for name, value in environment.items() if name != "MASTER_PORT"},
            mode="reference",
        )


def test_correctness_node_requires_native_and_torch_rendezvous() -> None:
    environment = {
        "MASTER_ADDR": "192.0.2.10",
        "MASTER_PORT": "29600",
        "TILEXR_COMM_ID": "192.0.2.10:12001",
        "TILEXR_MOONEP_BARRIER_ADDR": "192.0.2.10:12114",
        "TILEXR_MOONEP_LAUNCH_ID": "two-node-correctness",
        "TILEXR_MOONEP_LAUNCH_SECRET": "00" * 32,
    }
    _require_shared_environment(environment, mode="correctness")

    for missing_name in ("MASTER_PORT", "TILEXR_COMM_ID"):
        with pytest.raises(ValueError, match=missing_name):
            _require_shared_environment(
                {
                    name: value
                    for name, value in environment.items()
                    if name != missing_name
                },
                mode="correctness",
            )


def test_native_node_preserves_explicit_udma_qp_route_spec(
    monkeypatch, tmp_path: Path
) -> None:
    environments = []

    class FinishedProcess:
        @staticmethod
        def poll():
            return 0

        @staticmethod
        def terminate():
            return None

        @staticmethod
        def wait(timeout=None):
            return 0

        @staticmethod
        def kill():
            return None

    def popen(_command, *, cwd, env, stdout, stderr):
        environments.append(env)
        return FinishedProcess()

    environment = {
        "TILEXR_COMM_ID": "127.0.0.1:12001",
        "TILEXR_MOONEP_BARRIER_ADDR": "127.0.0.1:12114",
        "TILEXR_MOONEP_LAUNCH_ID": "qp-route-override",
        "TILEXR_MOONEP_LAUNCH_SECRET": "00" * 32,
        "TILEXR_UDMA_QP_ROUTE_SPEC": "port_count:2,port_count:6",
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(distributed_node.subprocess, "Popen", popen)

    assert distributed_node.main(
        [
            "--cases",
            str(tmp_path / "cases.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--install-prefix",
            str(tmp_path / "install"),
            "--node-count",
            "1",
            "--node-rank",
            "0",
            "--local-world-size",
            "1",
            "--physical-device-count",
            "1",
            "--benchmark-kind",
            "flow",
        ]
    ) == 0
    assert environments[0]["TILEXR_UDMA_QP_ROUTE_SPEC"] == environment[
        "TILEXR_UDMA_QP_ROUTE_SPEC"
    ]
