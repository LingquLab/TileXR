from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
GENERATOR_PATH = (
    ROOT / "tools" / "moonep" / "mindspeed" / "generate_rank_table.py"
)


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_rank_table", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def root_info(*primary_eids: str) -> dict[str, object]:
    rank_list = []
    for device_id, primary_eid in enumerate(primary_eids):
        rank_list.append(
            {
                "device_id": device_id,
                "local_id": device_id,
                "level_list": [
                    {
                        "net_layer": 0,
                        "rank_addr_list": [
                            {
                                "addr_type": "EID",
                                "addr": "f" * 32,
                                "ports": ["0/0"],
                            }
                        ],
                    },
                    {
                        "net_layer": 1,
                        "rank_addr_list": [
                            {
                                "addr_type": "EID",
                                "addr": "00000000007f02000010000000000001",
                                "ports": ["1/0", "1/1"],
                            },
                            {
                                "addr_type": "EID",
                                "addr": primary_eid,
                                "ports": ["0/1", "0/2"],
                            },
                        ],
                    },
                ],
            }
        )
    return {"version": "2.0", "rank_count": len(rank_list), "rank_list": rank_list}


def interface_info(address: str) -> list[dict[str, object]]:
    return [
        {
            "ifname": "data0.3001",
            "addr_info": [
                {
                    "family": "inet",
                    "local": address,
                    "scope": "global",
                }
            ],
        }
    ]


def test_builds_hccl_v12_table_in_host_and_device_order() -> None:
    generator = load_generator()
    hosts = ["node-a", "node-b"]
    snapshots = {
        "node-a": (
            interface_info("192.0.2.10"),
            root_info(
                "20010db8000000000000000000000001",
                "20010db8000000000000000000000002",
            ),
        ),
        "node-b": (
            interface_info("198.51.100.20"),
            root_info(
                "20010db8000000000000000000000101",
                "20010db8000000000000000000000102",
            ),
        ),
    }

    table = generator.build_rank_table(
        hosts=hosts,
        super_pod_ids=[5, 15],
        servers_per_super_pod=1,
        devices_per_server=2,
        collect=lambda host: snapshots[host],
    )

    assert table["version"] == "1.2"
    assert table["server_count"] == "2"
    assert table["status"] == "completed"
    assert [server["server_id"] for server in table["server_list"]] == [
        "192.0.2.10",
        "198.51.100.20",
    ]
    devices = [
        device
        for server in table["server_list"]
        for device in server["device"]
    ]
    assert [device["rank_id"] for device in devices] == ["0", "1", "2", "3"]
    assert devices[0] == {
        "device_id": "0",
        "rank_id": "0",
        "super_device_id": "20971520",
        "device_ip": "2001:db8::1",
    }
    assert devices[3]["super_device_id"] == "63176705"
    assert table["super_pod_list"] == [
        {
            "super_pod_id": "5",
            "server_list": [{"server_id": "192.0.2.10"}],
        },
        {
            "super_pod_id": "15",
            "server_list": [{"server_id": "198.51.100.20"}],
        },
    ]
    generator.validate_rank_table(table)


def test_host_inventory_accepts_comments_and_discards_credentials(tmp_path: Path) -> None:
    generator = load_generator()
    hosts_file = tmp_path / "hosts"
    hosts_file.write_text(
        "# pod 5\nnode-a:do-not-retain\n\nnode-b\n",
        encoding="utf-8",
    )

    assert generator.read_hosts(hosts_file) == ["node-a", "node-b"]


def test_system_ssh_collector_queries_only_interface_and_rootinfo(monkeypatch) -> None:
    generator = load_generator()
    captured: list[list[str]] = []

    def fake_run(command, **_kwargs):
        captured.append(command)
        stdout = json.dumps(interface_info("192.0.2.10"))
        stdout += json.dumps(root_info("20010db8000000000000000000000001"))
        return generator.subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(generator.subprocess, "run", fake_run)
    interface, root = generator.collect_host_snapshot(
        "node-a",
        ssh_user="root",
        interface="data0.3001",
        rootinfo_path="/etc/hccl_rootinfo.json.bak",
        timeout=30,
        ssh_options=["StrictHostKeyChecking=yes"],
    )

    assert interface == interface_info("192.0.2.10")
    assert root == root_info("20010db8000000000000000000000001")
    assert captured[0][-2] == "root@node-a"
    assert captured[0][-1] == (
        "ip -j -4 addr show dev data0.3001 && cat /etc/hccl_rootinfo.json.bak"
    )


def test_rejects_an_ambiguous_primary_eid() -> None:
    generator = load_generator()
    info = root_info("20010db8000000000000000000000001")
    level = info["rank_list"][0]["level_list"][1]
    level["rank_addr_list"].append(
        {
            "addr_type": "EID",
            "addr": "20010db8000000000000000000000002",
            "ports": ["0/3"],
        }
    )

    with pytest.raises(generator.RankTableError, match="unique 0/ EID"):
        generator.extract_devices(info, devices_per_server=1, host="node-a")


def test_rejects_duplicate_eids_across_servers() -> None:
    generator = load_generator()
    duplicate = "20010db8000000000000000000000001"
    snapshots = {
        "node-a": (interface_info("192.0.2.10"), root_info(duplicate)),
        "node-b": (interface_info("198.51.100.20"), root_info(duplicate)),
    }

    with pytest.raises(generator.RankTableError, match="duplicate device_ip"):
        generator.build_rank_table(
            hosts=["node-a", "node-b"],
            super_pod_ids=[5, 15],
            servers_per_super_pod=1,
            devices_per_server=1,
            collect=lambda host: snapshots[host],
        )


def test_rendered_table_is_stable_and_offline_check_rejects_gaps(tmp_path: Path) -> None:
    generator = load_generator()
    table = generator.build_rank_table(
        hosts=["node-a"],
        super_pod_ids=[5],
        servers_per_super_pod=1,
        devices_per_server=1,
        collect=lambda _host: (
            interface_info("192.0.2.10"),
            root_info("20010db8000000000000000000000001"),
        ),
    )
    output = tmp_path / "rank_table.json"
    generator.write_rank_table(table, output)
    assert output.read_text(encoding="utf-8") == json.dumps(table, indent=2) + "\n"
    assert output.read_bytes() == (json.dumps(table, indent=2) + "\n").encode("utf-8")

    table["server_list"][0]["device"][0]["rank_id"] = "1"
    with pytest.raises(generator.RankTableError, match="contiguous"):
        generator.validate_rank_table(table)
