#!/usr/bin/env python3
"""Generate an HCCL v1.2 rank table from Ascend 950 RootInfo files."""

from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Sequence


SUPER_POD_STRIDE = 4_194_304
DEVICE_STRIDE = 262_145
HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
HEX_EID_PATTERN = re.compile(r"^[0-9A-Fa-f]{32}$")


class RankTableError(ValueError):
    """Raised when discovery data cannot form an unambiguous rank table."""


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise RankTableError(f"{label} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise RankTableError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise RankTableError(f"{label} must not be negative")
    return parsed


def read_hosts(path: Path) -> list[str]:
    """Read ordered hosts, ignoring comments and optional credential suffixes."""
    hosts: list[str] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RankTableError(f"failed to read host inventory {path}: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split(maxsplit=1)[0]
        host = token.split(":", maxsplit=1)[0]
        if not HOST_PATTERN.fullmatch(host):
            raise RankTableError(f"invalid host at {path}:{line_number}")
        if host in seen:
            raise RankTableError(f"duplicate host in {path}: {host}")
        seen.add(host)
        hosts.append(host)

    if not hosts:
        raise RankTableError(f"host inventory is empty: {path}")
    return hosts


def _decode_json_documents(text: str, host: str) -> tuple[object, object]:
    decoder = json.JSONDecoder()
    documents: list[object] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position == len(text):
            break
        try:
            document, position = decoder.raw_decode(text, position)
        except json.JSONDecodeError as exc:
            raise RankTableError(f"{host}: SSH output is not valid JSON: {exc}") from exc
        documents.append(document)
    if len(documents) != 2:
        raise RankTableError(
            f"{host}: expected interface JSON and RootInfo JSON, got {len(documents)} documents"
        )
    return documents[0], documents[1]


def collect_host_snapshot(
    host: str,
    *,
    ssh_user: str,
    interface: str,
    rootinfo_path: str,
    timeout: int,
    ssh_options: Sequence[str],
) -> tuple[object, object]:
    if not HOST_PATTERN.fullmatch(ssh_user):
        raise RankTableError(f"invalid SSH user: {ssh_user}")
    for option in ssh_options:
        if "\n" in option or "\r" in option or "\0" in option:
            raise RankTableError("SSH options must be single-line values")

    remote_command = "ip -j -4 addr show dev {} && cat {}".format(
        shlex.quote(interface), shlex.quote(rootinfo_path)
    )
    command = [
        "ssh",
        "-T",
        "-o",
        f"ConnectTimeout={min(timeout, 15)}",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=2",
    ]
    for option in ssh_options:
        command.extend(("-o", option))
    command.extend((f"{ssh_user}@{host}", remote_command))

    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RankTableError("system ssh command was not found") from exc
    except subprocess.TimeoutExpired as exc:
        raise RankTableError(f"{host}: SSH discovery timed out after {timeout}s") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit code {completed.returncode}"
        raise RankTableError(f"{host}: SSH discovery failed: {detail}")
    return _decode_json_documents(completed.stdout, host)


def collect_snapshots(
    hosts: Sequence[str],
    *,
    ssh_user: str,
    interface: str,
    rootinfo_path: str,
    timeout: int,
    ssh_options: Sequence[str],
    jobs: int,
) -> dict[str, tuple[object, object]]:
    if jobs < 1:
        raise RankTableError("jobs must be at least 1")
    snapshots: dict[str, tuple[object, object]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, len(hosts))) as pool:
        futures = {
            pool.submit(
                collect_host_snapshot,
                host,
                ssh_user=ssh_user,
                interface=interface,
                rootinfo_path=rootinfo_path,
                timeout=timeout,
                ssh_options=ssh_options,
            ): host
            for host in hosts
        }
        for future in concurrent.futures.as_completed(futures):
            host = futures[future]
            snapshots[host] = future.result()
    return snapshots


def extract_server_id(interface_info: object, host: str) -> str:
    if not isinstance(interface_info, list):
        raise RankTableError(f"{host}: interface query must return a JSON array")
    addresses: list[str] = []
    for interface in interface_info:
        if not isinstance(interface, dict):
            continue
        addr_info = interface.get("addr_info", [])
        if not isinstance(addr_info, list):
            continue
        for address in addr_info:
            if not isinstance(address, dict):
                continue
            if address.get("family") == "inet" and address.get("scope") == "global":
                local = address.get("local")
                try:
                    parsed = ipaddress.IPv4Address(str(local))
                except ipaddress.AddressValueError as exc:
                    raise RankTableError(f"{host}: malformed interface IPv4 address") from exc
                addresses.append(str(parsed))
    addresses = list(dict.fromkeys(addresses))
    if len(addresses) != 1:
        raise RankTableError(f"{host}: expected one global interface IPv4 address")
    return addresses[0]


def _primary_eid(device: dict[str, object], host: str, device_id: int) -> str:
    level_list = device.get("level_list")
    if not isinstance(level_list, list):
        raise RankTableError(f"{host}: device {device_id} has no level_list")
    network_levels = [
        level
        for level in level_list
        if isinstance(level, dict) and _integer(level.get("net_layer"), "net_layer") == 1
    ]
    if len(network_levels) != 1:
        raise RankTableError(f"{host}: device {device_id} must have one net_layer=1 entry")

    rank_addresses = network_levels[0].get("rank_addr_list")
    if not isinstance(rank_addresses, list):
        raise RankTableError(f"{host}: device {device_id} has no rank_addr_list")
    candidates: list[str] = []
    for address in rank_addresses:
        if not isinstance(address, dict) or address.get("addr_type") != "EID":
            continue
        ports = address.get("ports")
        if not isinstance(ports, list) or not ports:
            continue
        if all(isinstance(port, str) and port.startswith("0/") for port in ports):
            candidates.append(str(address.get("addr", "")))
    if len(candidates) != 1:
        raise RankTableError(f"{host}: device {device_id} requires a unique 0/ EID")

    raw_eid = candidates[0]
    if not HEX_EID_PATTERN.fullmatch(raw_eid):
        raise RankTableError(f"{host}: device {device_id} has a malformed EID")
    return str(ipaddress.IPv6Address(bytes.fromhex(raw_eid)))


def extract_devices(
    root_info: object, *, devices_per_server: int, host: str
) -> list[tuple[int, str]]:
    if not isinstance(root_info, dict):
        raise RankTableError(f"{host}: RootInfo must be a JSON object")
    if str(root_info.get("version")) != "2.0":
        raise RankTableError(f"{host}: expected RootInfo version 2.0")
    if _integer(root_info.get("rank_count"), "rank_count") != devices_per_server:
        raise RankTableError(
            f"{host}: RootInfo rank_count does not match {devices_per_server} devices"
        )
    rank_list = root_info.get("rank_list")
    if not isinstance(rank_list, list) or len(rank_list) != devices_per_server:
        raise RankTableError(f"{host}: RootInfo rank_list is incomplete")

    by_device: dict[int, dict[str, object]] = {}
    for device in rank_list:
        if not isinstance(device, dict):
            raise RankTableError(f"{host}: RootInfo device entry must be an object")
        device_id = _integer(device.get("device_id"), "device_id")
        if device_id in by_device:
            raise RankTableError(f"{host}: duplicate device_id {device_id}")
        by_device[device_id] = device
    expected_ids = set(range(devices_per_server))
    if set(by_device) != expected_ids:
        raise RankTableError(f"{host}: device IDs must be 0..{devices_per_server - 1}")

    return [
        (device_id, _primary_eid(by_device[device_id], host, device_id))
        for device_id in range(devices_per_server)
    ]


def build_rank_table(
    *,
    hosts: Sequence[str],
    super_pod_ids: Sequence[int],
    servers_per_super_pod: int,
    devices_per_server: int,
    collect: Callable[[str], tuple[object, object]],
) -> dict[str, object]:
    if servers_per_super_pod < 1 or devices_per_server < 1:
        raise RankTableError("server and device counts must be at least 1")
    if len(set(super_pod_ids)) != len(super_pod_ids):
        raise RankTableError("super-pod IDs must be unique")
    if any(super_pod_id < 0 for super_pod_id in super_pod_ids):
        raise RankTableError("super-pod IDs must not be negative")
    expected_servers = len(super_pod_ids) * servers_per_super_pod
    if len(hosts) != expected_servers:
        raise RankTableError(
            f"host count {len(hosts)} does not match {expected_servers} super-pod servers"
        )

    server_list: list[dict[str, object]] = []
    for server_index, host in enumerate(hosts):
        interface_info, root_info = collect(host)
        server_id = extract_server_id(interface_info, host)
        super_pod_id = super_pod_ids[server_index // servers_per_super_pod]
        devices = []
        for device_id, device_ip in extract_devices(
            root_info, devices_per_server=devices_per_server, host=host
        ):
            rank_id = server_index * devices_per_server + device_id
            super_device_id = super_pod_id * SUPER_POD_STRIDE + device_id * DEVICE_STRIDE
            devices.append(
                {
                    "device_id": str(device_id),
                    "rank_id": str(rank_id),
                    "super_device_id": str(super_device_id),
                    "device_ip": device_ip,
                }
            )
        server_list.append({"server_id": server_id, "device": devices})

    super_pod_list = []
    for pod_index, super_pod_id in enumerate(super_pod_ids):
        begin = pod_index * servers_per_super_pod
        pod_servers = server_list[begin : begin + servers_per_super_pod]
        super_pod_list.append(
            {
                "super_pod_id": str(super_pod_id),
                "server_list": [
                    {"server_id": server["server_id"]} for server in pod_servers
                ],
            }
        )

    table: dict[str, object] = {
        "version": "1.2",
        "server_count": str(len(server_list)),
        "server_list": server_list,
        "super_pod_list": super_pod_list,
        "status": "completed",
    }
    validate_rank_table(table)
    return table


def validate_rank_table(table: object) -> None:
    if not isinstance(table, dict):
        raise RankTableError("rank table must be a JSON object")
    if table.get("version") != "1.2" or table.get("status") != "completed":
        raise RankTableError("rank table requires version 1.2 and completed status")
    server_list = table.get("server_list")
    super_pod_list = table.get("super_pod_list")
    if not isinstance(server_list, list) or not server_list:
        raise RankTableError("rank table server_list must not be empty")
    if not isinstance(super_pod_list, list) or not super_pod_list:
        raise RankTableError("rank table super_pod_list must not be empty")
    if _integer(table.get("server_count"), "server_count") != len(server_list):
        raise RankTableError("server_count does not match server_list")

    server_ids: list[str] = []
    rank_ids: list[int] = []
    device_ips: list[str] = []
    device_count: int | None = None
    devices_by_server: dict[str, list[dict[str, object]]] = {}
    for server in server_list:
        if not isinstance(server, dict) or not isinstance(server.get("device"), list):
            raise RankTableError("each server requires a device list")
        try:
            server_id = str(ipaddress.IPv4Address(str(server.get("server_id"))))
        except ipaddress.AddressValueError as exc:
            raise RankTableError("server_id must be an IPv4 address") from exc
        if server_id in devices_by_server:
            raise RankTableError(f"duplicate server_id: {server_id}")
        devices = server["device"]
        if not devices:
            raise RankTableError(f"{server_id}: device list must not be empty")
        if device_count is None:
            device_count = len(devices)
        elif len(devices) != device_count:
            raise RankTableError("all servers must have the same device count")
        devices_by_server[server_id] = devices
        server_ids.append(server_id)
        for device in devices:
            if not isinstance(device, dict):
                raise RankTableError("device entries must be objects")
            rank_ids.append(_integer(device.get("rank_id"), "rank_id"))
            device_ip = str(device.get("device_ip"))
            try:
                ipaddress.IPv6Address(device_ip)
            except ipaddress.AddressValueError as exc:
                raise RankTableError(f"invalid device_ip: {device_ip}") from exc
            device_ips.append(device_ip)
    if rank_ids != list(range(len(rank_ids))):
        raise RankTableError("rank IDs must be contiguous in server/device order")
    if len(set(device_ips)) != len(device_ips):
        raise RankTableError("rank table contains duplicate device_ip values")

    referenced_servers: list[str] = []
    seen_pods: set[int] = set()
    assert device_count is not None
    for pod in super_pod_list:
        if not isinstance(pod, dict) or not isinstance(pod.get("server_list"), list):
            raise RankTableError("each super-pod requires a server_list")
        super_pod_id = _integer(pod.get("super_pod_id"), "super_pod_id")
        if super_pod_id in seen_pods:
            raise RankTableError(f"duplicate super_pod_id: {super_pod_id}")
        seen_pods.add(super_pod_id)
        for server_reference in pod["server_list"]:
            if not isinstance(server_reference, dict):
                raise RankTableError("super-pod server references must be objects")
            server_id = str(server_reference.get("server_id"))
            if server_id not in devices_by_server:
                raise RankTableError(f"unknown super-pod server_id: {server_id}")
            referenced_servers.append(server_id)
            devices = devices_by_server[server_id]
            device_ids = [_integer(device.get("device_id"), "device_id") for device in devices]
            if device_ids != list(range(device_count)):
                raise RankTableError(f"{server_id}: device IDs must be contiguous")
            for device_id, device in zip(device_ids, devices):
                expected = super_pod_id * SUPER_POD_STRIDE + device_id * DEVICE_STRIDE
                actual = _integer(device.get("super_device_id"), "super_device_id")
                if actual != expected:
                    raise RankTableError(
                        f"{server_id}: super_device_id does not match super-pod/device encoding"
                    )
    if referenced_servers != server_ids:
        raise RankTableError("super_pod_list must reference every server once and in order")


def write_rank_table(table: dict[str, object], output: Path) -> None:
    validate_rank_table(table)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(table, indent=2) + "\n").encode("utf-8")
        output.write_bytes(payload)
    except OSError as exc:
        raise RankTableError(f"failed to write rank table {output}: {exc}") from exc


def _parse_super_pod_ids(value: str) -> list[int]:
    try:
        values = [_integer(part.strip(), "super_pod_id") for part in value.split(",")]
    except RankTableError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not values or any(not part.strip() for part in value.split(",")):
        raise argparse.ArgumentTypeError("super-pod IDs must be a comma-separated list")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hosts", type=Path, help="Ordered host inventory")
    source.add_argument("--check", type=Path, help="Validate an existing rank table")
    parser.add_argument("--super-pod-ids", type=_parse_super_pod_ids)
    parser.add_argument("--servers-per-super-pod", type=int, default=8)
    parser.add_argument("--devices-per-server", type=int, default=8)
    parser.add_argument("--interface", default="data0.3001")
    parser.add_argument("--rootinfo-path", default="/etc/hccl_rootinfo.json.bak")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ssh-user", default="root")
    parser.add_argument(
        "--ssh-option",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional system SSH -o option; repeat as needed",
    )
    parser.add_argument("--timeout", type=int, default=30, help="Per-host SSH timeout")
    parser.add_argument("--jobs", type=int, default=16, help="Parallel SSH queries")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.check is not None:
            table = json.loads(args.check.read_text(encoding="utf-8"))
            validate_rank_table(table)
            rank_count = sum(len(server["device"]) for server in table["server_list"])
            print(f"valid HCCL v1.2 rank table: servers={table['server_count']} ranks={rank_count}")
            return 0
        if args.super_pod_ids is None:
            parser.error("--super-pod-ids is required with --hosts")
        if args.output is None:
            parser.error("--output is required with --hosts")
        if args.timeout < 1:
            parser.error("--timeout must be at least 1")

        hosts = read_hosts(args.hosts)
        snapshots = collect_snapshots(
            hosts,
            ssh_user=args.ssh_user,
            interface=args.interface,
            rootinfo_path=args.rootinfo_path,
            timeout=args.timeout,
            ssh_options=args.ssh_option,
            jobs=args.jobs,
        )
        table = build_rank_table(
            hosts=hosts,
            super_pod_ids=args.super_pod_ids,
            servers_per_super_pod=args.servers_per_super_pod,
            devices_per_server=args.devices_per_server,
            collect=lambda host: snapshots[host],
        )
        write_rank_table(table, args.output)
        rank_count = len(hosts) * args.devices_per_server
        print(f"wrote {args.output}: servers={len(hosts)} ranks={rank_count}")
        return 0
    except (OSError, json.JSONDecodeError, RankTableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
