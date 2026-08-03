from __future__ import annotations

import hashlib
import hmac
import json
import os
import socket
import struct
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


_MAX_FRAME_BYTES = 4096
_BARRIER_ADDRESS_ENV = "TILEXR_MOONEP_BARRIER_ADDR"
_BARRIER_TIMEOUT_ENV = "TILEXR_MOONEP_BARRIER_TIMEOUT_SEC"
_LAUNCH_ID_ENV = "TILEXR_MOONEP_LAUNCH_ID"
_LAUNCH_SECRET_ENV = "TILEXR_MOONEP_LAUNCH_SECRET"
_MANAGED_LAUNCH_ENV = "TILEXR_MOONEP_MANAGED_LAUNCH"


@dataclass(frozen=True)
class CompletionDecision:
    release: bool
    abort: bool


def parse_host_port(value: str) -> tuple[str, int]:
    host, separator, port_text = value.rpartition(":")
    if not separator or not host:
        raise ValueError(f"invalid host:port endpoint: {value!r}")
    try:
        port = int(port_text)
    except ValueError as exc:
        raise ValueError(f"invalid host:port endpoint: {value!r}") from exc
    if port <= 0 or port > 65535:
        raise ValueError(f"invalid host:port endpoint: {value!r}")
    return host, port


def offset_host_port(value: str, offset: int = 113) -> str:
    host, port = parse_host_port(value)
    candidate = port + offset
    if candidate > 65535:
        candidate = port - offset
    if candidate <= 0:
        raise ValueError(f"cannot derive barrier endpoint from {value!r}")
    return f"{host}:{candidate}"


def _recv_exact(sock: socket.socket, byte_count: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < byte_count:
        chunk = sock.recv(byte_count - len(chunks))
        if not chunk:
            raise ConnectionError("completion rendezvous peer closed the socket")
        chunks.extend(chunk)
    return bytes(chunks)


def _send_frame(sock: socket.socket, value: dict[str, object]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(payload) > _MAX_FRAME_BYTES:
        raise ValueError("completion rendezvous frame is too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def _frame_mac(value: dict[str, object], secret: str) -> str:
    try:
        key = bytes.fromhex(secret)
    except ValueError as exc:
        raise ValueError("completion rendezvous secret must be hexadecimal") from exc
    if len(key) != 32:
        raise ValueError("completion rendezvous secret must contain 32 bytes")
    unsigned = {name: field for name, field in value.items() if name != "mac"}
    payload = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _sign_frame(value: dict[str, object], secret: str) -> dict[str, object]:
    signed = dict(value)
    signed["mac"] = _frame_mac(signed, secret)
    return signed


def _frame_authenticated(value: dict[str, object], secret: str) -> bool:
    mac = value.get("mac")
    return type(mac) is str and hmac.compare_digest(mac, _frame_mac(value, secret))


def _recv_frame(sock: socket.socket) -> dict[str, object]:
    byte_count = struct.unpack("!I", _recv_exact(sock, 4))[0]
    if byte_count <= 0 or byte_count > _MAX_FRAME_BYTES:
        raise ValueError("invalid completion rendezvous frame size")
    value = json.loads(_recv_exact(sock, byte_count).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("completion rendezvous frame must be an object")
    return value


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("completion rendezvous timed out")
    return remaining


def _arrival_valid(
    value: dict[str, object],
    *,
    launch_id: str,
    secret: str,
    case_id: str,
    world_size: int,
    seen: set[int],
) -> bool:
    rank = value.get("rank")
    return (
        value.get("kind") == "arrival"
        and value.get("launch_id") == launch_id
        and value.get("case_id") == case_id
        and type(rank) is int
        and 0 < rank < world_size
        and rank not in seen
        and type(value.get("quiesced")) is bool
        and type(value.get("passed")) is bool
        and _frame_authenticated(value, secret)
    )


def _server(
    world_size: int,
    host: str,
    port: int,
    launch_id: str,
    secret: str,
    case_id: str,
    quiesced: bool,
    passed: bool,
    timeout_sec: float,
) -> CompletionDecision:
    deadline = time.monotonic() + timeout_sec
    clients: dict[int, socket.socket] = {}
    states = {0: (quiesced, passed)}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(world_size - 1)
        try:
            while len(clients) < world_size - 1:
                listener.settimeout(_remaining(deadline))
                client, _ = listener.accept()
                try:
                    client.settimeout(_remaining(deadline))
                    value = _recv_frame(client)
                    if not _arrival_valid(
                        value,
                        launch_id=launch_id,
                        secret=secret,
                        case_id=case_id,
                        world_size=world_size,
                        seen=set(clients),
                    ):
                        client.close()
                        continue
                    rank = int(value["rank"])
                    clients[rank] = client
                    states[rank] = (bool(value["quiesced"]), bool(value["passed"]))
                except Exception:
                    client.close()
                    continue

            decision = CompletionDecision(
                release=all(state[0] for state in states.values()),
                abort=not all(state[1] for state in states.values()),
            )
            if not decision.release:
                decision = CompletionDecision(release=False, abort=True)
            response = _sign_frame(
                {
                    "kind": "decision",
                    "launch_id": launch_id,
                    "case_id": case_id,
                    "release": decision.release,
                    "abort": decision.abort,
                },
                secret,
            )
            for client in clients.values():
                client.settimeout(_remaining(deadline))
                _send_frame(client, response)
            return decision
        finally:
            for client in clients.values():
                client.close()


def _decision_valid(
    value: dict[str, object], *, launch_id: str, secret: str, case_id: str
) -> bool:
    return (
        value.get("kind") == "decision"
        and value.get("launch_id") == launch_id
        and value.get("case_id") == case_id
        and type(value.get("release")) is bool
        and type(value.get("abort")) is bool
        and _frame_authenticated(value, secret)
    )


def _client(
    rank: int,
    host: str,
    port: int,
    launch_id: str,
    secret: str,
    case_id: str,
    quiesced: bool,
    passed: bool,
    timeout_sec: float,
) -> CompletionDecision:
    deadline = time.monotonic() + timeout_sec
    arrival = _sign_frame(
        {
            "kind": "arrival",
            "launch_id": launch_id,
            "case_id": case_id,
            "rank": rank,
            "quiesced": quiesced,
            "passed": passed,
        },
        secret,
    )
    while True:
        remaining = _remaining(deadline)
        try:
            with socket.create_connection((host, port), timeout=min(1.0, remaining)) as sock:
                sock.settimeout(_remaining(deadline))
                _send_frame(sock, arrival)
                decision = _recv_frame(sock)
                if not _decision_valid(
                    decision, launch_id=launch_id, secret=secret, case_id=case_id
                ):
                    raise RuntimeError("invalid completion rendezvous decision")
                return CompletionDecision(
                    release=bool(decision["release"]),
                    abort=bool(decision["abort"]),
                )
        except (ConnectionRefusedError, ConnectionError, socket.timeout, OSError):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"completion rendezvous rank {rank} could not reach {host}:{port}"
                )
            time.sleep(min(0.05, _remaining(deadline)))


def completion_barrier(
    rank: int,
    world_size: int,
    endpoint: str,
    *,
    launch_id: str,
    secret: str,
    case_id: str,
    quiesced: bool,
    passed: bool,
    timeout_sec: float,
) -> CompletionDecision:
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid completion rendezvous rank metadata")
    if world_size == 1:
        return CompletionDecision(release=quiesced, abort=not passed or not quiesced)
    if timeout_sec <= 0 or not launch_id or not secret or not case_id:
        raise ValueError("completion rendezvous metadata is incomplete")
    host, port = parse_host_port(endpoint)
    if rank == 0:
        return _server(
            world_size,
            host,
            port,
            launch_id,
            secret,
            case_id,
            quiesced,
            passed,
            timeout_sec,
        )
    return _client(
        rank,
        host,
        port,
        launch_id,
        secret,
        case_id,
        quiesced,
        passed,
        timeout_sec,
    )


def completion_barrier_from_env(
    rank: int,
    world_size: int,
    *,
    case_id: str,
    quiesced: bool,
    passed: bool,
    environment: Mapping[str, str] | None = None,
) -> CompletionDecision:
    env = os.environ if environment is None else environment
    endpoint = env.get(_BARRIER_ADDRESS_ENV)
    if not endpoint:
        comm_id = env.get("TILEXR_COMM_ID")
        if not comm_id:
            raise RuntimeError(f"{_BARRIER_ADDRESS_ENV} or TILEXR_COMM_ID is required")
        endpoint = offset_host_port(comm_id)
    launch_id = env.get(_LAUNCH_ID_ENV, "")
    secret = env.get(_LAUNCH_SECRET_ENV, "")
    timeout_sec = float(env.get(_BARRIER_TIMEOUT_ENV, "1800"))
    return completion_barrier(
        rank,
        world_size,
        endpoint,
        launch_id=launch_id,
        secret=secret,
        case_id=case_id,
        quiesced=quiesced,
        passed=passed,
        timeout_sec=timeout_sec,
    )


def signal_managed_abort(output_root: Path, rank: int, reason: str) -> Path:
    launch_id = os.environ.get(_LAUNCH_ID_ENV, "")
    if not launch_id or os.environ.get(_MANAGED_LAUNCH_ENV) != "1":
        raise RuntimeError(reason)
    path = output_root / f".tilexr_abort_{launch_id}_{rank}"
    path.write_text(reason[:_MAX_FRAME_BYTES], encoding="utf-8")
    return path


def hold_for_managed_abort() -> None:
    while True:
        time.sleep(3600)
