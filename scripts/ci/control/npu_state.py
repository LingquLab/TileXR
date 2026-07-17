#!/usr/bin/env python3
"""Read NPU process and health state without changing NPU workloads."""

import dataclasses
import os
import pwd
import re
import subprocess
from typing import Callable, List, Optional, Tuple


class ResourceTimeout(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class NpuProcess:
    device: int
    pid: int
    name: str
    owner: str = "unknown"


@dataclasses.dataclass(frozen=True)
class Snapshot:
    healthy: bool
    processes: Tuple[NpuProcess, ...]

    @property
    def idle(self):
        return self.healthy and not self.processes


def parse_process_table(text: str) -> List[NpuProcess]:
    """Extract valid process rows after the npu-smi process-table heading."""
    processes = []
    in_process_table = False
    for line in text.splitlines():
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table or "|" not in line:
            continue

        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue
        device_chip = cells[0].split()
        pid = cells[1]
        if not device_chip or not device_chip[0].isdecimal() or not pid.isdecimal():
            continue
        processes.append(NpuProcess(int(device_chip[0]), int(pid), cells[2]))
    return processes


def parse_health(text: str) -> str:
    """Return the first top-level Health field reported by npu-smi."""
    for line in text.splitlines():
        match = re.match(r"^\s*Health\s*:\s*(.*?)\s*$", line)
        if match:
            return match.group(1)
    return ""


def _run(command: List[str]) -> Optional[subprocess.CompletedProcess]:
    try:
        return subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError:
        return None


def _owner_for_pid(pid: int) -> str:
    try:
        uid = os.stat("/proc/%d" % pid).st_uid
        return pwd.getpwuid(uid).pw_name
    except (KeyError, OSError):
        return "unknown"


def read_snapshot() -> Snapshot:
    """Collect processes and all device health reports in one read-only snapshot."""
    info = _run(["npu-smi", "info"])
    process_text = info.stdout if info is not None else ""
    processes = tuple(
        dataclasses.replace(process, owner=_owner_for_pid(process.pid))
        for process in parse_process_table(process_text)
    )
    healthy = info is not None and info.returncode == 0

    for device in range(8):
        health = _run(["npu-smi", "info", "-t", "health", "-i", str(device)])
        if health is None or health.returncode != 0 or parse_health(health.stdout) != "OK":
            healthy = False
    return Snapshot(healthy=healthy, processes=processes)


def _status_line(snapshot: Snapshot, elapsed: float, remaining: float) -> str:
    busy = ", ".join(
        "device=%d pid=%d owner=%s" % (process.device, process.pid, process.owner)
        for process in snapshot.processes
    ) or "none"
    return "healthy=%s busy=%s elapsed=%g remaining=%g" % (
        str(snapshot.healthy).lower(), busy, elapsed, remaining)


def wait_for_idle(
    read_snapshot: Callable[[], Snapshot],
    sleep: Callable[[float], None],
    now: Callable[[], float],
    emit: Callable[[str], None],
    max_wait_seconds: float,
    poll_seconds: float,
    stable_samples: int,
) -> Snapshot:
    """Wait for the requested number of consecutive healthy, idle samples."""
    started = now()
    stable = 0
    while True:
        snapshot = read_snapshot()
        elapsed = max(0.0, now() - started)
        remaining = max(0.0, max_wait_seconds - elapsed)
        stable = stable + 1 if snapshot.idle else 0
        emit(_status_line(snapshot, elapsed, remaining))
        if stable >= stable_samples:
            return snapshot
        if elapsed >= max_wait_seconds:
            raise ResourceTimeout("NPU resources did not become idle before the deadline")
        sleep(min(poll_seconds, remaining))
