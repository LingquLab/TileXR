#!/usr/bin/env python3
"""Read NPU process and health state without changing NPU workloads."""

import dataclasses
import os
import pwd
import re
import subprocess
from typing import Callable, List, Optional, Tuple


NPU_SMI_TIMEOUT_SECONDS = 10
_EXPECTED_DEVICES = frozenset(range(8))


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


def _parse_process_table_state(text: str) -> Tuple[List[NpuProcess], bool, frozenset]:
    """Parse process rows and report whether the table is complete."""
    processes = []
    in_process_table = False
    covered_devices = set()
    for line in text.splitlines():
        if "Process id" in line and "Process name" in line:
            in_process_table = True
            continue
        if not in_process_table:
            continue

        no_processes = re.search(r"No running processes found in NPU\s+(\d+)", line)
        if no_processes:
            covered_devices.add(int(no_processes.group(1)))
            continue
        if "|" not in line:
            continue

        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue
        device_chip = cells[0].split()
        pid = cells[1]
        if not device_chip or not device_chip[0].isdecimal() or not pid.isdecimal():
            continue
        device = int(device_chip[0])
        covered_devices.add(device)
        processes.append(NpuProcess(device, int(pid), cells[2]))
    return processes, in_process_table, frozenset(covered_devices)


def parse_process_table(text: str) -> List[NpuProcess]:
    """Extract valid process rows after the npu-smi process-table heading."""
    processes, _, _ = _parse_process_table_state(text)
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
            timeout=NPU_SMI_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
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
    parsed_processes, table_recognized, covered_devices = _parse_process_table_state(process_text)
    processes = tuple(
        dataclasses.replace(process, owner=_owner_for_pid(process.pid))
        for process in parsed_processes
    )
    healthy = (
        info is not None
        and info.returncode == 0
        and table_recognized
        and _EXPECTED_DEVICES.issubset(covered_devices)
    )

    for device in range(8):
        health = _run(["npu-smi", "info", "-t", "health", "-i", str(device)])
        if health is None or health.returncode != 0 or parse_health(health.stdout) != "OK":
            healthy = False
    return Snapshot(healthy=healthy, processes=processes)


def _status_line(
    snapshot: Snapshot, elapsed: float, remaining: float, stable: int, stable_samples: int
) -> str:
    busy = ", ".join(
        "device=%d pid=%d owner=%s" % (process.device, process.pid, process.owner)
        for process in snapshot.processes
    ) or "none"
    return "healthy=%s busy=%s stable=%d/%d elapsed=%g remaining=%g" % (
        str(snapshot.healthy).lower(), busy, stable, stable_samples, elapsed, remaining)


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
        elapsed = max(0.0, now() - started)
        if elapsed >= max_wait_seconds:
            raise ResourceTimeout("NPU resources did not become idle before the deadline")
        snapshot = read_snapshot()
        elapsed = max(0.0, now() - started)
        remaining = max(0.0, max_wait_seconds - elapsed)
        candidate_stable = stable + 1 if snapshot.idle else 0
        if elapsed >= max_wait_seconds:
            emit(_status_line(snapshot, elapsed, remaining, stable, stable_samples))
            raise ResourceTimeout("NPU resources did not become idle before the deadline")
        stable = candidate_stable
        emit(_status_line(snapshot, elapsed, remaining, stable, stable_samples))
        if stable >= stable_samples:
            return snapshot
        sleep(min(poll_seconds, remaining))
