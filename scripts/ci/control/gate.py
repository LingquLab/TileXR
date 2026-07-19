#!/usr/bin/env python3
"""Trusted controller for the TileXR pull-request NPU gate."""

import argparse
import contextlib
import ctypes
import dataclasses
import fcntl
import json
import math
import os
import pathlib
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, IO, List, Mapping, Optional, Tuple

import npu_state


EXPECTED_REPOSITORY = "LingquLab/TileXR"
GITHUB_API_VERSION = "2022-11-28"
GITHUB_TIMEOUT_SECONDS = 10
BUILD_TIMEOUT_SECONDS = 7200
HARDWARE_TIMEOUT_SECONDS = 7200
QUEUE_TIMEOUT_SECONDS = 21600
QUEUE_POLL_SECONDS = 60
QUEUE_STABLE_SAMPLES = 2
FINAL_ARTIFACT_TIMEOUT_SECONDS = 30.0
MAX_FINAL_ARTIFACT_ENTRIES = 10000
MAX_FINAL_ARTIFACT_BYTES = 100 * 1024 * 1024
MAX_AUTHORITATIVE_STEP_SUMMARY_BYTES = 64 * 1024
LOCK_PATH = pathlib.Path("/home/tilexr-ci/locks/npu8.lock")
CONTROL_DIR = pathlib.Path(__file__).resolve().parent


class GateFailure(RuntimeError):
    """Base class for failures with a stable public exit classification."""


class CodeFailure(GateFailure):
    pass


class ResourceCollision(GateFailure):
    pass


class InfrastructureFailure(GateFailure):
    pass


class Cancelled(GateFailure):
    pass


class ObsoleteRun(GateFailure):
    pass


def seed_empty_case_evidence(path: pathlib.Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags, 0o600)
    except OSError as error:
        raise InfrastructureFailure(
            "could not create trusted empty cases.tsv: %s" % error
        ) from error
    os.close(descriptor)


@dataclasses.dataclass(frozen=True)
class Config:
    source: pathlib.Path
    artifacts: pathlib.Path
    expected_merge_sha: str
    repository: str
    pr_number: int


@dataclasses.dataclass
class GateReport:
    pr_number: int
    head_sha: str = "unknown"
    base_sha: str = "unknown"
    merge_sha: str = "unknown"
    build_outcome: str = "not run"
    queue_outcome: str = "not run"
    hardware_outcome: str = "not run"
    wait_seconds: float = 0.0
    execution_seconds: float = 0.0
    failure_class: str = "none"
    failure_detail: str = ""
    cleanup_checked: bool = dataclasses.field(default=False, repr=False)


class CancellationState:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()

    def wait(self, seconds):
        return self._event.wait(seconds)


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise InfrastructureFailure("invalid command line: %s" % message)


def validate_config(config: Config) -> Config:
    if config.repository != EXPECTED_REPOSITORY:
        raise InfrastructureFailure("repository must be %s" % EXPECTED_REPOSITORY)
    if config.pr_number <= 0:
        raise InfrastructureFailure("PR number must be positive")
    if not config.expected_merge_sha:
        raise InfrastructureFailure("expected merge SHA must not be empty")
    return config


def parse_config(argv=None) -> Config:
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--artifacts", type=pathlib.Path, required=True)
    parser.add_argument("--expected-merge-sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    args = parser.parse_args(argv)
    return validate_config(
        Config(
            source=args.source.resolve(),
            artifacts=args.artifacts.resolve(),
            expected_merge_sha=args.expected_merge_sha,
            repository=args.repository,
            pr_number=args.pr_number,
        )
    )


def _run_git_head(source: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", source, "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise InfrastructureFailure("could not read source merge SHA: %s" % error) from error
    return result.stdout.strip()


def verify_merge_sha(source, expected: str, run_git: Callable[[str], str] = _run_git_head) -> str:
    try:
        actual = run_git(str(source))
    except ObsoleteRun:
        raise
    except InfrastructureFailure:
        raise
    except Exception as error:
        raise InfrastructureFailure("could not read source merge SHA: %s" % error) from error
    if not isinstance(actual, str) or not actual:
        raise InfrastructureFailure("git returned an empty source merge SHA")
    if actual != expected:
        raise ObsoleteRun("source HEAD %s does not match expected merge %s" % (actual, expected))
    return actual


def policy_violation(
    phase: str, state: npu_state.Snapshot, ci_user: str = "tilexr-ci"
) -> Optional[str]:
    if not state.healthy:
        raise InfrastructureFailure("npu-smi reported an unhealthy or incomplete state")
    if phase == "build":
        if any(item.owner == ci_user for item in state.processes):
            return "ci-npu-before-lock"
        return None
    if phase == "hardware":
        if any(item.owner != ci_user for item in state.processes):
            return "foreign-npu-process"
        return None
    raise InfrastructureFailure("unknown gate phase %s" % phase)


def _enforce_policy(phase: str, state: npu_state.Snapshot):
    violation = policy_violation(phase, state)
    if violation == "ci-npu-before-lock":
        raise ResourceCollision(violation)
    if violation == "foreign-npu-process":
        raise ResourceCollision(violation)


_SENSITIVE_CHILD_KEYS = frozenset(
    (
        "GITHUB_TOKEN",
        "TILEXR_CI_GITHUB_TOKEN",
        "ACTIONS_RUNTIME_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "GITHUB_ENV",
        "GITHUB_PATH",
        "GITHUB_OUTPUT",
        "GITHUB_STATE",
        "GITHUB_STEP_SUMMARY",
    )
)


def child_environment(environ=None) -> Dict[str, str]:
    child = dict(os.environ if environ is None else environ)
    for key in _SENSITIVE_CHILD_KEYS:
        child.pop(key, None)
    child["PYTHONDONTWRITEBYTECODE"] = "1"
    return child


def phase_environment(environ, source, artifacts, pr_number: int) -> Dict[str, str]:
    phase = dict(environ)
    phase.update(
        {
            "PR_NUMBER": str(pr_number),
            "TILEXR_CI_SOURCE_DIR": str(source),
            "TILEXR_CI_ARTIFACT_DIR": str(artifacts),
            "TILEXR_CANN_HOME": "/home/tilexr-ci/toolchains/cann/9.1.0",
        }
    )
    return phase


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # macOS can briefly report EPERM while an orphaned group is being reaped.
        return True


def _linux_session_members(pgid: int) -> Tuple[int, ...]:
    members = []
    expected_uid = os.geteuid()
    proc_root = pathlib.Path("/proc")
    try:
        entries = list(proc_root.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            if entry.stat().st_uid != expected_uid:
                continue
            stat_text = (entry / "stat").read_text(encoding="ascii")
            fields = stat_text.rsplit(")", 1)[1].split()
            process_group = int(fields[2])
            session = int(fields[3])
        except (IndexError, OSError, ValueError):
            continue
        if process_group == pgid and session == pgid:
            members.append(int(entry.name))
    return tuple(members)


@dataclasses.dataclass(frozen=True)
class ProcessRecord:
    pid: int
    parent_pid: int
    process_group: int
    session: int
    start_time: int
    uid: int

    @property
    def identity(self):
        return (self.pid, self.start_time)


def _scan_linux_processes() -> Dict[int, ProcessRecord]:
    records = {}
    try:
        entries = list(pathlib.Path("/proc").iterdir())
    except OSError:
        return records
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            uid = entry.stat().st_uid
            fields = (entry / "stat").read_text(encoding="ascii").rsplit(")", 1)[1].split()
            record = ProcessRecord(
                pid=int(entry.name),
                parent_pid=int(fields[1]),
                process_group=int(fields[2]),
                session=int(fields[3]),
                start_time=int(fields[19]),
                uid=uid,
            )
        except (IndexError, OSError, ValueError):
            continue
        records[record.pid] = record
    return records


def _enable_linux_subreaper():
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error_number = ctypes.get_errno()
        raise InfrastructureFailure(
            "could not enable Linux child subreaper: %s" % os.strerror(error_number)
        )


class LinuxProcessBoundary:
    def __init__(self, *, parent_pid, baseline, scan, signal_identity=None):
        self.parent_pid = parent_pid
        self.baseline = set(baseline)
        self.scan = scan
        self._injected_signaler = signal_identity is not None
        self.signal_identity = signal_identity or self._signal_record
        self.root_pid = None
        self.root_identity = None
        self.known = {}
        self.seen_starts = {}
        self.pidfds = {}
        self._before_step = None

    @classmethod
    def prepare(cls):
        _enable_linux_subreaper()
        if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
            raise InfrastructureFailure(
                "Linux pidfd support is required for tracked process cleanup"
            )
        try:
            capability_fd = os.pidfd_open(os.getpid(), 0)
            os.close(capability_fd)
        except OSError as error:
            raise InfrastructureFailure(
                "Linux pidfd cleanup capability is unavailable: %s" % error
            ) from error
        records = _scan_linux_processes()
        parent_pid = os.getpid()
        baseline = (
            record.identity
            for record in records.values()
            if record.parent_pid == parent_pid
        )
        return cls(
            parent_pid=parent_pid,
            baseline=baseline,
            scan=_scan_linux_processes,
        )

    def _scan(self):
        if self._before_step is not None:
            self._before_step()
        return self.scan()

    def _remember(self, record, *, lineage_verified=False):
        previous_start = self.seen_starts.get(record.pid)
        if previous_start is not None and previous_start != record.start_time:
            if record.pid == self.root_pid or not lineage_verified:
                return False
            previous_identity = (record.pid, previous_start)
            self.known.pop(previous_identity, None)
            previous_fd = self.pidfds.pop(previous_identity, None)
            if previous_fd is not None:
                try:
                    os.close(previous_fd)
                except OSError:
                    pass
        if record.identity in self.baseline:
            return False
        self.seen_starts[record.pid] = record.start_time
        self.known[record.identity] = record
        if record.identity not in self.pidfds and not self._injected_signaler:
            if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
                raise InfrastructureFailure(
                    "Linux pidfd support is required for tracked process cleanup"
                )
            fd = None
            try:
                fd = os.pidfd_open(record.pid, 0)
                current = self._scan().get(record.pid)
                if current is None or current.identity != record.identity:
                    os.close(fd)
                    fd = None
                    self.known.pop(record.identity, None)
                    return False
                else:
                    self.pidfds[record.identity] = fd
                    fd = None
            except ProcessLookupError:
                self.known.pop(record.identity, None)
                return False
            except OSError as error:
                raise InfrastructureFailure(
                    "could not open pidfd for tracked process %d: %s"
                    % (record.pid, error)
                ) from error
            finally:
                if fd is not None:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
        return True

    def attach(
        self,
        root_pid,
        child=None,
        *,
        deadline=None,
        now=time.monotonic,
        before_step=None,
    ):
        self.root_pid = root_pid
        self.root_identity = None
        self._before_step = before_step
        try:
            child_exited = child is not None and child.poll() is not None
            records = self._scan()
            root = records.get(root_pid)
            if child_exited:
                root = None
            elif root is None:
                raise InfrastructureFailure(
                    "spawned process root identity is missing from /proc"
                )
            elif child is not None and root.parent_pid != self.parent_pid:
                raise InfrastructureFailure(
                    "spawned process root is not a direct controller child"
                )
            elif not self._remember(root):
                raise InfrastructureFailure(
                    "could not pin spawned process root identity"
                )
            else:
                self.root_identity = root.identity
            self._refresh(records)
            if deadline is not None and now() >= deadline and before_step is not None:
                before_step()
        finally:
            self._before_step = None

    def _refresh(self, records=None):
        records = self._scan() if records is None else records
        same_uid = {
            pid: record
            for pid, record in records.items()
            if record.uid == os.geteuid()
        }
        if self.root_identity is not None and self.root_pid in same_uid:
            root = same_uid[self.root_pid]
            if root.identity == self.root_identity:
                self._remember(root)

        tracked_pids = {
            record.pid
            for identity, record in self.known.items()
            if same_uid.get(record.pid) is not None
            and same_uid[record.pid].identity == identity
        }
        changed = True
        while changed:
            changed = False
            for record in same_uid.values():
                if record.parent_pid in tracked_pids and record.pid not in tracked_pids:
                    if self._remember(record, lineage_verified=True):
                        tracked_pids.add(record.pid)
                        changed = True

        for record in same_uid.values():
            if (
                record.pid != self.root_pid
                and record.parent_pid == self.parent_pid
                and record.identity not in self.baseline
            ):
                self._remember(record, lineage_verified=True)

        return [
            current
            for identity, record in self.known.items()
            for current in (same_uid.get(record.pid),)
            if current is not None and current.identity == identity
        ]

    def _signal_record(self, record, signum):
        fd = self.pidfds.get(record.identity)
        if fd is None:
            raise InfrastructureFailure(
                "tracked process %d has no pidfd identity" % record.pid
            )
        try:
            signal.pidfd_send_signal(fd, signum, None, 0)
        except ProcessLookupError:
            pass

    def signal_tracked(self, signum):
        records = self._refresh()
        for record in records:
            self.signal_identity(record, signum)
        return records

    def _reap_adopted(self):
        for record in tuple(self.known.values()):
            if record.pid == self.root_pid:
                continue
            try:
                os.waitpid(record.pid, os.WNOHANG)
            except (ChildProcessError, OSError):
                pass

    def _close_pidfds(self):
        for fd in self.pidfds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self.pidfds.clear()

    def abort(self, child, *, now=time.monotonic, sleep=time.sleep):
        errors = []

        def cleanup_sleep(seconds):
            # Caller-provided sleeps may be cancellation-aware; abort must still kill.
            try:
                sleep(seconds)
            except Exception:
                pass

        def signal_records(records, signum):
            for record in records:
                try:
                    self.signal_identity(record, signum)
                except BaseException as error:
                    errors.append(error)

        try:
            records = list(self.known.values())
            try:
                records = self._refresh()
            except BaseException as error:
                errors.append(error)
            signal_records(records, signal.SIGTERM)

            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                except BaseException as error:
                    errors.append(error)
            cleanup_sleep(0.1)

            try:
                records = self._refresh()
            except BaseException as error:
                errors.append(error)
                records = list(self.known.values())
            signal_records(records, signal.SIGKILL)
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except BaseException as error:
                    errors.append(error)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired as error:
                errors.append(error)

            verification_deadline = now() + 2.0
            live = []
            while True:
                self._reap_adopted()
                try:
                    live = self._refresh()
                except BaseException as error:
                    errors.append(error)
                    break
                signal_records(live, signal.SIGKILL)
                if not live or now() >= verification_deadline:
                    break
                cleanup_sleep(
                    min(0.05, max(0.0, verification_deadline - now()))
                )
            if live:
                errors.append(
                    InfrastructureFailure("tracked processes remain after abort")
                )
        finally:
            self._close_pidfds()
        if errors:
            raise InfrastructureFailure(
                "Linux process-boundary abort was incomplete: %s"
                % "; ".join(str(error) for error in errors)
            )

    def terminate(self, child, *, now=time.monotonic, sleep=time.sleep):
        try:
            term_signalled = set()
            for record in self.signal_tracked(signal.SIGTERM):
                term_signalled.add(record.identity)
            deadline = now() + 10.0
            while True:
                child.poll()
                self._reap_adopted()
                live = self._refresh()
                for record in live:
                    if record.identity not in term_signalled:
                        self.signal_identity(record, signal.SIGTERM)
                        term_signalled.add(record.identity)
                if not live or now() >= deadline:
                    break
                sleep(min(0.1, max(0.0, deadline - now())))
            live = self._refresh()
            kill_signalled = set()
            for record in live:
                self.signal_identity(record, signal.SIGKILL)
                kill_signalled.add(record.identity)
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            verification_deadline = now() + 2.0
            while True:
                self._reap_adopted()
                live = self._refresh()
                for record in live:
                    if record.identity not in kill_signalled:
                        self.signal_identity(record, signal.SIGKILL)
                        kill_signalled.add(record.identity)
                if not live or now() >= verification_deadline:
                    break
                sleep(min(0.05, max(0.0, verification_deadline - now())))
            if live:
                details = ", ".join(
                    "%d:%d" % record.identity for record in live
                )
                raise InfrastructureFailure(
                    "tracked processes remain after SIGKILL: %s" % details
                )
        finally:
            self._close_pidfds()


def terminate_group(
    child,
    killpg=os.killpg,
    *,
    group_exists: Callable[[int], bool] = _process_group_exists,
    session_members: Optional[Callable[[int], Tuple[int, ...]]] = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
):
    """Stop only the process group created for the supplied child."""
    boundary = getattr(child, "_tilexr_process_boundary", None)
    if isinstance(boundary, LinuxProcessBoundary):
        boundary.terminate(child, now=now, sleep=sleep)
        return

    leader_live = child.poll() is None

    def members():
        if session_members is not None:
            return tuple(session_members(child.pid))
        if group_exists is not _process_group_exists:
            return (child.pid,) if group_exists(child.pid) else ()
        if sys.platform.startswith("linux"):
            return _linux_session_members(child.pid)
        return (child.pid,) if group_exists(child.pid) else ()

    if not leader_live and not members():
        child.wait()
        return
    try:
        killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = now() + 10.0
    while True:
        leader_live = child.poll() is None
        if (not leader_live and not members()) or now() >= deadline:
            break
        sleep(min(0.1, max(0.0, deadline - now())))
    leader_live = child.poll() is None
    if leader_live or members():
        try:
            killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    child.wait()


def cancellation_aware_sleep(seconds: float, cancellation: CancellationState):
    if cancellation.cancelled or cancellation.wait(max(0.0, seconds)):
        raise Cancelled("gate was cancelled")


def run_phase(
    phase: str,
    script: pathlib.Path,
    source: pathlib.Path,
    artifacts: pathlib.Path,
    env: Mapping[str, str],
    *,
    timeout_seconds: float,
    read_snapshot: Callable[[], npu_state.Snapshot] = npu_state.read_snapshot,
    read_snapshot_with_deadline: Optional[Callable[[float], npu_state.Snapshot]] = None,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    cancellation: Optional[CancellationState] = None,
    popen: Callable = subprocess.Popen,
    terminate: Callable = terminate_group,
    process_boundary_factory: Optional[Callable] = None,
) -> int:
    cancellation = cancellation or CancellationState()
    command = [str(script), str(source), str(artifacts)]

    def command_checkpoint():
        if cancellation.cancelled:
            raise Cancelled("gate was cancelled during %s NPU inspection" % phase)

    if cancellation.cancelled:
        raise Cancelled("gate was cancelled before %s" % phase)
    if read_snapshot is npu_state.read_snapshot:
        preflight_state = read_snapshot(before_command=command_checkpoint)
    else:
        preflight_state = read_snapshot()
    _enforce_policy(phase, preflight_state)
    process_boundary = None
    if process_boundary_factory is not None:
        process_boundary = process_boundary_factory()
    elif sys.platform.startswith("linux") and popen is subprocess.Popen:
        process_boundary = LinuxProcessBoundary.prepare()
    started = now()
    deadline = started + timeout_seconds

    def attach_checkpoint():
        if cancellation.cancelled:
            raise Cancelled("gate was cancelled while attaching the %s phase" % phase)
        if now() >= deadline:
            raise CodeFailure("%s phase exceeded %.0f seconds" % (phase, timeout_seconds))

    try:
        child = popen(
            command,
            cwd=str(source),
            env=dict(env),
            start_new_session=True,
        )
    except OSError as error:
        raise InfrastructureFailure("could not start trusted %s phase: %s" % (phase, error)) from error
    if process_boundary is not None:
        child._tilexr_process_boundary = process_boundary
        try:
            process_boundary.attach(
                child.pid,
                child=child,
                deadline=deadline,
                now=now,
                before_step=attach_checkpoint,
            )
            attach_checkpoint()
        except BaseException as attach_error:
            try:
                process_boundary.abort(child, now=now, sleep=sleep)
            except BaseException as termination_error:
                raise _tracked_termination_failure(
                    attach_error, termination_error, collector=False
                ) from termination_error
            if isinstance(attach_error, GateFailure):
                raise
            raise InfrastructureFailure(
                "could not attach spawned phase to Linux process boundary: %s"
                % attach_error
            ) from attach_error
    else:
        try:
            attach_checkpoint()
        except BaseException as attach_error:
            try:
                terminate(child)
            except BaseException as termination_error:
                raise _tracked_termination_failure(
                    attach_error, termination_error, collector=False
                ) from termination_error
            raise

    def monitored_snapshot():
        if read_snapshot_with_deadline is not None:
            return read_snapshot_with_deadline(deadline)
        if read_snapshot is npu_state.read_snapshot:
            return read_snapshot(
                deadline=deadline, now=now, before_command=command_checkpoint
            )
        return read_snapshot()

    def classify_exit(returncode):
        if returncode != 0:
            raise CodeFailure("%s phase exited with status %d" % (phase, returncode))
        return returncode

    primary_error = None
    try:
        while True:
            if cancellation.cancelled:
                raise Cancelled("gate was cancelled during %s" % phase)

            if now() >= deadline:
                raise CodeFailure("%s phase exceeded %.0f seconds" % (phase, timeout_seconds))
            returncode = child.poll()
            if returncode is not None:
                return classify_exit(returncode)
            state = monitored_snapshot()
            if now() >= deadline:
                raise CodeFailure("%s phase exceeded %.0f seconds" % (phase, timeout_seconds))
            returncode = child.poll()
            if returncode is not None:
                return classify_exit(returncode)
            _enforce_policy(phase, state)

            elapsed = max(0.0, now() - started)
            if elapsed >= timeout_seconds:
                raise CodeFailure("%s phase exceeded %.0f seconds" % (phase, timeout_seconds))
            sleep(min(10.0, max(0.0, timeout_seconds - elapsed)))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            terminate(child)
        except BaseException as termination_error:
            raise _tracked_termination_failure(
                primary_error, termination_error, collector=False
            ) from termination_error


def fetch_current_merge_sha(
    pr_number: int,
    token: str,
    *,
    urlopen: Callable = urllib.request.urlopen,
) -> str:
    if not token:
        raise InfrastructureFailure("TILEXR_CI_GITHUB_TOKEN is required")
    url = "https://api.github.com/repos/%s/pulls/%d" % (EXPECTED_REPOSITORY, pr_number)
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": "Bearer %s" % token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "User-Agent": "TileXR-trusted-pr-gate",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=GITHUB_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, UnicodeError, urllib.error.HTTPError) as error:
        raise InfrastructureFailure("could not fetch current pull request: %s" % error) from error
    if not isinstance(payload, dict):
        raise InfrastructureFailure("GitHub pull-request response was not an object")
    merge_sha = payload.get("merge_commit_sha")
    if not isinstance(merge_sha, str) or not merge_sha:
        raise InfrastructureFailure("GitHub pull-request response has no merge_commit_sha")
    return merge_sha


@contextlib.contextmanager
def npu_lock(
    path: pathlib.Path = LOCK_PATH,
    *,
    cancellation: Optional[CancellationState] = None,
    max_wait_seconds: float = QUEUE_TIMEOUT_SECONDS,
    now: Callable[[], float] = time.monotonic,
    sleep: Optional[Callable[[float], None]] = None,
):
    cancellation = cancellation or CancellationState()
    lock_sleep = sleep or (lambda seconds: cancellation_aware_sleep(seconds, cancellation))
    started = now()
    acquired = False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+") as lock_file:
            while not acquired:
                if cancellation.cancelled:
                    raise Cancelled("gate was cancelled while waiting for the NPU lock")
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    elapsed = max(0.0, now() - started)
                    if elapsed >= max_wait_seconds:
                        raise npu_state.ResourceTimeout(
                            "NPU lock was not acquired before the resource deadline"
                        )
                    lock_sleep(min(1.0, max_wait_seconds - elapsed))
            try:
                yield
            finally:
                if acquired:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except OSError as error:
        raise InfrastructureFailure("could not use NPU lock %s: %s" % (path, error)) from error


def orchestrate(
    config: Config,
    *,
    control_dir: pathlib.Path,
    token: str,
    cancellation: CancellationState,
    report: Optional[GateReport] = None,
    run_phase_fn: Callable = run_phase,
    wait_for_idle_fn: Callable = npu_state.wait_for_idle,
    read_snapshot_fn: Callable = npu_state.read_snapshot,
    fetch_merge_sha_fn: Callable = fetch_current_merge_sha,
    verify_merge_sha_fn: Callable = verify_merge_sha,
    lock_fn: Callable = npu_lock,
    sleep_fn: Optional[Callable[[float], None]] = None,
    emit: Callable[[str], None] = print,
    resource_now: Optional[Callable[[], float]] = None,
    queue_snapshot_fn: Optional[Callable] = None,
):
    validate_config(config)
    report = report or GateReport(pr_number=config.pr_number, merge_sha=config.expected_merge_sha)
    if cancellation.cancelled:
        raise Cancelled("gate was cancelled before source verification")
    verify_merge_sha_fn(config.source, config.expected_merge_sha)
    if cancellation.cancelled:
        raise Cancelled("gate was cancelled before build")
    env = phase_environment(
        child_environment(), config.source, config.artifacts, config.pr_number
    )
    phase_sleep = sleep_fn or (lambda seconds: cancellation_aware_sleep(seconds, cancellation))
    resource_clock = resource_now or time.monotonic

    report.build_outcome = "running"
    try:
        run_phase_fn(
            "build",
            control_dir / "build_blue.sh",
            config.source,
            config.artifacts,
            env,
            timeout_seconds=BUILD_TIMEOUT_SECONDS,
            read_snapshot=read_snapshot_fn,
            now=time.monotonic,
            sleep=phase_sleep,
            cancellation=cancellation,
        )
    except Exception:
        report.build_outcome = "failed"
        raise
    report.build_outcome = "passed"

    report.queue_outcome = "waiting"
    queue_started = resource_clock()
    resource_deadline = queue_started + QUEUE_TIMEOUT_SECONDS
    queue_finished = False
    try:
        lock_budget = max(0.0, resource_deadline - resource_clock())
        if lock_budget <= 0:
            raise npu_state.ResourceTimeout(
                "NPU resource budget was exhausted before lock acquisition"
            )
        with lock_fn(
            cancellation=cancellation,
            max_wait_seconds=lock_budget,
            now=resource_clock,
            sleep=phase_sleep,
        ):
            locked_error = None
            try:
                remaining_budget = max(
                    0.0, QUEUE_TIMEOUT_SECONDS - (resource_clock() - queue_started)
                )
                if remaining_budget <= 0:
                    raise npu_state.ResourceTimeout(
                        "NPU resource budget was exhausted while acquiring the lock"
                    )
                def queue_snapshot():
                    if queue_snapshot_fn is not None:
                        return queue_snapshot_fn(
                            deadline=resource_deadline, now=resource_clock
                        )

                    def command_checkpoint():
                        if cancellation.cancelled:
                            raise Cancelled(
                                "gate was cancelled during queued NPU inspection"
                            )

                    return npu_state.read_snapshot(
                        deadline=resource_deadline,
                        now=resource_clock,
                        before_command=command_checkpoint,
                    )

                try:
                    wait_for_idle_fn(
                        read_snapshot=queue_snapshot,
                        sleep=phase_sleep,
                        now=resource_clock,
                        emit=emit,
                        max_wait_seconds=remaining_budget,
                        poll_seconds=QUEUE_POLL_SECONDS,
                        stable_samples=QUEUE_STABLE_SAMPLES,
                    )
                except npu_state.ResourceTimeout:
                    report.queue_outcome = "timed out"
                    raise
                except Exception:
                    report.queue_outcome = "failed"
                    raise
                finally:
                    report.wait_seconds = max(0.0, resource_clock() - queue_started)
                    queue_finished = True
                report.queue_outcome = "passed"

                if cancellation.cancelled:
                    raise Cancelled("gate was cancelled after resource wait")
                current_merge = fetch_merge_sha_fn(config.pr_number, token)
                if cancellation.cancelled:
                    raise Cancelled("gate was cancelled during merge revalidation")
                if current_merge != config.expected_merge_sha:
                    raise ObsoleteRun(
                        "pull request merge changed from %s to %s"
                        % (config.expected_merge_sha, current_merge)
                    )

                report.hardware_outcome = "running"
                hardware_started = time.monotonic()
                try:
                    run_phase_fn(
                        "hardware",
                        control_dir / "run_hardware.sh",
                        config.source,
                        config.artifacts,
                        env,
                        timeout_seconds=HARDWARE_TIMEOUT_SECONDS,
                        read_snapshot=read_snapshot_fn,
                        now=time.monotonic,
                        sleep=phase_sleep,
                        cancellation=cancellation,
                    )
                except Exception:
                    report.hardware_outcome = "failed"
                    raise
                finally:
                    report.execution_seconds = max(0.0, time.monotonic() - hardware_started)
                report.hardware_outcome = "passed"
            except BaseException as error:
                locked_error = error
                raise
            finally:
                report.cleanup_checked = True
                try:
                    verify_final_cleanup(read_snapshot_fn)
                except BaseException as cleanup_error:
                    raise prefer_cleanup_failure(locked_error, cleanup_error) from cleanup_error
    except npu_state.ResourceTimeout:
        if not queue_finished:
            report.queue_outcome = "timed out"
        raise
    except BaseException:
        if not queue_finished and report.queue_outcome == "waiting":
            report.queue_outcome = "failed"
        raise
    finally:
        if not queue_finished:
            report.wait_seconds = max(0.0, resource_clock() - queue_started)
    return report


def parse_cases(path: pathlib.Path) -> Tuple[List[Tuple[str, str, int, float]], List[str]]:
    cases = []
    diagnostics = []
    if not path.exists():
        return cases, ["cases.tsv is missing"]
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return cases, ["could not read cases.tsv: %s" % error]
    names = set()
    for number, line in enumerate(lines, 1):
        fields = line.split("\t")
        try:
            if len(fields) != 4 or not fields[0].strip() or not fields[1].strip():
                raise ValueError("expected four tab-separated fields")
            name = fields[0].strip()
            status = fields[1].strip().upper()
            exit_code = int(fields[2])
            elapsed = float(fields[3])
            if name in names:
                raise ValueError("duplicate case name %s" % name)
            if status not in ("PASS", "FAIL"):
                raise ValueError("unsupported result status %s" % status)
            if exit_code < 0:
                raise ValueError("exit code must be non-negative")
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("elapsed seconds must be finite and non-negative")
            if status == "PASS" and exit_code != 0:
                raise ValueError("passing case must have exit code zero")
            if status == "FAIL" and exit_code == 0:
                raise ValueError("failing case must have a nonzero exit code")
            names.add(name)
            cases.append((name, status, exit_code, elapsed))
        except ValueError as error:
            diagnostics.append("line %d: %s" % (number, error))
    return cases, diagnostics


def validate_success_cases(path: pathlib.Path):
    cases, diagnostics = parse_cases(path)
    if diagnostics:
        raise InfrastructureFailure("invalid case evidence: %s" % "; ".join(diagnostics))
    if not cases:
        raise InfrastructureFailure("case evidence contains no cases")
    failed = [case[0] for case in cases if case[1] != "PASS"]
    if failed:
        raise InfrastructureFailure(
            "successful gate contains failed cases: %s" % ", ".join(failed)
        )
    return cases


def _summary_text(report: GateReport, cases, diagnostics) -> str:
    passed = [case for case in cases if case[1] == "PASS"]
    failed = [case for case in cases if case not in passed]
    failed_names = ", ".join(case[0] for case in failed) or "none"
    detail = report.failure_detail or "none"
    return "\n".join(
        (
            "# TileXR trusted PR gate",
            "",
            "- PR #%d" % report.pr_number,
            "- Head SHA: `%s`" % report.head_sha,
            "- Base SHA: `%s`" % report.base_sha,
            "- Merge SHA: `%s`" % report.merge_sha,
            "- Build outcome: %s" % report.build_outcome,
            "- Queue outcome: %s" % report.queue_outcome,
            "- Hardware outcome: %s" % report.hardware_outcome,
            "- Queue wait seconds: %.1f" % report.wait_seconds,
            "- Hardware execution seconds: %.2f" % report.execution_seconds,
            "- Passed cases: %d" % len(passed),
            "- Failed cases: %d" % len(failed),
            "- Failed names: %s" % failed_names,
            "- Failure class: %s" % report.failure_class,
            "- Failure detail: %s" % detail,
            "- Malformed cases.tsv rows: %d" % len(diagnostics),
            "",
            "UDMA data plane: out of scope on 910B3",
            "",
            "Multi-host validation: out of scope for this gate",
            "",
            "## Diagnostics",
            "",
            *("- %s" % item for item in diagnostics),
        )
    ).rstrip()


def write_summary(report: GateReport, artifacts: pathlib.Path, step_summary: Optional[IO[str]] = None) -> str:
    try:
        artifacts.mkdir(parents=True, exist_ok=True)
        cases, diagnostics = parse_cases(artifacts / "cases.tsv")
        text = _summary_text(report, cases, diagnostics)
        (artifacts / "summary.md").write_text(text, encoding="utf-8")
        if step_summary is not None:
            step_summary.write(text + "\n")
            step_summary.flush()
        return text
    except OSError as error:
        raise InfrastructureFailure("could not write gate summary: %s" % error) from error


def _authoritative_step_summary_text(
    report: GateReport, artifacts: pathlib.Path
) -> str:
    uploaded_summary = "Uploaded artifact summary unavailable."
    summary_path = artifacts / "summary.md"
    try:
        summary_status = os.lstat(str(summary_path))
        if (
            stat.S_ISREG(summary_status.st_mode)
            and summary_status.st_size <= MAX_AUTHORITATIVE_STEP_SUMMARY_BYTES
        ):
            uploaded_summary = summary_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        pass

    detail = report.failure_detail or "none"
    if len(detail) > 8192:
        detail = detail[:8176] + " ... [truncated]"
    return (
        "## Authoritative final gate result\n\n"
        "- Failure class: %s\n"
        "- Failure detail: %s\n\n"
        "%s\n"
        % (report.failure_class, detail, uploaded_summary)
    )


def _append_authoritative_step_text(path: pathlib.Path, text: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as step_summary:
            step_summary.write(text)
            step_summary.flush()
    except OSError as error:
        raise InfrastructureFailure(
            "could not append authoritative GitHub step summary: %s" % error
        ) from error


def append_authoritative_step_summary(
    path: pathlib.Path, report: GateReport, artifacts: pathlib.Path
) -> None:
    _append_authoritative_step_text(
        path, _authoritative_step_summary_text(report, artifacts)
    )


def step_summary_size(path: pathlib.Path) -> int:
    try:
        path_status = os.lstat(str(path))
    except FileNotFoundError:
        return 0
    except OSError as error:
        raise InfrastructureFailure(
            "could not inspect GitHub step summary: %s" % error
        ) from error
    if not stat.S_ISREG(path_status.st_mode):
        raise InfrastructureFailure(
            "GitHub step summary path is not a regular file: %s" % path
        )
    return path_status.st_size


def rollback_step_summary(path: pathlib.Path, size: int) -> None:
    descriptor = -1
    flags = os.O_WRONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
        path_status = os.lstat(str(path))
        opened_status = os.fstat(descriptor)
        if not stat.S_ISREG(path_status.st_mode) or (
            path_status.st_dev,
            path_status.st_ino,
        ) != (opened_status.st_dev, opened_status.st_ino):
            raise InfrastructureFailure(
                "GitHub step summary changed during rollback"
            )
        if opened_status.st_size < size:
            raise InfrastructureFailure(
                "GitHub step summary shrank before rollback"
            )
        os.ftruncate(descriptor, size)
        os.fsync(descriptor)
    except OSError as error:
        raise InfrastructureFailure(
            "could not roll back GitHub step summary: %s" % error
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _path_has_control_characters(path: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in path)


def _unsafe_artifact_failure(message: str, path: pathlib.Path) -> InfrastructureFailure:
    error = InfrastructureFailure(message)
    error.unsafe_artifact_path = path
    return error


def reset_collector_manifest_boundary(artifacts: pathlib.Path) -> None:
    manifest = artifacts / "manifest.txt"
    try:
        manifest_status = os.lstat(str(manifest))
    except FileNotFoundError:
        return
    except OSError as error:
        raise InfrastructureFailure(
            "could not inspect stale collector manifest: %s" % error
        ) from error
    if stat.S_ISDIR(manifest_status.st_mode):
        raise InfrastructureFailure(
            "stale collector manifest path is a directory: %s" % manifest
        )
    try:
        os.unlink(str(manifest))
    except OSError as error:
        raise InfrastructureFailure(
            "could not remove stale collector manifest: %s" % error
        ) from error


def _disable_artifact_upload_path(artifacts: pathlib.Path) -> bool:
    try:
        path_status = os.lstat(str(artifacts))
    except FileNotFoundError:
        return True
    except OSError:
        return False
    if not stat.S_ISDIR(path_status.st_mode):
        try:
            os.unlink(str(artifacts))
            return True
        except OSError:
            return False
    descriptor = -1
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(artifacts), flags)
        opened_status = os.fstat(descriptor)
        if (opened_status.st_dev, opened_status.st_ino) != (
            path_status.st_dev,
            path_status.st_ino,
        ):
            return False
        os.fchmod(descriptor, 0)
        return True
    except OSError:
        return False
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def quarantine_artifact_upload_root(artifacts: pathlib.Path) -> pathlib.Path:
    quarantine = None
    try:
        quarantine = pathlib.Path(
            tempfile.mkdtemp(
                prefix=".tilexr-quarantined-artifacts-", dir=str(artifacts.parent)
            )
        )
        os.replace(str(artifacts), str(quarantine / "upload-root"))
        return quarantine
    except OSError as error:
        if quarantine is not None:
            try:
                os.rmdir(str(quarantine))
            except OSError:
                pass
        disabled = _disable_artifact_upload_path(artifacts)
        raise InfrastructureFailure(
            "could not quarantine artifact upload root%s: %s"
            % ("; upload traversal disabled" if disabled else "", error)
        ) from error


def _create_minimal_evidence_file(path: pathlib.Path, contents: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = -1
    try:
        descriptor = os.open(str(path), flags, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise InfrastructureFailure(
            "could not create minimal artifact evidence %s: %s" % (path.name, error)
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def recover_minimal_artifact_upload(
    artifacts: pathlib.Path,
    report: GateReport,
    failure: BaseException,
) -> pathlib.Path:
    quarantine = quarantine_artifact_upload_root(artifacts)
    try:
        os.mkdir(str(artifacts), 0o700)
    except OSError as error:
        try:
            os.replace(
                str(artifacts), str(quarantine / "failed-recreated-upload-root")
            )
        except OSError:
            _disable_artifact_upload_path(artifacts)
        raise InfrastructureFailure(
            "artifact root was quarantined but could not be recreated: %s" % error
        ) from error

    selected_detail = report.failure_detail or str(failure)
    diagnostic = "artifact upload root quarantined as sibling %s" % quarantine.name
    suffix = "; %s" % diagnostic
    marker = " ... [truncated]"
    if len(selected_detail) + len(suffix) > 8192:
        selected_detail = selected_detail[: 8192 - len(suffix) - len(marker)] + marker
    combined_detail = selected_detail + suffix
    report.failure_class = failure_class_for(failure)
    report.failure_detail = combined_detail
    summary = (
        "# TileXR PR Gate\n\n"
        "Terminal artifact recovery\n\n"
        "- PR: #%d\n"
        "- Failure class: %s\n"
        "- Failure detail: %s\n"
        % (report.pr_number, report.failure_class, report.failure_detail)
    ).encode("utf-8")
    rejection = (
        "artifact-root\tquarantined-after-terminal-finalization-failure\n"
    ).encode("utf-8")
    contents = {
        "artifact-rejections.txt": rejection,
        "cases.tsv": b"",
        "summary.md": summary,
    }
    try:
        for name in sorted(contents):
            _create_minimal_evidence_file(artifacts / name, contents[name])
        manifest = "".join(
            "%s\t%d\n" % (name, len(contents[name])) for name in sorted(contents)
        ).encode("utf-8")
        _create_minimal_evidence_file(artifacts / "manifest.txt", manifest)
    except BaseException as error:
        containment = "partial upload root quarantined"
        try:
            os.replace(
                str(artifacts), str(quarantine / "failed-minimal-evidence-root")
            )
        except OSError as quarantine_error:
            try:
                disabled = _disable_artifact_upload_path(artifacts)
                disable_detail = (
                    "upload traversal disabled"
                    if disabled
                    else "upload traversal could not be disabled"
                )
            except BaseException as disable_error:
                disable_detail = "upload traversal disable failed: %s" % _as_gate_failure(
                    disable_error
                )
            containment = "second quarantine failed: %s; %s" % (
                quarantine_error,
                disable_detail,
            )
        raise InfrastructureFailure(
            "could not create bounded minimal artifact evidence: %s; %s"
            % (_as_gate_failure(error), containment)
        ) from error
    return quarantine


def refresh_final_manifest(
    artifacts: pathlib.Path,
    *,
    deadline: Optional[float] = None,
    now: Callable[[], float] = time.monotonic,
    cancellation: Optional[CancellationState] = None,
    cancellation_accounted: bool = False,
    max_entries: int = MAX_FINAL_ARTIFACT_ENTRIES,
) -> None:
    if deadline is None:
        deadline = now() + FINAL_ARTIFACT_TIMEOUT_SECONDS
    if max_entries < 1 or max_entries > MAX_FINAL_ARTIFACT_ENTRIES:
        raise InfrastructureFailure(
            "invalid final artifact entry limit: %d" % max_entries
        )

    def checkpoint() -> None:
        if (
            cancellation is not None
            and cancellation.cancelled
            and not cancellation_accounted
        ):
            raise Cancelled(
                "gate was cancelled during final artifact manifest refresh"
            )
        if now() >= deadline:
            raise InfrastructureFailure(
                "final artifact manifest refresh exceeded its absolute deadline"
            )

    manifest = artifacts / "manifest.txt"
    checkpoint()
    try:
        root_status = os.lstat(str(artifacts))
        if not stat.S_ISDIR(root_status.st_mode):
            raise InfrastructureFailure(
                "artifact root is not a real directory: %s" % artifacts
            )
        checkpoint()
        manifest_status = os.lstat(str(manifest))
        if not stat.S_ISREG(manifest_status.st_mode):
            raise _unsafe_artifact_failure(
                "collector manifest boundary is not a regular file: %s" % manifest,
                manifest,
            )
    except FileNotFoundError as error:
        raise InfrastructureFailure(
            "collector did not produce manifest.txt"
        ) from error
    except OSError as error:
        raise InfrastructureFailure(
            "could not validate collector manifest boundary: %s" % error
        ) from error

    boundary_identity = (manifest_status.st_dev, manifest_status.st_ino)
    rows = []
    pending = [artifacts]
    entry_count = 0
    manifest_seen = False

    try:
        while pending:
            checkpoint()
            current_path = pending.pop()
            with os.scandir(str(current_path)) as scanner:
                for entry in scanner:
                    checkpoint()
                    entry_count += 1
                    if entry_count > max_entries:
                        raise InfrastructureFailure(
                            "final artifact tree exceeds the %d entry limit"
                            % max_entries
                        )
                    path = pathlib.Path(entry.path)
                    relative = path.relative_to(artifacts).as_posix()
                    if _path_has_control_characters(relative):
                        raise _unsafe_artifact_failure(
                            "artifact path contains control characters: %r" % relative,
                            path,
                        )
                    path_status = entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(path_status.st_mode):
                        pending.append(path)
                        continue
                    if relative == "manifest.txt":
                        if not stat.S_ISREG(path_status.st_mode) or (
                            path_status.st_dev,
                            path_status.st_ino,
                        ) != boundary_identity:
                            raise _unsafe_artifact_failure(
                                "collector manifest boundary changed during finalization",
                                manifest,
                            )
                        manifest_seen = True
                        continue
                    if not stat.S_ISREG(path_status.st_mode):
                        raise _unsafe_artifact_failure(
                            "artifact tree contains a nonregular entry: %s" % relative,
                            path,
                        )
                    if path_status.st_size > MAX_FINAL_ARTIFACT_BYTES:
                        raise _unsafe_artifact_failure(
                            "final artifact exceeds 100 MiB: %s" % relative,
                            path,
                        )
                    rows.append((relative, path_status.st_size))
    except OSError as error:
        raise InfrastructureFailure(
            "could not inspect final artifact tree: %s" % error
        ) from error

    if not manifest_seen:
        raise InfrastructureFailure(
            "collector manifest boundary disappeared during finalization"
        )

    rows.sort()
    descriptor = -1
    temporary = None
    try:
        checkpoint()
        descriptor, temporary = tempfile.mkstemp(
            prefix=".manifest.txt.", dir=str(artifacts)
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
            descriptor = -1
            for relative, size in rows:
                checkpoint()
                output.write("%s\t%d\n" % (relative, size))
            checkpoint()
            output.flush()
            os.fsync(output.fileno())
        checkpoint()
        os.replace(temporary, str(manifest))
        temporary = None
        checkpoint()
    except OSError as error:
        raise InfrastructureFailure(
            "could not atomically refresh final artifact manifest: %s" % error
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def initial_report(config: Config, environ: Mapping[str, str]) -> GateReport:
    return GateReport(
        pr_number=config.pr_number,
        head_sha=environ.get("TILEXR_CI_HEAD_SHA", "unknown"),
        base_sha=environ.get("TILEXR_CI_BASE_SHA", "unknown"),
        merge_sha=config.expected_merge_sha,
    )


def verify_final_cleanup(
    read_snapshot: Callable[[], npu_state.Snapshot] = npu_state.read_snapshot,
    *,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
    grace_seconds: float = 10.0,
    poll_seconds: float = 1.0,
):
    started = now()
    deadline = started + grace_seconds
    while True:
        if read_snapshot is npu_state.read_snapshot:
            state = read_snapshot(deadline=deadline, now=now)
        else:
            state = read_snapshot()
        if not state.healthy:
            raise InfrastructureFailure("could not verify final NPU cleanup")
        leftovers = [item for item in state.processes if item.owner == "tilexr-ci"]
        if not leftovers:
            return
        remaining = max(0.0, deadline - now())
        if remaining <= 0:
            details = ", ".join(
                "device=%d pid=%d" % (item.device, item.pid) for item in leftovers
            )
            raise InfrastructureFailure(
                "tilexr-ci NPU processes remain after child cleanup: %s" % details
            )
        sleep(min(poll_seconds, remaining))


def prefer_cleanup_failure(primary, cleanup: BaseException) -> InfrastructureFailure:
    cleanup_failure = _as_gate_failure(cleanup)
    if primary is None:
        result = InfrastructureFailure("final cleanup failed: %s" % cleanup_failure)
    else:
        primary_failure = _as_gate_failure(primary)
        result = InfrastructureFailure(
            "primary failure (%s): %s; final cleanup failed: %s"
            % (failure_class_for(primary_failure), primary_failure, cleanup_failure)
        )
    result.cleanup_failure = True
    return result


def select_final_failure(current, candidate):
    candidate = _as_gate_failure(candidate)
    if current is None:
        return candidate
    if getattr(current, "cleanup_failure", False) or getattr(
        current, "collector_failure", False
    ):
        return current
    if (
        getattr(candidate, "cleanup_failure", False)
        or getattr(candidate, "collector_failure", False)
        or isinstance(candidate, Cancelled)
    ):
        return candidate
    return current


def requires_outer_cleanup(report: GateReport) -> bool:
    return not report.cleanup_checked


def invoke_collector(
    script: pathlib.Path,
    config: Config,
    env: Mapping[str, str],
    *,
    cancellation: Optional[CancellationState] = None,
    popen: Callable = subprocess.Popen,
    now: Callable[[], float] = time.monotonic,
    sleep: Optional[Callable[[float], None]] = None,
    terminate: Callable = terminate_group,
    process_boundary_factory: Optional[Callable] = None,
):
    if not script.is_file() or not os.access(str(script), os.X_OK):
        raise _collector_failure(
            "trusted artifact collector is missing or not executable: %s" % script
        )
    cancellation = cancellation or CancellationState()
    collector_sleep = sleep or time.sleep
    process_boundary = None
    if process_boundary_factory is not None:
        process_boundary = process_boundary_factory()
    elif sys.platform.startswith("linux") and popen is subprocess.Popen:
        process_boundary = LinuxProcessBoundary.prepare()
    started = now()
    normal_deadline = started + 600.0
    cancellation_deadline = started + 30.0 if cancellation.cancelled else None

    def attach_checkpoint():
        nonlocal cancellation_deadline
        current = now()
        if cancellation.cancelled and cancellation_deadline is None:
            cancellation_deadline = current + 30.0
        if cancellation_deadline is not None and current >= min(
            normal_deadline, cancellation_deadline
        ):
            if cancellation_deadline <= normal_deadline:
                raise Cancelled(
                    "artifact collection reached the cancellation finalization deadline"
                )
        if current >= normal_deadline:
            raise _collector_failure(
                "artifact collector exceeded its bounded finalization deadline"
            )

    try:
        child = popen(
            [str(script), str(config.source), str(config.artifacts)],
            cwd=str(config.source),
            env=dict(env),
            start_new_session=True,
        )
    except OSError as error:
        raise _collector_failure("artifact collector failed to run: %s" % error) from error
    if process_boundary is not None:
        child._tilexr_process_boundary = process_boundary
        try:
            process_boundary.attach(
                child.pid,
                child=child,
                deadline=min(
                    normal_deadline,
                    cancellation_deadline
                    if cancellation_deadline is not None
                    else normal_deadline,
                ),
                now=now,
                before_step=attach_checkpoint,
            )
            attach_checkpoint()
        except BaseException as attach_error:
            try:
                process_boundary.abort(child, now=now, sleep=collector_sleep)
            except BaseException as termination_error:
                raise _tracked_termination_failure(
                    attach_error, termination_error, collector=True
                ) from termination_error
            if isinstance(attach_error, Cancelled):
                raise
            if getattr(attach_error, "collector_failure", False):
                raise
            raise _collector_failure(
                "could not attach artifact collector to Linux process boundary: %s"
                % attach_error
            ) from attach_error
    else:
        try:
            attach_checkpoint()
        except BaseException as attach_error:
            try:
                terminate(child)
            except BaseException as termination_error:
                raise _tracked_termination_failure(
                    attach_error, termination_error, collector=True
                ) from termination_error
            raise
    primary_error = None
    try:
        while True:
            current = now()
            if cancellation.cancelled and cancellation_deadline is None:
                cancellation_deadline = current + 30.0
            deadline = normal_deadline
            if cancellation_deadline is not None:
                deadline = min(deadline, cancellation_deadline)
            if current >= deadline:
                if (
                    cancellation_deadline is not None
                    and cancellation_deadline <= normal_deadline
                ):
                    raise Cancelled(
                        "artifact collection reached the cancellation finalization deadline"
                    )
                raise _collector_failure(
                    "artifact collector exceeded its bounded finalization deadline"
                )
            returncode = child.poll()
            if returncode is not None:
                if returncode != 0:
                    raise _collector_failure(
                        "artifact collector exited with status %d" % returncode
                    )
                return
            collector_sleep(min(1.0, deadline - current))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            terminate(child)
        except BaseException as termination_error:
            raise _tracked_termination_failure(
                primary_error, termination_error, collector=True
            ) from termination_error


def failure_class_for(error: BaseException) -> str:
    if isinstance(error, CodeFailure):
        return "code-or-test-failure"
    if isinstance(error, npu_state.ResourceTimeout):
        return "resource-timeout"
    if isinstance(error, ResourceCollision):
        return "resource-collision"
    if isinstance(error, (Cancelled, ObsoleteRun)):
        return "cancelled-or-obsolete"
    return "runner-or-toolchain-failure"


def _collector_failure(message: str) -> InfrastructureFailure:
    error = InfrastructureFailure(message)
    error.collector_failure = True
    return error


def _tracked_termination_failure(primary, termination, *, collector: bool):
    termination_detail = _as_gate_failure(termination)
    if primary is None:
        message = "tracked process-group cleanup failed: %s" % termination_detail
    else:
        primary_detail = _as_gate_failure(primary)
        message = (
            "primary failure (%s): %s; tracked process-group cleanup failed: %s"
            % (failure_class_for(primary_detail), primary_detail, termination_detail)
        )
    error = InfrastructureFailure(message)
    error.cleanup_failure = True
    if collector:
        error.collector_failure = True
    return error


def exit_code_for(error: BaseException) -> int:
    return {
        "code-or-test-failure": 20,
        "resource-timeout": 21,
        "resource-collision": 22,
        "runner-or-toolchain-failure": 23,
        "cancelled-or-obsolete": 130,
    }[failure_class_for(error)]


def _as_gate_failure(error: BaseException) -> BaseException:
    if isinstance(error, (GateFailure, npu_state.ResourceTimeout)):
        return error
    return InfrastructureFailure("unexpected controller failure: %s" % error)


@contextlib.contextmanager
def _signal_cancellation(cancellation: CancellationState):
    previous = {}

    def cancel(signum, frame):
        cancellation.cancel()

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, cancel)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _run_controller_body(
    config: Config,
    environ: Mapping[str, str],
    control_dir: pathlib.Path,
    cancellation: CancellationState,
    *,
    now: Callable[[], float] = time.monotonic,
) -> int:
    token = environ.get("TILEXR_CI_GITHUB_TOKEN", "")
    step_summary_path = environ.get("GITHUB_STEP_SUMMARY")
    report = initial_report(config, environ)
    failure = None
    cancellation_accounted = False
    child_env = phase_environment(
        child_environment(environ), config.source, config.artifacts, config.pr_number
    )
    try:
        config.artifacts.mkdir(parents=True, exist_ok=True)
        seed_empty_case_evidence(config.artifacts / "cases.tsv")
        if not token:
            raise InfrastructureFailure("TILEXR_CI_GITHUB_TOKEN is required")
        orchestrate(
            config,
            control_dir=control_dir,
            token=token,
            cancellation=cancellation,
            report=report,
        )
        validate_success_cases(config.artifacts / "cases.tsv")
    except BaseException as error:
        failure = _as_gate_failure(error)

    if requires_outer_cleanup(report):
        try:
            verify_final_cleanup()
            report.cleanup_checked = True
        except BaseException as error:
            failure = prefer_cleanup_failure(failure, error)

    if failure is not None:
        report.failure_class = failure_class_for(failure)
        report.failure_detail = str(failure)

    # The collector consumes summary.md. Collection still runs if summary writing
    # fails so that the finally path does not silently skip a cleanup boundary.
    try:
        write_summary(report, config.artifacts)
    except BaseException as error:
        if failure is None:
            failure = _as_gate_failure(error)
            report.failure_class = failure_class_for(failure)
            report.failure_detail = str(failure)

    try:
        reset_collector_manifest_boundary(config.artifacts)
    except BaseException as error:
        boundary_failure = _collector_failure(
            "could not prepare collector manifest boundary: %s"
            % _as_gate_failure(error)
        )
        selected = select_final_failure(failure, boundary_failure)
        if selected is not failure:
            failure = selected
            report.failure_class = failure_class_for(failure)
            report.failure_detail = str(failure)

    try:
        invoke_collector(
            control_dir / "collect_artifacts.sh",
            config,
            child_env,
            cancellation=cancellation,
        )
    except BaseException as error:
        collector_failure = _as_gate_failure(error)
        selected = select_final_failure(failure, collector_failure)
        if selected is not failure:
            failure = selected
            report.failure_class = failure_class_for(failure)
            report.failure_detail = str(failure)

    if cancellation.cancelled:
        if not isinstance(failure, Cancelled):
            failure = select_final_failure(
                failure, Cancelled("gate was cancelled during finalization")
            )
            report.failure_class = failure_class_for(failure)
            report.failure_detail = str(failure)
        cancellation_accounted = True

    if cancellation.cancelled:
        if not isinstance(failure, Cancelled):
            selected = select_final_failure(
                failure, Cancelled("gate was cancelled during final summary writing")
            )
            if selected is not failure:
                failure = selected
                report.failure_class = failure_class_for(failure)
                report.failure_detail = str(failure)
        cancellation_accounted = True

    # cancellation_accounted distinguishes signals classified above from a new
    # signal arriving while the bounded final evidence pass is running.
    finalization_deadline = now() + FINAL_ARTIFACT_TIMEOUT_SECONDS

    def update_report(candidate: BaseException) -> bool:
        nonlocal failure
        selected = _as_gate_failure(candidate)
        selected = select_final_failure(failure, selected)
        changed = selected is not failure
        failure = selected
        report.failure_class = failure_class_for(failure)
        report.failure_detail = str(failure)
        return changed

    def recover_terminal_evidence() -> bool:
        nonlocal failure
        if failure is None:
            failure = _collector_failure("terminal artifact finalization failed")
            report.failure_class = failure_class_for(failure)
            report.failure_detail = str(failure)
        try:
            recover_minimal_artifact_upload(config.artifacts, report, failure)
            return True
        except BaseException as error:
            prior = failure
            failure = _collector_failure(
                "primary failure (%s): %s; minimal artifact recovery failed: %s"
                % (
                    failure_class_for(prior),
                    prior,
                    _as_gate_failure(error),
                )
            )
            report.failure_class = failure_class_for(failure)
            report.failure_detail = str(failure)
            print("ERROR: %s" % failure, file=sys.stderr)
            return False

    def write_final_summary() -> bool:
        for attempt in range(2):
            try:
                write_summary(report, config.artifacts)
                return True
            except BaseException as error:
                update_report(
                    _collector_failure(
                        "final artifact summary write failed: %s"
                        % _as_gate_failure(error)
                    )
                )
                if attempt == 1:
                    recover_terminal_evidence()
                    return False
        return False

    def account_new_cancellation(detail: str) -> bool:
        nonlocal cancellation_accounted
        prior = failure
        if prior is not None and not isinstance(prior, Cancelled):
            detail = "%s; prior failure (%s): %s" % (
                detail,
                failure_class_for(prior),
                prior,
            )
        changed = update_report(Cancelled(detail))
        cancellation_accounted = True
        return changed

    artifact_finalization_terminal = not write_final_summary()
    refresh_attempt = 0
    while not artifact_finalization_terminal and refresh_attempt < 2:
        if cancellation.cancelled and not cancellation_accounted:
            account_new_cancellation(
                "gate was cancelled before final artifact manifest refresh"
            )
            if not write_final_summary():
                artifact_finalization_terminal = True
                break

        refresh_attempt += 1
        failure_changed = False
        try:
            refresh_final_manifest(
                config.artifacts,
                deadline=finalization_deadline,
                now=now,
                cancellation=cancellation,
                cancellation_accounted=cancellation_accounted,
            )
        except BaseException as error:
            if isinstance(error, Cancelled) or (
                cancellation.cancelled and not cancellation_accounted
            ):
                detail = str(error) if isinstance(error, Cancelled) else (
                    "gate was cancelled during final artifact manifest refresh"
                )
                account_new_cancellation(detail)
                failure_changed = True
            else:
                failure_changed = update_report(
                    _collector_failure(
                        "final artifact manifest refresh failed: %s"
                        % _as_gate_failure(error)
                    )
                )
            if refresh_attempt == 2:
                if failure_changed and not write_final_summary():
                    artifact_finalization_terminal = True
                    break
                recover_terminal_evidence()
                artifact_finalization_terminal = True
                break
            if not write_final_summary():
                artifact_finalization_terminal = True
                break
            continue

        # A signal can arrive after the refresh's last checkpoint but before it
        # returns to the controller. The summary changed by this classification
        # must consume the one allowed manifest retry or invalidate the manifest.
        if cancellation.cancelled and not cancellation_accounted:
            account_new_cancellation(
                "gate was cancelled after final artifact manifest refresh"
            )
            if not write_final_summary():
                artifact_finalization_terminal = True
                break
            if refresh_attempt == 2:
                recover_terminal_evidence()
                artifact_finalization_terminal = True
                break
            continue
        break

    if step_summary_path:
        step_path = pathlib.Path(step_summary_path)
        try:
            step_summary_prefix_size = step_summary_size(step_path)
        except BaseException as error:
            changed = update_report(
                _collector_failure(
                    "authoritative GitHub step summary preparation failed: %s"
                    % _as_gate_failure(error)
                )
            )
            if changed:
                recover_terminal_evidence()
            step_summary_prefix_size = None

        def roll_back_controller_step_summary() -> bool:
            if step_summary_prefix_size is None:
                return False
            try:
                rollback_step_summary(step_path, step_summary_prefix_size)
                return True
            except BaseException as error:
                changed = update_report(
                    _collector_failure(
                        "authoritative GitHub step summary rollback failed: %s"
                        % _as_gate_failure(error)
                    )
                )
                if changed:
                    recover_terminal_evidence()
                print(
                    "ERROR: could not roll back authoritative GitHub step summary: %s"
                    % _as_gate_failure(error),
                    file=sys.stderr,
                )
                return False

        def emit_terminal_step_summary() -> None:
            # Remove every controller-owned block before the private final write.
            # If that write fails too, remove any partial write so an earlier
            # success block cannot contradict the selected exit state.
            for terminal_attempt in range(2):
                roll_back_controller_step_summary()
                if cancellation.cancelled and not cancellation_accounted:
                    changed = account_new_cancellation(
                        "gate was cancelled before terminal step summary emission"
                    )
                    if changed:
                        recover_terminal_evidence()
                try:
                    _append_authoritative_step_text(
                        step_path,
                        _authoritative_step_summary_text(report, config.artifacts),
                    )
                except BaseException as error:
                    changed = update_report(
                        _collector_failure(
                            "terminal GitHub step summary write failed: %s"
                            % _as_gate_failure(error)
                        )
                    )
                    if changed:
                        recover_terminal_evidence()
                    roll_back_controller_step_summary()
                    print(
                        "ERROR: could not emit terminal GitHub step summary: %s"
                        % _as_gate_failure(error),
                        file=sys.stderr,
                    )
                    return

                # CancellationState is monotonic, so at most one correction is
                # needed for a signal arriving during the private final write.
                if cancellation.cancelled and not cancellation_accounted:
                    changed = account_new_cancellation(
                        "gate was cancelled during terminal step summary emission"
                    )
                    if changed:
                        recover_terminal_evidence()
                        if terminal_attempt == 0:
                            continue
                return

        for step_attempt in range(2):
            try:
                append_authoritative_step_summary(step_path, report, config.artifacts)
            except BaseException as error:
                changed = update_report(
                    _collector_failure(
                        "authoritative GitHub step summary write failed: %s"
                        % _as_gate_failure(error)
                    )
                )
                if changed:
                    recover_terminal_evidence()
                if step_attempt == 1:
                    emit_terminal_step_summary()
                    break
                continue

            if cancellation.cancelled and not cancellation_accounted:
                changed = account_new_cancellation(
                    "gate was cancelled during authoritative step summary emission"
                )
                if changed:
                    recover_terminal_evidence()
                    if step_attempt == 0:
                        continue
                    emit_terminal_step_summary()
            break

    return 0 if failure is None else exit_code_for(failure)


def run_controller(config: Config, *, environ=None, control_dir: pathlib.Path = CONTROL_DIR) -> int:
    copied_environ = dict(os.environ if environ is None else environ)
    cancellation = CancellationState()
    with _signal_cancellation(cancellation):
        return _run_controller_body(config, copied_environ, control_dir, cancellation)


def main(argv=None) -> int:
    try:
        config = parse_config(argv)
    except BaseException as error:
        failure = _as_gate_failure(error)
        print(str(failure), file=sys.stderr)
        return exit_code_for(failure)
    return run_controller(config)


if __name__ == "__main__":
    sys.exit(main())
