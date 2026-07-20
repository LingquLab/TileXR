#!/usr/bin/env python3
"""Trusted controller for the TileXR pull-request NPU gate."""

import argparse
import contextlib
import ctypes
import dataclasses
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Dict, Mapping, Optional, Tuple

import npu_state


EXPECTED_REPOSITORY = "LingquLab/TileXR"
BUILD_TIMEOUT_SECONDS = 7200
HARDWARE_TIMEOUT_SECONDS = 7200
QUEUE_TIMEOUT_SECONDS = 21600
QUEUE_POLL_SECONDS = 60
QUEUE_STABLE_SAMPLES = 2
PR_SET_DUMPABLE = 4
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


@dataclasses.dataclass(frozen=True)
class Config:
    source: pathlib.Path
    scratch: pathlib.Path
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
    parser.add_argument("--scratch", type=pathlib.Path, required=True)
    parser.add_argument("--expected-merge-sha", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    args = parser.parse_args(argv)
    return validate_config(
        Config(
            source=args.source.resolve(),
            scratch=args.scratch.resolve(),
            expected_merge_sha=args.expected_merge_sha,
            repository=args.repository,
            pr_number=args.pr_number,
        )
    )


def disable_process_dumpability(
    *,
    platform: Optional[str] = None,
    prctl: Optional[Callable] = None,
    get_errno: Callable[[], int] = ctypes.get_errno,
) -> None:
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux"):
        raise InfrastructureFailure("the trusted controller requires Linux")
    try:
        if prctl is None:
            libc = ctypes.CDLL(None, use_errno=True)
            prctl = libc.prctl
            prctl.argtypes = [
                ctypes.c_int,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            prctl.restype = ctypes.c_int
        result = prctl(PR_SET_DUMPABLE, 0, 0, 0, 0)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise InfrastructureFailure(
            "could not disable controller process dumpability: %s" % error
        ) from error
    if result != 0:
        error_number = get_errno()
        detail = os.strerror(error_number) if error_number else "unknown error"
        raise InfrastructureFailure(
            "could not disable controller process dumpability: %s" % detail
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
            return "ci-npu-during-build"
        return None
    if phase == "hardware":
        if any(item.owner != ci_user for item in state.processes):
            return "foreign-npu-process"
        return None
    raise InfrastructureFailure("unknown gate phase %s" % phase)


def _enforce_policy(phase: str, state: npu_state.Snapshot):
    violation = policy_violation(phase, state)
    if violation == "ci-npu-during-build":
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


def phase_environment(environ, source, scratch, pr_number: int) -> Dict[str, str]:
    phase = dict(environ)
    phase.update(
        {
            "PR_NUMBER": str(pr_number),
            "TILEXR_CI_SOURCE_DIR": str(source),
            "TILEXR_CI_SCRATCH_DIR": str(scratch),
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
    scratch: pathlib.Path,
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
    command = [str(script), str(source), str(scratch)]

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
                    attach_error, termination_error
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
                    attach_error, termination_error
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
                primary_error, termination_error
            ) from termination_error


def orchestrate(
    config: Config,
    *,
    control_dir: pathlib.Path,
    cancellation: CancellationState,
    report: Optional[GateReport] = None,
    run_phase_fn: Callable = run_phase,
    wait_for_idle_fn: Callable = npu_state.wait_for_idle,
    read_snapshot_fn: Callable = npu_state.read_snapshot,
    verify_merge_sha_fn: Callable = verify_merge_sha,
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
        child_environment(), config.source, config.scratch, config.pr_number
    )
    phase_sleep = sleep_fn or (lambda seconds: cancellation_aware_sleep(seconds, cancellation))
    resource_clock = resource_now or time.monotonic

    report.build_outcome = "running"
    try:
        run_phase_fn(
            "build",
            control_dir / "build_blue.sh",
            config.source,
            config.scratch,
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
    resource_error = None
    try:
        def queue_snapshot():
            if queue_snapshot_fn is not None:
                return queue_snapshot_fn(deadline=resource_deadline, now=resource_clock)

            def command_checkpoint():
                if cancellation.cancelled:
                    raise Cancelled("gate was cancelled during queued NPU inspection")

            return npu_state.read_snapshot(
                deadline=resource_deadline,
                now=resource_clock,
                before_command=command_checkpoint,
            )

        wait_for_idle_fn(
            read_snapshot=queue_snapshot,
            sleep=phase_sleep,
            now=resource_clock,
            emit=emit,
            max_wait_seconds=QUEUE_TIMEOUT_SECONDS,
            poll_seconds=QUEUE_POLL_SECONDS,
            stable_samples=QUEUE_STABLE_SAMPLES,
        )
        queue_finished = True
        report.wait_seconds = max(0.0, resource_clock() - queue_started)
        if resource_clock() >= resource_deadline:
            raise npu_state.ResourceTimeout(
                "NPU resource budget expired before hardware launch"
            )
        report.queue_outcome = "passed"

        if cancellation.cancelled:
            raise Cancelled("gate was cancelled after resource wait")

        report.hardware_outcome = "running"
        hardware_started = time.monotonic()
        try:
            run_phase_fn(
                "hardware",
                control_dir / "run_hardware.sh",
                config.source,
                config.scratch,
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
    except npu_state.ResourceTimeout as error:
        resource_error = error
        report.queue_outcome = "timed out"
        raise
    except BaseException as error:
        resource_error = error
        if not queue_finished and report.queue_outcome == "waiting":
            report.queue_outcome = "failed"
        raise
    finally:
        if not queue_finished:
            report.wait_seconds = max(0.0, resource_clock() - queue_started)
        report.cleanup_checked = True
        try:
            verify_final_cleanup(read_snapshot_fn)
        except BaseException as cleanup_error:
            raise prefer_cleanup_failure(resource_error, cleanup_error) from cleanup_error
    return report


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
    if getattr(current, "cleanup_failure", False):
        return current
    if (
        getattr(candidate, "cleanup_failure", False)
        or isinstance(candidate, Cancelled)
    ):
        return candidate
    return current


def requires_outer_cleanup(report: GateReport) -> bool:
    return not report.cleanup_checked


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


def _tracked_termination_failure(primary, termination):
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


def format_report(report: GateReport) -> str:
    detail = report.failure_detail or "none"
    return "\n".join(
        (
            "TileXR trusted PR gate",
            "PR: #%d" % report.pr_number,
            "Head SHA: %s" % report.head_sha,
            "Base SHA: %s" % report.base_sha,
            "Merge SHA: %s" % report.merge_sha,
            "Build outcome: %s" % report.build_outcome,
            "NPU wait outcome: %s" % report.queue_outcome,
            "Hardware outcome: %s" % report.hardware_outcome,
            "NPU wait seconds: %.1f" % report.wait_seconds,
            "Hardware execution seconds: %.1f" % report.execution_seconds,
            "Failure class: %s" % report.failure_class,
            "Failure detail: %s" % detail,
            "UDMA data plane: out of scope on 910B3",
            "Multi-host validation: out of scope for this gate",
        )
    )


def _run_controller_body(
    config: Config,
    environ: Mapping[str, str],
    control_dir: pathlib.Path,
    cancellation: CancellationState,
) -> int:
    report = initial_report(config, environ)
    failure = None
    try:
        if config.scratch.is_symlink():
            raise InfrastructureFailure(
                "scratch directory must not be a symbolic link: %s" % config.scratch
            )
        config.scratch.mkdir(parents=True, exist_ok=True)
        if not config.scratch.is_dir() or config.scratch.is_symlink():
            raise InfrastructureFailure(
                "scratch path is not a real directory: %s" % config.scratch
            )
        orchestrate(
            config,
            control_dir=control_dir,
            cancellation=cancellation,
            report=report,
        )
    except BaseException as error:
        failure = _as_gate_failure(error)

    if requires_outer_cleanup(report):
        try:
            verify_final_cleanup()
            report.cleanup_checked = True
        except BaseException as error:
            failure = prefer_cleanup_failure(failure, error)

    if cancellation.cancelled and not isinstance(failure, Cancelled):
        failure = select_final_failure(
            failure, Cancelled("gate was cancelled during finalization")
        )

    if failure is not None:
        report.failure_class = failure_class_for(failure)
        report.failure_detail = str(failure)

    print(format_report(report))
    if failure is not None:
        print("ERROR: %s" % failure, file=sys.stderr)
    return 0 if failure is None else exit_code_for(failure)


def run_controller(
    config: Config,
    *,
    environ=None,
    control_dir: pathlib.Path = CONTROL_DIR,
    harden_process: Callable[[], None] = disable_process_dumpability,
) -> int:
    harden_process()
    copied_environ = dict(os.environ if environ is None else environ)
    cancellation = CancellationState()
    with _signal_cancellation(cancellation):
        return _run_controller_body(
            config,
            copied_environ,
            control_dir,
            cancellation,
        )


def main(argv=None, *, environ=None) -> int:
    copied_environ = dict(os.environ if environ is None else environ)
    try:
        config = parse_config(argv)
        return run_controller(config, environ=copied_environ)
    except BaseException as error:
        failure = _as_gate_failure(error)
        print(str(failure), file=sys.stderr)
        return exit_code_for(failure)


if __name__ == "__main__":
    sys.exit(main())
