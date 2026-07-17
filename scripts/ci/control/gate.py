#!/usr/bin/env python3
"""Trusted controller for the TileXR pull-request NPU gate."""

import argparse
import contextlib
import dataclasses
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Dict, IO, List, Mapping, Optional, Tuple

import npu_state


EXPECTED_REPOSITORY = "LingquLab/TileXR"
GITHUB_API_VERSION = "2022-11-28"
GITHUB_TIMEOUT_SECONDS = 30
BUILD_TIMEOUT_SECONDS = 7200
HARDWARE_TIMEOUT_SECONDS = 7200
QUEUE_TIMEOUT_SECONDS = 21600
QUEUE_POLL_SECONDS = 60
QUEUE_STABLE_SAMPLES = 2
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


def terminate_group(
    child,
    killpg=os.killpg,
    *,
    group_exists: Callable[[int], bool] = _process_group_exists,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
):
    """Stop only the process group created for the supplied child."""
    try:
        killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = now() + 10.0
    while group_exists(child.pid) and now() < deadline:
        sleep(min(0.1, max(0.0, deadline - now())))
    if group_exists(child.pid):
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
) -> int:
    cancellation = cancellation or CancellationState()
    command = [str(script), str(source), str(artifacts)]
    if cancellation.cancelled:
        raise Cancelled("gate was cancelled before %s" % phase)
    _enforce_policy(phase, read_snapshot())
    try:
        child = popen(
            command,
            cwd=str(source),
            env=dict(env),
            start_new_session=True,
        )
    except OSError as error:
        raise InfrastructureFailure("could not start trusted %s phase: %s" % (phase, error)) from error

    started = now()
    deadline = started + timeout_seconds

    def monitored_snapshot():
        if read_snapshot_with_deadline is not None:
            return read_snapshot_with_deadline(deadline)
        if read_snapshot is npu_state.read_snapshot:
            return read_snapshot(deadline=deadline, now=now)
        return read_snapshot()

    try:
        while True:
            if cancellation.cancelled:
                raise Cancelled("gate was cancelled during %s" % phase)

            if now() >= deadline:
                raise CodeFailure("%s phase exceeded %.0f seconds" % (phase, timeout_seconds))
            state = monitored_snapshot()
            if now() >= deadline:
                raise CodeFailure("%s phase exceeded %.0f seconds" % (phase, timeout_seconds))
            _enforce_policy(phase, state)

            returncode = child.poll()
            if returncode is not None:
                if returncode != 0:
                    if returncode in (23, 126, 127):
                        raise InfrastructureFailure(
                            "%s phase reported a runner or toolchain failure (status %d)"
                            % (phase, returncode)
                        )
                    raise CodeFailure("%s phase exited with status %d" % (phase, returncode))
                return returncode

            elapsed = max(0.0, now() - started)
            if elapsed >= timeout_seconds:
                raise CodeFailure("%s phase exceeded %.0f seconds" % (phase, timeout_seconds))
            sleep(min(10.0, max(0.0, timeout_seconds - elapsed)))
    finally:
        terminate(child)


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
    queue_finished = False
    try:
        with lock_fn(
            cancellation=cancellation,
            max_wait_seconds=QUEUE_TIMEOUT_SECONDS,
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
                try:
                    wait_for_idle_fn(
                        read_snapshot=read_snapshot_fn,
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
        return cases, diagnostics
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return cases, ["could not read cases.tsv: %s" % error]
    for number, line in enumerate(lines, 1):
        fields = line.split("\t")
        try:
            if len(fields) != 4 or not fields[0].strip() or not fields[1].strip():
                raise ValueError("expected four tab-separated fields")
            cases.append(
                (fields[0].strip(), fields[1].strip(), int(fields[2]), float(fields[3]))
            )
        except ValueError as error:
            diagnostics.append("line %d: %s" % (number, error))
    return cases, diagnostics


def _summary_text(report: GateReport, cases, diagnostics) -> str:
    passed = [case for case in cases if case[1].lower() in ("pass", "passed", "success")]
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
):
    if not script.is_file() or not os.access(str(script), os.X_OK):
        raise _collector_failure(
            "trusted artifact collector is missing or not executable: %s" % script
        )
    cancellation = cancellation or CancellationState()
    collector_sleep = sleep or time.sleep
    try:
        child = popen(
            [str(script), str(config.source), str(config.artifacts)],
            cwd=str(config.source),
            env=dict(env),
            start_new_session=True,
        )
    except OSError as error:
        raise _collector_failure("artifact collector failed to run: %s" % error) from error
    started = now()
    try:
        while True:
            returncode = child.poll()
            if returncode is not None:
                if returncode != 0:
                    raise _collector_failure(
                        "artifact collector exited with status %d" % returncode
                    )
                return
            elapsed = max(0.0, now() - started)
            if elapsed >= 600:
                raise _collector_failure("artifact collector exceeded 600 seconds")
            collector_sleep(min(1.0, 600 - elapsed))
    finally:
        terminate(child)


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
) -> int:
    token = environ.get("TILEXR_CI_GITHUB_TOKEN", "")
    step_summary_path = environ.get("GITHUB_STEP_SUMMARY")
    report = initial_report(config, environ)
    failure = None
    child_env = phase_environment(
        child_environment(environ), config.source, config.artifacts, config.pr_number
    )
    try:
        if not token:
            raise InfrastructureFailure("TILEXR_CI_GITHUB_TOKEN is required")
        config.artifacts.mkdir(parents=True, exist_ok=True)
        orchestrate(
            config,
            control_dir=control_dir,
            token=token,
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
        try:
            write_summary(report, config.artifacts)
        except GateFailure:
            pass

    if cancellation.cancelled and not isinstance(failure, Cancelled):
        failure = select_final_failure(
            failure, Cancelled("gate was cancelled during finalization")
        )
        report.failure_class = failure_class_for(failure)
        report.failure_detail = str(failure)
        try:
            write_summary(report, config.artifacts)
        except GateFailure:
            pass

    if step_summary_path:
        try:
            with pathlib.Path(step_summary_path).open("a", encoding="utf-8") as step_summary:
                write_summary(report, config.artifacts, step_summary=step_summary)
        except BaseException as error:
            if failure is None:
                failure = _as_gate_failure(error)
                report.failure_class = failure_class_for(failure)
                report.failure_detail = str(failure)
                try:
                    write_summary(report, config.artifacts)
                except GateFailure:
                    pass

    if cancellation.cancelled and not isinstance(failure, Cancelled):
        selected = select_final_failure(
            failure, Cancelled("gate was cancelled during final summary writing")
        )
        if selected is not failure:
            failure = selected
            report.failure_class = failure_class_for(failure)
            report.failure_detail = str(failure)
            try:
                write_summary(report, config.artifacts)
            except GateFailure:
                pass

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
