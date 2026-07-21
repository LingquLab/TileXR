#!/usr/bin/env python3
import errno
import io
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


CONTROL_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ci" / "control"
sys.path.insert(0, str(CONTROL_DIR))

import gate
import npu_state


def process_is_live(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if not sys.platform.startswith("linux"):
        return True
    try:
        state = pathlib.Path("/proc/%d/stat" % pid).read_text(
            encoding="ascii"
        ).rsplit(")", 1)[1].split()[0]
    except (FileNotFoundError, IndexError, OSError):
        return False
    return state != "Z"


def snapshot(*processes, healthy=True):
    return npu_state.Snapshot(healthy=healthy, processes=tuple(processes))


def process(owner, pid=100, device=0):
    return npu_state.NpuProcess(device=device, pid=pid, name="python3", owner=owner)


class MergeVerificationTests(unittest.TestCase):
    def test_mismatched_local_merge_is_obsolete(self):
        with self.assertRaises(gate.ObsoleteRun):
            gate.verify_merge_sha("/source", "expected", run_git=lambda path: "actual")

    def test_matching_local_merge_succeeds(self):
        self.assertEqual(
            gate.verify_merge_sha("/source", "expected", run_git=lambda path: "expected"),
            "expected",
        )

    def test_git_failure_is_infrastructure_failure(self):
        def fail(path):
            raise subprocess.CalledProcessError(128, ["git"])

        with self.assertRaises(gate.InfrastructureFailure):
            gate.verify_merge_sha("/source", "expected", run_git=fail)


class PolicyTests(unittest.TestCase):
    def test_policy_uses_supplied_ci_owner(self):
        state = snapshot(process("ci-special"))
        self.assertIsNone(
            gate.policy_violation("hardware", state, "ci-special")
        )
        self.assertEqual(
            gate.policy_violation("hardware", state, "tilexr-ci"),
            "foreign-npu-process",
        )

    def test_build_permits_all_npu_process_owners(self):
        state = snapshot(process("tilexr-ci"), process("alice", pid=101))
        self.assertIsNone(gate.policy_violation("build", state))

    def test_unhealthy_state_is_infrastructure_failure(self):
        for phase in ("build", "hardware"):
            with self.subTest(phase=phase), self.assertRaises(gate.InfrastructureFailure):
                gate.policy_violation(phase, snapshot(healthy=False))

    def test_hardware_rejects_foreign_and_unknown_processes(self):
        for owner in ("alice", "unknown"):
            with self.subTest(owner=owner):
                self.assertEqual(
                    gate.policy_violation("hardware", snapshot(process(owner))),
                    "foreign-npu-process",
                )

    def test_hardware_allows_ci_owned_processes(self):
        self.assertIsNone(
            gate.policy_violation(
                "hardware", snapshot(process("tilexr-ci", 100), process("tilexr-ci", 101))
            )
        )


class ChildEnvironmentTests(unittest.TestCase):
    def test_tokens_and_github_command_files_are_removed(self):
        removed = {
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
        }
        parent = {name: "/secret/%s" % name for name in removed}
        parent["PATH"] = "/usr/bin:/bin"
        parent["HOME"] = "/home/tilexr-ci"

        child = gate.child_environment(parent)

        self.assertTrue(removed.isdisjoint(child))
        self.assertEqual(child["PATH"], "/usr/bin:/bin")
        self.assertEqual(child["HOME"], "/home/tilexr-ci")
        self.assertEqual(child["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertNotIn("PR_NUMBER", child)

        phase = gate.phase_environment(child, "/source", "/scratch", 42)
        self.assertEqual(phase["PR_NUMBER"], "42")
        self.assertEqual(phase["TILEXR_CI_SOURCE_DIR"], "/source")
        self.assertEqual(phase["TILEXR_CI_SCRATCH_DIR"], "/scratch")
        self.assertEqual(phase["TILEXR_CANN_HOME"], "/home/tilexr-ci/toolchains/cann/9.1.0")

class FakeProcess:
    def __init__(self, polls=(None,), returncode=0, forced=False):
        self.pid = 4242
        self._polls = list(polls)
        self.returncode = returncode
        self.wait_calls = []
        self.forced = forced

    def poll(self):
        if self._polls:
            value = self._polls.pop(0)
            if value is not None:
                self.returncode = value
            return value
        return self.returncode

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self.forced and len(self.wait_calls) == 1:
            raise subprocess.TimeoutExpired(["phase"], timeout)
        self.returncode = -signal.SIGTERM
        return self.returncode


def partial_boundary(signals):
    root = gate.ProcessRecord(700, 999, 700, 700, 10, os.geteuid())
    escaped = gate.ProcessRecord(701, 700, 701, 701, 11, os.geteuid())
    state = {700: root, 701: escaped}

    class PartialBoundary(gate.LinuxProcessBoundary):
        def attach(self, root_pid, child=None, **kwargs):
            self.root_pid = root_pid
            self._remember(root)
            self._remember(escaped, lineage_verified=True)
            raise gate.InfrastructureFailure("partial descendant pidfd attach failed")

    def signaler(record, signum):
        signals.append((record.pid, signum))
        if signum == signal.SIGKILL:
            state.clear()

    return PartialBoundary(
        parent_pid=999,
        baseline=(),
        scan=lambda: dict(state),
        signal_identity=signaler,
    )


def cancelling_partial_boundary(signals, cancellation, clock):
    root = gate.ProcessRecord(700, 999, 700, 700, 10, os.geteuid())
    escaped = gate.ProcessRecord(701, 700, 701, 701, 11, os.geteuid())
    state = {700: root, 701: escaped}

    class CancellingBoundary(gate.LinuxProcessBoundary):
        def attach(self, root_pid, child=None, **kwargs):
            self.root_pid = root_pid
            self._remember(root)
            self._remember(escaped, lineage_verified=True)
            cancellation.cancel()
            clock[0] = 30.0
            kwargs["before_step"]()
            clock[0] = 60.0
            kwargs["before_step"]()

    def signaler(record, signum):
        signals.append((record.pid, signum))
        if signum == signal.SIGKILL:
            state.clear()

    return CancellingBoundary(
        parent_pid=999,
        baseline=(),
        scan=lambda: dict(state),
        signal_identity=signaler,
    )


class TerminationTests(unittest.TestCase):
    def test_live_term_resistant_leader_is_killed_even_when_membership_scan_is_empty(self):
        child = mock.Mock(pid=4242)
        child.poll.return_value = None
        child.wait.return_value = -signal.SIGKILL
        signals = []
        clock = [0.0]

        def advance(seconds):
            clock[0] += seconds

        gate.terminate_group(
            child,
            killpg=lambda pgid, signum: signals.append(signum),
            session_members=lambda pgid: (),
            now=lambda: clock[0],
            sleep=advance,
        )

        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])

    def test_reused_descendant_pid_with_tracked_lineage_gets_new_identity(self):
        root = gate.ProcessRecord(600, 999, 600, 600, 5, os.geteuid())
        old = gate.ProcessRecord(700, 600, 600, 600, 10, os.geteuid())
        reused = gate.ProcessRecord(700, 600, 600, 600, 20, os.geteuid())
        state = {600: root, 700: old}
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: dict(state),
        )

        with mock.patch.object(
            gate.os, "pidfd_open", side_effect=(60, 70, 71), create=True
        ), mock.patch.object(
            gate.signal, "pidfd_send_signal", create=True
        ) as pidfd_signal, mock.patch.object(gate.os, "close") as close:
            boundary.attach(600)
            state[700] = reused

            boundary.signal_tracked(signal.SIGTERM)

            self.assertIn(mock.call(70), close.mock_calls)
            self.assertIn(
                mock.call(71, signal.SIGTERM, None, 0),
                pidfd_signal.mock_calls,
            )
            self.assertNotIn(
                mock.call(70, signal.SIGTERM, None, 0),
                pidfd_signal.mock_calls,
            )
            boundary._close_pidfds()

    def test_unrelated_reused_pid_is_not_accepted_as_new_generation(self):
        root = gate.ProcessRecord(600, 999, 600, 600, 5, os.geteuid())
        old = gate.ProcessRecord(700, 600, 600, 600, 10, os.geteuid())
        unrelated = gate.ProcessRecord(700, 888, 888, 888, 20, os.geteuid())
        state = {600: root, 700: old}
        signals = []
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: dict(state),
            signal_identity=lambda identity, signum: signals.append((identity, signum)),
        )
        boundary.attach(600)

        state[700] = unrelated
        boundary.signal_tracked(signal.SIGTERM)

        self.assertNotIn((unrelated, signal.SIGTERM), signals)

    def test_reused_root_pid_is_never_accepted_as_new_generation(self):
        old = gate.ProcessRecord(700, 999, 700, 700, 10, os.geteuid())
        reused = gate.ProcessRecord(700, 999, 700, 700, 20, os.geteuid())
        scans = iter(({700: old}, {700: reused}))
        signals = []
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: next(scans),
            signal_identity=lambda identity, signum: signals.append((identity, signum)),
        )
        boundary.attach(700)

        boundary.signal_tracked(signal.SIGTERM)

        self.assertEqual(signals, [])

    def test_exited_child_does_not_pin_unrelated_reused_root_pid(self):
        reused_root = gate.ProcessRecord(700, 888, 700, 700, 20, os.geteuid())
        adopted = gate.ProcessRecord(701, 999, 701, 701, 21, os.geteuid())
        signals = []
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: {700: reused_root, 701: adopted},
            signal_identity=lambda identity, signum: signals.append((identity, signum)),
        )
        child = mock.Mock(pid=700)
        child.poll.return_value = 0

        boundary.attach(700, child=child)
        boundary.signal_tracked(signal.SIGTERM)

        self.assertEqual(signals, [(adopted, signal.SIGTERM)])

    def test_live_child_root_with_wrong_parent_is_rejected(self):
        wrong_parent = gate.ProcessRecord(700, 888, 700, 700, 20, os.geteuid())
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: {700: wrong_parent},
            signal_identity=lambda identity, signum: None,
        )
        child = mock.Mock(pid=700)
        child.poll.return_value = None

        with self.assertRaises(gate.InfrastructureFailure):
            boundary.attach(700, child=child)

        self.assertEqual(boundary.known, {})

    def test_linux_boundary_fails_closed_when_pidfd_open_is_unavailable(self):
        record = gate.ProcessRecord(700, 999, 700, 700, 10, os.geteuid())
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: {700: record},
        )

        with mock.patch.object(
            gate.os, "pidfd_open", side_effect=PermissionError, create=True
        ), mock.patch.object(
            gate.signal, "pidfd_send_signal", create=True
        ):
            with self.assertRaises(gate.InfrastructureFailure):
                boundary.attach(700)

    def test_attach_checkpoint_after_pidfd_open_closes_partial_fd(self):
        record = gate.ProcessRecord(700, 999, 700, 700, 10, os.geteuid())
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: {700: record},
        )
        checkpoints = []

        def checkpoint():
            checkpoints.append(True)
            if len(checkpoints) == 2:
                raise gate.Cancelled("attach cancelled")

        with mock.patch.object(
            gate.os, "pidfd_open", return_value=99, create=True
        ), mock.patch.object(
            gate.signal, "pidfd_send_signal", create=True
        ), mock.patch.object(gate.os, "close") as close:
            with self.assertRaises(gate.Cancelled):
                boundary.attach(700, before_step=checkpoint)

        close.assert_called_once_with(99)

    def test_linux_boundary_fails_if_process_remains_after_kill_and_reap_grace(self):
        record = gate.ProcessRecord(700, 999, 700, 700, 10, os.geteuid())
        signals = []
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: {700: record},
            signal_identity=lambda identity, signum: signals.append(signum),
        )
        boundary.attach(700)
        child = mock.Mock(pid=700)
        child.poll.return_value = None
        child.wait.return_value = -signal.SIGKILL
        clock = [0.0]

        with self.assertRaises(gate.InfrastructureFailure):
            boundary.terminate(
                child,
                now=lambda: clock[0],
                sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
            )

        self.assertIn(signal.SIGKILL, signals)

    def test_newly_adopted_descendant_after_parent_kill_is_also_killed(self):
        root = gate.ProcessRecord(700, 999, 700, 700, 10, os.geteuid())
        escaped = gate.ProcessRecord(701, 999, 701, 701, 11, os.geteuid())
        scans = iter(
            (
                {700: root},
                {700: root},
                {700: root},
                {700: root},
                {701: escaped},
                {},
            )
        )
        signals = []
        boundary = gate.LinuxProcessBoundary(
            parent_pid=999,
            baseline=(),
            scan=lambda: next(scans),
            signal_identity=lambda record, signum: signals.append((record.pid, signum)),
        )
        boundary.attach(700)
        child = mock.Mock(pid=700)
        child.poll.return_value = None
        child.wait.return_value = -signal.SIGKILL
        clock = iter((0.0, 10.0, 10.0, 10.0, 10.1))

        boundary.terminate(
            child, now=lambda: next(clock), sleep=lambda seconds: None
        )

        self.assertIn((701, signal.SIGKILL), signals)

    @unittest.skipUnless(
        sys.platform.startswith("linux")
        and hasattr(os, "pidfd_open")
        and hasattr(signal, "pidfd_send_signal"),
        "Linux pidfd/subreaper regression",
    )
    def test_subreaper_cleans_descendant_that_escapes_session_and_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = pathlib.Path(directory) / "escaped.pid"
            boundary = gate.LinuxProcessBoundary.prepare()
            leader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, subprocess; "
                        "child=subprocess.Popen(['sleep', '60'], start_new_session=True); "
                        "pathlib.Path(%r).write_text(str(child.pid), encoding='ascii')"
                    )
                    % str(pid_file),
                ],
                start_new_session=True,
            )
            boundary.attach(leader.pid)
            leader._tilexr_process_boundary = boundary
            leader.wait(timeout=10)
            escaped_pid = int(pid_file.read_text(encoding="ascii"))

            gate.terminate_group(leader)

            with self.assertRaises(ProcessLookupError):
                os.kill(escaped_pid, 0)

    def test_exited_leader_with_reused_numeric_pgid_is_not_signalled(self):
        child = FakeProcess(polls=(0,), returncode=0)
        killpg = mock.Mock()

        gate.terminate_group(
            child,
            killpg=killpg,
            session_members=lambda pgid: (),
            now=lambda: 0.0,
            sleep=lambda seconds: None,
        )

        killpg.assert_not_called()
        self.assertEqual(child.wait_calls, [None])

    def test_live_group_leader_is_reaped_promptly_after_term(self):
        child = subprocess.Popen(["sleep", "60"], start_new_session=True)
        signals = []

        def killpg(pgid, signum):
            signals.append(signum)
            os.killpg(pgid, signum)

        started = time.monotonic()
        try:
            gate.terminate_group(child, killpg=killpg)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()

        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(signals, [signal.SIGTERM])
        self.assertIsNotNone(child.returncode)

    def test_exited_group_leader_does_not_leave_background_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            pid_file = pathlib.Path(directory) / "descendant.pid"
            leader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import pathlib, subprocess; "
                        "child=subprocess.Popen(['sleep', '60']); "
                        "pathlib.Path(%r).write_text(str(child.pid), encoding='ascii')"
                    )
                    % str(pid_file),
                ],
                start_new_session=True,
            )
            try:
                leader.wait(timeout=10)
                descendant_pid = int(pid_file.read_text(encoding="ascii"))
                self.assertTrue(process_is_live(descendant_pid))

                gate.terminate_group(leader)

                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    if not process_is_live(descendant_pid):
                        break
                    time.sleep(0.02)
                else:
                    self.fail("background descendant survived tracked process-group cleanup")
            finally:
                try:
                    os.killpg(leader.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass

    def test_graceful_termination_targets_only_child_group(self):
        child = FakeProcess()
        killpg = mock.Mock()
        group_exists = iter((True, False, False))

        gate.terminate_group(
            child,
            killpg=killpg,
            group_exists=lambda pgid: next(group_exists),
            now=lambda: 0.0,
            sleep=lambda seconds: None,
        )

        killpg.assert_called_once_with(4242, signal.SIGTERM)
        self.assertEqual(child.wait_calls, [None])

    def test_forced_termination_kills_only_child_group_after_ten_seconds(self):
        child = FakeProcess()
        killpg = mock.Mock()

        gate.terminate_group(
            child,
            killpg=killpg,
            group_exists=lambda pgid: True,
            now=iter((0.0, 10.0)).__next__,
            sleep=lambda seconds: None,
        )

        self.assertEqual(
            killpg.call_args_list,
            [mock.call(4242, signal.SIGTERM), mock.call(4242, signal.SIGKILL)],
        )
        self.assertEqual(child.wait_calls, [None])


class RunPhaseTests(unittest.TestCase):
    def test_cancelled_attach_cleanup_ignores_cancellable_sleep_and_returns_130(self):
        cancellation = gate.CancellationState()
        clock = [0.0]
        signals = []
        group_signals = []
        child = mock.Mock(pid=700)
        child.poll.return_value = None
        child.wait.return_value = -signal.SIGKILL
        boundary = cancelling_partial_boundary(signals, cancellation, clock)

        with mock.patch.object(
            gate.os,
            "killpg",
            side_effect=lambda pgid, signum: group_signals.append((pgid, signum)),
        ), mock.patch.object(
            boundary, "_reap_adopted", wraps=boundary._reap_adopted
        ) as reap:
            with self.assertRaises(gate.Cancelled) as raised:
                gate.run_phase(
                    "build", pathlib.Path("/trusted/build_blue.sh"),
                    pathlib.Path("/source"), pathlib.Path("/artifacts"), {},
                    timeout_seconds=60, read_snapshot=lambda: snapshot(),
                    now=lambda: clock[0],
                    sleep=lambda seconds: (_ for _ in ()).throw(
                        gate.Cancelled("cancellable sleep")
                    ),
                    cancellation=cancellation, popen=lambda *args, **kwargs: child,
                    process_boundary_factory=lambda: boundary,
                )

        self.assertEqual(gate.exit_code_for(raised.exception), 130)
        self.assertIn((701, signal.SIGTERM), signals)
        self.assertIn((701, signal.SIGKILL), signals)
        self.assertEqual(
            group_signals,
            [(700, signal.SIGTERM), (700, signal.SIGKILL)],
        )
        child.wait.assert_called_once_with(timeout=2)
        reap.assert_called()

    def test_partial_phase_attach_aborts_tracked_and_live_root_group(self):
        child = mock.Mock(pid=700)
        child.poll.return_value = None
        child.wait.return_value = -signal.SIGKILL
        signals = []
        group_signals = []
        boundary = partial_boundary(signals)

        with mock.patch.object(
            gate.os,
            "killpg",
            side_effect=lambda pgid, signum: group_signals.append((pgid, signum)),
        ):
            with self.assertRaises(gate.InfrastructureFailure) as raised:
                gate.run_phase(
                    "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                    pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                    read_snapshot=lambda: snapshot(), now=lambda: 0.0,
                    sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                    popen=lambda *args, **kwargs: child,
                    process_boundary_factory=lambda: boundary,
                )

        self.assertIn("partial descendant", str(raised.exception))
        self.assertIn((701, signal.SIGKILL), signals)
        self.assertEqual(
            group_signals,
            [(700, signal.SIGTERM), (700, signal.SIGKILL)],
        )

    def test_phase_boundary_attach_time_counts_against_hard_deadline(self):
        clock = [0.0]
        child = FakeProcess(polls=(0,), returncode=0)
        boundary = mock.Mock()
        boundary.attach.side_effect = lambda *args, **kwargs: clock.__setitem__(0, 5.0)

        with self.assertRaises(gate.CodeFailure):
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(), now=lambda: clock[0],
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child,
                process_boundary_factory=lambda: boundary,
            )

        boundary.abort.assert_called_once()

    def test_linux_boundary_attach_failure_terminates_spawned_phase_child_once(self):
        child = FakeProcess(polls=(None,))
        boundary = mock.Mock()
        boundary.attach.side_effect = gate.InfrastructureFailure("root identity missing")
        terminate = mock.Mock()

        with self.assertRaises(gate.InfrastructureFailure) as raised:
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(), now=lambda: 0.0,
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child, terminate=terminate,
                process_boundary_factory=lambda: boundary,
            )

        self.assertIn("root identity missing", str(raised.exception))
        boundary.abort.assert_called_once()
        terminate.assert_not_called()

    def test_child_exit_observed_after_deadline_is_timeout_not_success(self):
        child = FakeProcess(polls=(0,), returncode=0)
        terminate = mock.Mock()
        clock = iter((0.0, 5.0))

        with self.assertRaises(gate.CodeFailure):
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(), now=lambda: next(clock),
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child, terminate=terminate,
            )

        self.assertEqual(child._polls, [0])
        terminate.assert_called_once_with(child)

    def test_phase_termination_failure_overrides_and_preserves_primary_code_failure(self):
        child = FakeProcess(polls=(7,), returncode=7)

        with self.assertRaises(gate.InfrastructureFailure) as raised:
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(), now=lambda: 0.0,
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child,
                terminate=lambda process: (_ for _ in ()).throw(OSError("cannot reap group")),
            )

        self.assertEqual(gate.exit_code_for(raised.exception), 23)
        self.assertIn("code-or-test-failure", str(raised.exception))
        self.assertIn("status 7", str(raised.exception))
        self.assertIn("cannot reap group", str(raised.exception))

    def test_completed_hardware_child_is_classified_before_new_foreign_snapshot(self):
        child = FakeProcess(polls=(0,), returncode=0)
        reads = mock.Mock(side_effect=(snapshot(), snapshot(process("alice"))))

        result = gate.run_phase(
            "hardware", pathlib.Path("/trusted/run_hardware.sh"), pathlib.Path("/source"),
            pathlib.Path("/artifacts"), {}, timeout_seconds=5,
            read_snapshot=reads, now=lambda: 0.0, sleep=lambda seconds: None,
            cancellation=gate.CancellationState(), popen=lambda *args, **kwargs: child,
            terminate=lambda process: None,
        )

        self.assertEqual(result, 0)
        self.assertEqual(reads.call_count, 1)

    def test_running_hardware_child_still_detects_foreign_snapshot(self):
        child = FakeProcess(polls=(None, None))
        reads = mock.Mock(side_effect=(snapshot(), snapshot(process("alice"))))

        with self.assertRaises(gate.ResourceCollision):
            gate.run_phase(
                "hardware", pathlib.Path("/trusted/run_hardware.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=reads, now=lambda: 0.0, sleep=lambda seconds: None,
                cancellation=gate.CancellationState(), popen=lambda *args, **kwargs: child,
                terminate=lambda process: None,
            )

    def run_phase(self, child, states=None, clock=None, cancelled=None):
        popen = mock.Mock(return_value=child)
        terminate = mock.Mock()
        states = iter(states or (snapshot(), snapshot()))
        clock = iter(clock or (0.0, 1.0, 1.0))
        cancellation = cancelled or gate.CancellationState()
        result = gate.run_phase(
            "build",
            pathlib.Path("/trusted/build_blue.sh"),
            pathlib.Path("/source"),
            pathlib.Path("/artifacts"),
            {"PATH": "/usr/bin"},
            timeout_seconds=7200,
            read_snapshot=lambda: next(states),
            now=lambda: next(clock),
            sleep=lambda seconds: None,
            cancellation=cancellation,
            popen=popen,
            terminate=terminate,
        )
        return result, popen, terminate

    def test_launches_trusted_script_in_source_as_new_session(self):
        child = FakeProcess(polls=(0,), returncode=0)
        result, popen, terminate = self.run_phase(child)

        self.assertEqual(result, 0)
        popen.assert_called_once_with(
            ["/trusted/build_blue.sh", "/source", "/artifacts"],
            cwd="/source",
            env={"PATH": "/usr/bin"},
            start_new_session=True,
        )
        terminate.assert_called_once_with(child)

    def test_nonzero_child_exit_is_code_failure(self):
        with self.assertRaises(gate.CodeFailure):
            self.run_phase(FakeProcess(polls=(7,), returncode=7))

    def test_all_raw_nonzero_phase_statuses_are_code_failures(self):
        for status in (23, 126, 127):
            with self.subTest(status=status), self.assertRaises(gate.CodeFailure) as raised:
                self.run_phase(FakeProcess(polls=(status,), returncode=status))
            self.assertEqual(gate.exit_code_for(raised.exception), 20)

    def test_timeout_terminates_child(self):
        child = FakeProcess(polls=(None,))
        terminate = mock.Mock()
        with self.assertRaises(gate.CodeFailure):
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(), now=iter((0.0, 6.0)).__next__,
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child, terminate=terminate,
            )
        terminate.assert_called_once_with(child)

    def test_snapshot_that_consumes_remaining_deadline_terminates_child(self):
        child = FakeProcess(polls=(None, None))
        terminate = mock.Mock()
        clock = [0.0]
        snapshots = iter((snapshot(),))

        def deadline_snapshot(deadline):
            self.assertEqual(deadline, 5.0)
            clock[0] = deadline
            return snapshot()

        with self.assertRaises(gate.CodeFailure):
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: next(snapshots),
                read_snapshot_with_deadline=deadline_snapshot,
                now=lambda: clock[0], sleep=lambda seconds: None,
                cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child, terminate=terminate,
            )
        terminate.assert_called_once_with(child)

    def test_cancellation_terminates_child(self):
        child = FakeProcess(polls=(None,))
        cancellation = gate.CancellationState()
        terminate = mock.Mock()

        def popen(*args, **kwargs):
            cancellation.cancel()
            return child

        with self.assertRaises(gate.Cancelled):
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(), now=lambda: 0.0,
                sleep=lambda seconds: None, cancellation=cancellation,
                popen=popen, terminate=terminate,
            )
        terminate.assert_called_once_with(child)

    def test_unhealthy_build_preflight_prevents_child_launch(self):
        child = FakeProcess(polls=(None,))
        popen = mock.Mock(return_value=child)
        terminate = mock.Mock()
        with self.assertRaises(gate.InfrastructureFailure) as raised:
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(healthy=False), now=lambda: 0.0,
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=popen, terminate=terminate,
            )
        self.assertEqual(gate.exit_code_for(raised.exception), 23)
        popen.assert_not_called()
        terminate.assert_not_called()

    def test_unhealthy_build_state_after_launch_terminates_child(self):
        child = FakeProcess(polls=(None, None))
        states = iter((snapshot(), snapshot(healthy=False)))
        terminate = mock.Mock()
        with self.assertRaises(gate.InfrastructureFailure):
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: next(states), now=lambda: 0.0,
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child, terminate=terminate,
            )
        terminate.assert_called_once_with(child)

    def test_hardware_foreign_collision_terminates_child(self):
        child = FakeProcess(polls=(None, None))
        states = iter((snapshot(), snapshot(process("unknown"))))
        terminate = mock.Mock()
        with self.assertRaises(gate.ResourceCollision):
            gate.run_phase(
                "hardware", pathlib.Path("/trusted/run_hardware.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: next(states), now=lambda: 0.0,
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=lambda *args, **kwargs: child, terminate=terminate,
            )
        terminate.assert_called_once_with(child)

class CancellationTests(unittest.TestCase):
    def test_cancellation_aware_sleep_wakes_when_cancelled(self):
        state = gate.CancellationState()
        state.cancel()
        with self.assertRaises(gate.Cancelled):
            gate.cancellation_aware_sleep(60, state)

    def test_exit_mapping_is_centralized(self):
        self.assertEqual(gate.exit_code_for(gate.CodeFailure("x")), 20)
        self.assertEqual(gate.exit_code_for(npu_state.ResourceTimeout("x")), 21)
        self.assertEqual(gate.exit_code_for(gate.ResourceCollision("x")), 22)
        self.assertEqual(gate.exit_code_for(gate.InfrastructureFailure("x")), 23)
        unhealthy = npu_state.UnhealthyState("persistent unhealthy NPU state")
        self.assertEqual(gate.exit_code_for(unhealthy), 23)
        self.assertIs(gate._as_gate_failure(unhealthy), unhealthy)
        self.assertEqual(gate.exit_code_for(gate.Cancelled("x")), 130)
        self.assertEqual(gate.exit_code_for(gate.ObsoleteRun("x")), 130)

    def test_signal_handler_sets_cancellation_state_and_is_restored(self):
        state = gate.CancellationState()
        previous = signal.getsignal(signal.SIGTERM)
        with gate._signal_cancellation(state):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            self.assertTrue(state.cancelled)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

class SimplifiedOrchestrationTests(unittest.TestCase):
    def test_build_wait_and_hardware_run_without_token_refetch_or_lock(self):
        events = []

        def phase(name, *args, **kwargs):
            events.append(name)

        def wait(**kwargs):
            events.append("wait")
            return snapshot()

        gate.orchestrate(
            gate.Config(
                pathlib.Path("/source"),
                pathlib.Path("/scratch"),
                "merge",
                "LingquLab/TileXR",
                42,
            ),
            control_dir=pathlib.Path("/trusted"),
            cancellation=gate.CancellationState(),
            run_phase_fn=phase,
            wait_for_idle_fn=wait,
            read_snapshot_fn=lambda: events.append("cleanup") or snapshot(),
            verify_merge_sha_fn=lambda source, expected: events.append("verify") or expected,
            sleep_fn=lambda seconds: None,
            emit=lambda line: None,
        )

        self.assertEqual(events, ["verify", "build", "wait", "hardware", "cleanup"])

class QueueDeadlineTests(unittest.TestCase):
    def test_idle_result_at_deadline_cannot_launch_hardware(self):
        clock = [0.0]
        phases = []

        def wait(**kwargs):
            clock[0] = gate.QUEUE_TIMEOUT_SECONDS
            return snapshot()

        with self.assertRaises(npu_state.ResourceTimeout):
            gate.orchestrate(
                gate.Config(pathlib.Path("/source"), pathlib.Path("/scratch"), "merge", "LingquLab/TileXR", 42),
                control_dir=pathlib.Path("/trusted"),
                cancellation=gate.CancellationState(),
                run_phase_fn=lambda name, *args, **kwargs: phases.append(name),
                wait_for_idle_fn=wait,
                read_snapshot_fn=lambda: snapshot(),
                verify_merge_sha_fn=lambda source, expected: expected,
                resource_now=lambda: clock[0],
                sleep_fn=lambda seconds: None,
                emit=lambda line: None,
            )

        self.assertEqual(phases, ["build"])

    def test_cancellation_interrupts_idle_wait(self):
        cancellation = gate.CancellationState()
        phases = []

        def wait(**kwargs):
            cancellation.cancel()
            kwargs["sleep"](60)

        with self.assertRaises(gate.Cancelled):
            gate.orchestrate(
                gate.Config(pathlib.Path("/source"), pathlib.Path("/scratch"), "merge", "LingquLab/TileXR", 42),
                control_dir=pathlib.Path("/trusted"),
                cancellation=cancellation,
                run_phase_fn=lambda name, *args, **kwargs: phases.append(name),
                wait_for_idle_fn=wait,
                read_snapshot_fn=lambda: snapshot(),
                verify_merge_sha_fn=lambda source, expected: expected,
                emit=lambda line: None,
            )

        self.assertEqual(phases, ["build"])


class CleanupTests(unittest.TestCase):
    def test_final_cleanup_polls_read_only_during_bounded_driver_grace(self):
        states = iter(
            (
                snapshot(process("tilexr-ci", 500)),
                snapshot(process("tilexr-ci", 500)),
                snapshot(),
            )
        )
        clock = iter((0.0, 0.0, 0.5, 0.5, 1.0))
        sleeps = []

        gate.verify_final_cleanup(
            lambda: next(states),
            sleep=sleeps.append,
            now=lambda: next(clock),
            grace_seconds=2.0,
            poll_seconds=0.5,
        )

        self.assertEqual(sleeps, [0.5, 0.5])

    def test_leftover_ci_process_fails_without_signalling_any_pid(self):
        with mock.patch.object(gate.os, "killpg") as killpg:
            with self.assertRaises(gate.InfrastructureFailure):
                gate.verify_final_cleanup(
                    lambda: snapshot(process("tilexr-ci", 500), process("alice", 600)),
                    grace_seconds=0,
                )
            killpg.assert_not_called()

    def test_foreign_process_is_not_a_cleanup_failure(self):
        gate.verify_final_cleanup(lambda: snapshot(process("alice", 600)))

    def test_cleanup_failure_overrides_primary_failure_and_preserves_detail(self):
        primary = gate.CodeFailure("tests failed")
        cleanup = gate.InfrastructureFailure("tilexr-ci pid 500 remains")
        final = gate.prefer_cleanup_failure(primary, cleanup)
        self.assertIsInstance(final, gate.InfrastructureFailure)
        self.assertIn("tests failed", str(final))
        self.assertIn("tilexr-ci pid 500 remains", str(final))
        self.assertIs(gate.select_final_failure(final, gate.Cancelled("late signal")), final)

class ReportTests(unittest.TestCase):
    def test_initial_report_reads_pr_identity(self):
        config = gate.Config(pathlib.Path("/source"), pathlib.Path("/scratch"), "merge", "LingquLab/TileXR", 42)
        report = gate.initial_report(
            config,
            {"TILEXR_CI_HEAD_SHA": "head", "TILEXR_CI_BASE_SHA": "base"},
        )
        self.assertEqual(
            (report.pr_number, report.head_sha, report.base_sha, report.merge_sha),
            (42, "head", "base", "merge"),
        )

    def test_report_is_plain_log_text(self):
        report = gate.GateReport(
            pr_number=42,
            head_sha="head",
            base_sha="base",
            merge_sha="merge",
            build_outcome="passed",
            queue_outcome="passed",
            hardware_outcome="failed",
            wait_seconds=65.0,
            execution_seconds=3.75,
            failure_class="code-or-test-failure",
            failure_detail="tests failed",
        )

        text = gate.format_report(report)

        for expected in (
            "PR: #42",
            "Head SHA: head",
            "Base SHA: base",
            "Merge SHA: merge",
            "Build outcome: passed",
            "NPU wait outcome: passed",
            "Hardware outcome: failed",
            "NPU wait seconds: 65.0",
            "Hardware execution seconds: 3.8",
            "Failure class: code-or-test-failure",
            "Failure detail: tests failed",
            "UDMA data plane: out of scope on 910B3",
            "Multi-host validation: out of scope for this gate",
        ):
            self.assertIn(expected, text)


class ControllerTests(unittest.TestCase):
    def argv(self, scratch):
        return [
            "--source", "/source",
            "--scratch", str(scratch),
            "--expected-merge-sha", "merge",
            "--repository", "LingquLab/TileXR",
            "--pr-number", "42",
        ]

    def test_cli_uses_scratch_and_has_no_token_input(self):
        config = gate.parse_config(self.argv("/scratch"))
        self.assertEqual(config.scratch, pathlib.Path("/scratch"))
        with self.assertRaises(gate.InfrastructureFailure):
            gate.parse_config(self.argv("/scratch") + ["--github-token-fd", "3"])

    def test_cli_preserves_path_identity_until_controller_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            argv = self.argv(root / "scratch")
            argv[1] = str(source_link)

            config = gate.parse_config(argv)

            self.assertEqual(config.source, source_link.absolute())
            self.assertTrue(config.source.is_symlink())

    def test_controller_disables_process_dumpability_before_running_body(self):
        events = []
        config = gate.Config(
            pathlib.Path("/source"),
            pathlib.Path("/scratch"),
            "merge",
            "LingquLab/TileXR",
            42,
        )
        with mock.patch.object(
            gate,
            "_run_controller_body",
            side_effect=lambda *args: events.append("body") or 0,
        ):
            result = gate.run_controller(
                config,
                environ={},
                harden_process=lambda: events.append("harden"),
            )

        self.assertEqual(result, 0)
        self.assertEqual(events, ["harden", "body"])

    def test_dumpability_hardening_fails_closed_on_prctl_error(self):
        prctl = mock.Mock(return_value=-1)

        with self.assertRaises(gate.InfrastructureFailure):
            gate.disable_process_dumpability(
                platform="linux",
                prctl=prctl,
                get_errno=lambda: errno.EPERM,
            )

        prctl.assert_called_once_with(gate.PR_SET_DUMPABLE, 0, 0, 0, 0)

    def test_dumpability_hardening_requires_linux(self):
        with self.assertRaises(gate.InfrastructureFailure):
            gate.disable_process_dumpability(
                platform="darwin",
                prctl=lambda *args: 0,
            )

    def test_main_classifies_dumpability_hardening_failure(self):
        stderr = io.StringIO()
        with mock.patch.object(
            gate,
            "run_controller",
            side_effect=gate.InfrastructureFailure("prctl failed"),
        ), mock.patch.object(gate.sys, "stderr", stderr):
            result = gate.main(self.argv("/scratch"), environ={})

        self.assertEqual(result, 23)
        self.assertIn("prctl failed", stderr.getvalue())

    def test_controller_success_only_prints_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            scratch = root / "scratch"
            config = gate.Config(source, scratch, "merge", "LingquLab/TileXR", 42)
            output = io.StringIO()
            with mock.patch.object(gate, "orchestrate"), mock.patch.object(
                gate, "verify_final_cleanup"
            ), mock.patch.object(gate.sys, "stdout", output):
                result = gate._run_controller_body(
                    config, {}, pathlib.Path("/trusted"), gate.CancellationState()
                )

            self.assertEqual(result, 0)
            self.assertEqual(list(scratch.iterdir()), [])
            self.assertIn("Failure class: none", output.getvalue())

    def test_controller_rejects_source_symlink_before_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            config = gate.Config(
                source_link,
                root / "scratch",
                "merge",
                "LingquLab/TileXR",
                42,
            )
            with mock.patch.object(gate, "orchestrate") as orchestrate, mock.patch.object(
                gate.sys, "stdout", io.StringIO()
            ), mock.patch.object(gate.sys, "stderr", io.StringIO()):
                result = gate._run_controller_body(
                    config, {}, pathlib.Path("/trusted"), gate.CancellationState()
                )

        self.assertEqual(result, 23)
        orchestrate.assert_not_called()

    def test_controller_rejects_scratch_symlink_before_orchestration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source"
            source.mkdir()
            scratch_target = root / "scratch-target"
            scratch_target.mkdir()
            scratch_link = root / "scratch-link"
            scratch_link.symlink_to(scratch_target, target_is_directory=True)
            config = gate.Config(
                source,
                scratch_link,
                "merge",
                "LingquLab/TileXR",
                42,
            )
            with mock.patch.object(gate, "orchestrate") as orchestrate, mock.patch.object(
                gate.sys, "stdout", io.StringIO()
            ), mock.patch.object(gate.sys, "stderr", io.StringIO()):
                result = gate._run_controller_body(
                    config, {}, pathlib.Path("/trusted"), gate.CancellationState()
                )

        self.assertEqual(result, 23)
        orchestrate.assert_not_called()

    def test_controller_preserves_code_failure_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source"
            source.mkdir()
            config = gate.Config(
                source,
                pathlib.Path(directory) / "scratch",
                "merge",
                "LingquLab/TileXR",
                42,
            )
            with mock.patch.object(
                gate, "orchestrate", side_effect=gate.CodeFailure("tests failed")
            ), mock.patch.object(gate, "verify_final_cleanup"), mock.patch.object(
                gate.sys, "stdout", io.StringIO()
            ), mock.patch.object(gate.sys, "stderr", io.StringIO()):
                result = gate._run_controller_body(
                    config, {}, pathlib.Path("/trusted"), gate.CancellationState()
                )
        self.assertEqual(result, 20)

    def test_cancelled_finalization_returns_130(self):
        cancellation = gate.CancellationState()
        cancellation.cancel()
        with tempfile.TemporaryDirectory() as directory:
            source = pathlib.Path(directory) / "source"
            source.mkdir()
            config = gate.Config(
                source,
                pathlib.Path(directory) / "scratch",
                "merge",
                "LingquLab/TileXR",
                42,
            )
            with mock.patch.object(gate, "orchestrate"), mock.patch.object(
                gate, "verify_final_cleanup"
            ), mock.patch.object(gate.sys, "stdout", io.StringIO()), mock.patch.object(
                gate.sys, "stderr", io.StringIO()
            ):
                result = gate._run_controller_body(
                    config, {}, pathlib.Path("/trusted"), cancellation
                )
        self.assertEqual(result, 130)

    def test_repository_and_pr_number_are_strictly_validated(self):
        with self.assertRaises(gate.InfrastructureFailure):
            gate.validate_config(
                gate.Config(pathlib.Path("/s"), pathlib.Path("/scratch"), "sha", "other/repo", 1)
            )
        with self.assertRaises(gate.InfrastructureFailure):
            gate.validate_config(
                gate.Config(pathlib.Path("/s"), pathlib.Path("/scratch"), "sha", "LingquLab/TileXR", 0)
            )


if __name__ == "__main__":
    unittest.main()
