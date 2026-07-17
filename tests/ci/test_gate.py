#!/usr/bin/env python3
import contextlib
import io
import json
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
        self.assertEqual(
            gate.policy_violation("build", state, "ci-special"),
            "ci-npu-before-lock",
        )
        self.assertIsNone(gate.policy_violation("build", state, "tilexr-ci"))

    def test_build_rejects_ci_npu_process_before_lock(self):
        state = snapshot(process("tilexr-ci"))
        self.assertEqual(gate.policy_violation("build", state), "ci-npu-before-lock")

    def test_build_permits_foreign_npu_process(self):
        self.assertIsNone(gate.policy_violation("build", snapshot(process("alice"))))

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

        phase = gate.phase_environment(child, "/source", "/artifacts", 42)
        self.assertEqual(phase["PR_NUMBER"], "42")
        self.assertEqual(phase["TILEXR_CI_SOURCE_DIR"], "/source")
        self.assertEqual(phase["TILEXR_CI_ARTIFACT_DIR"], "/artifacts")
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


class TerminationTests(unittest.TestCase):
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
                os.kill(descendant_pid, 0)

                gate.terminate_group(leader)

                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    try:
                        os.kill(descendant_pid, 0)
                    except ProcessLookupError:
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

    def test_build_policy_preflight_prevents_child_launch(self):
        child = FakeProcess(polls=(None,))
        popen = mock.Mock(return_value=child)
        terminate = mock.Mock()
        with self.assertRaises(gate.ResourceCollision) as raised:
            gate.run_phase(
                "build", pathlib.Path("/trusted/build_blue.sh"), pathlib.Path("/source"),
                pathlib.Path("/artifacts"), {}, timeout_seconds=5,
                read_snapshot=lambda: snapshot(process("tilexr-ci")), now=lambda: 0.0,
                sleep=lambda seconds: None, cancellation=gate.CancellationState(),
                popen=popen, terminate=terminate,
            )
        self.assertEqual(gate.exit_code_for(raised.exception), 22)
        popen.assert_not_called()
        terminate.assert_not_called()

    def test_build_policy_violation_after_launch_terminates_child(self):
        child = FakeProcess(polls=(None, None))
        states = iter((snapshot(), snapshot(process("tilexr-ci"))))
        terminate = mock.Mock()
        with self.assertRaises(gate.ResourceCollision):
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


class FetchMergeTests(unittest.TestCase):
    def response(self, payload):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(payload).encode("utf-8")
        return response

    def test_fetch_uses_authenticated_versioned_github_request(self):
        payload = {
            "url": "https://api.github.com/repos/LingquLab/TileXR/pulls/42",
            "number": 42,
            "state": "open",
            "head": {"sha": "head-sha", "ref": "feature", "repo": {"full_name": "fork/TileXR"}},
            "base": {"sha": "base-sha", "ref": "main", "repo": {"full_name": "LingquLab/TileXR"}},
            "merge_commit_sha": "merge-sha",
        }
        urlopen = mock.Mock(return_value=self.response(payload))

        sha = gate.fetch_current_merge_sha(42, "read-token", urlopen=urlopen)

        self.assertEqual(sha, "merge-sha")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, payload["url"])
        self.assertEqual(request.get_header("Authorization"), "Bearer read-token")
        self.assertEqual(request.get_header("Accept"), "application/vnd.github+json")
        self.assertEqual(request.get_header("X-github-api-version"), "2022-11-28")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 10)

    def test_fetch_errors_are_infrastructure_failures(self):
        cases = (
            lambda request, timeout: (_ for _ in ()).throw(OSError("offline")),
            lambda request, timeout: self.response("not-an-object"),
            lambda request, timeout: self.response({"number": 42}),
        )
        for urlopen in cases:
            with self.subTest(urlopen=urlopen), self.assertRaises(gate.InfrastructureFailure):
                gate.fetch_current_merge_sha(42, "token", urlopen=urlopen)


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
        self.assertEqual(gate.exit_code_for(gate.Cancelled("x")), 130)
        self.assertEqual(gate.exit_code_for(gate.ObsoleteRun("x")), 130)

    def test_signal_handler_sets_cancellation_state_and_is_restored(self):
        state = gate.CancellationState()
        previous = signal.getsignal(signal.SIGTERM)
        with gate._signal_cancellation(state):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            self.assertTrue(state.cancelled)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_cancellation_during_final_step_summary_returns_130(self):
        cancellation = gate.CancellationState()
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            step_summary = root / "step-summary.md"
            step_summary.touch()
            config = gate.Config(
                root / "source", root / "artifacts", "merge", "LingquLab/TileXR", 42
            )
            writes = []

            def write_summary(*args, **kwargs):
                writes.append(kwargs.get("step_summary"))
                if kwargs.get("step_summary") is not None:
                    cancellation.cancel()
                return "summary"

            with mock.patch.object(gate, "orchestrate"), \
                 mock.patch.object(gate, "verify_final_cleanup"), \
                 mock.patch.object(gate, "invoke_collector"), \
                 mock.patch.object(gate, "write_summary", side_effect=write_summary):
                result = gate._run_controller_body(
                    config,
                    {
                        "TILEXR_CI_GITHUB_TOKEN": "token",
                        "GITHUB_STEP_SUMMARY": str(step_summary),
                    },
                    root / "trusted",
                    cancellation,
                )

        self.assertEqual(result, 130)
        self.assertGreaterEqual(len(writes), 3)


class LockOrchestrationTests(unittest.TestCase):
    def test_build_precedes_lock_and_lock_wraps_wait_revalidation_and_hardware(self):
        events = []

        @contextlib.contextmanager
        def lock(**kwargs):
            events.append("lock-enter")
            try:
                yield
            finally:
                events.append("lock-exit")

        def phase(name, *args, **kwargs):
            events.append(name)
            return 0

        def wait(**kwargs):
            events.append("wait")
            self.assertGreater(kwargs["max_wait_seconds"], 0)
            self.assertLessEqual(kwargs["max_wait_seconds"], 21600)
            self.assertEqual(kwargs["poll_seconds"], 60)
            self.assertEqual(kwargs["stable_samples"], 2)
            self.assertIs(kwargs["now"], gate.time.monotonic)
            return snapshot()

        state = lambda: events.append("cleanup") or snapshot()
        gate.orchestrate(
            gate.Config(pathlib.Path("/source"), pathlib.Path("/artifacts"), "merge", "LingquLab/TileXR", 42),
            control_dir=pathlib.Path("/trusted"), token="token", cancellation=gate.CancellationState(),
            run_phase_fn=phase, wait_for_idle_fn=wait, read_snapshot_fn=state,
            fetch_merge_sha_fn=lambda pr, token: events.append("refetch") or "merge",
            verify_merge_sha_fn=lambda source, expected: events.append("verify") or expected,
            lock_fn=lock, sleep_fn=lambda seconds: None, emit=lambda line: None,
        )

        self.assertEqual(
            events,
            ["verify", "build", "lock-enter", "wait", "refetch", "hardware", "cleanup", "lock-exit"],
        )

    def test_changed_merge_is_obsolete_and_hardware_never_launches(self):
        phases = []

        @contextlib.contextmanager
        def lock(**kwargs):
            yield

        with self.assertRaises(gate.ObsoleteRun):
            gate.orchestrate(
                gate.Config(pathlib.Path("/source"), pathlib.Path("/artifacts"), "old", "LingquLab/TileXR", 42),
                control_dir=pathlib.Path("/trusted"), token="token", cancellation=gate.CancellationState(),
                run_phase_fn=lambda name, *args, **kwargs: phases.append(name) or 0,
                wait_for_idle_fn=lambda **kwargs: snapshot(), read_snapshot_fn=lambda: snapshot(),
                fetch_merge_sha_fn=lambda pr, token: "new",
                verify_merge_sha_fn=lambda source, expected: expected,
                lock_fn=lock, sleep_fn=lambda seconds: None, emit=lambda line: None,
            )

        self.assertEqual(phases, ["build"])

    def test_lock_and_idle_wait_share_one_resource_budget(self):
        clock = [100.0]
        observed = {}

        @contextlib.contextmanager
        def lock(**kwargs):
            clock[0] += 120.0
            yield

        def wait(**kwargs):
            observed["remaining"] = kwargs["max_wait_seconds"]
            clock[0] += 30.0
            return snapshot()

        gate.orchestrate(
            gate.Config(pathlib.Path("/source"), pathlib.Path("/artifacts"), "merge", "LingquLab/TileXR", 42),
            control_dir=pathlib.Path("/trusted"), token="token", cancellation=gate.CancellationState(),
            run_phase_fn=lambda name, *args, **kwargs: 0,
            wait_for_idle_fn=wait, read_snapshot_fn=lambda: snapshot(),
            fetch_merge_sha_fn=lambda pr, token: "merge",
            verify_merge_sha_fn=lambda source, expected: expected,
            lock_fn=lock, emit=lambda line: None, resource_now=lambda: clock[0],
        )
        self.assertEqual(observed["remaining"], 21600 - 120)

    def test_exhausted_lock_budget_prevents_idle_wait_and_hardware(self):
        clock = [0.0]
        phases = []
        waiter = mock.Mock()

        @contextlib.contextmanager
        def lock(**kwargs):
            clock[0] = 21600.0
            yield

        with self.assertRaises(npu_state.ResourceTimeout):
            gate.orchestrate(
                gate.Config(pathlib.Path("/source"), pathlib.Path("/artifacts"), "merge", "LingquLab/TileXR", 42),
                control_dir=pathlib.Path("/trusted"), token="token", cancellation=gate.CancellationState(),
                run_phase_fn=lambda name, *args, **kwargs: phases.append(name) or 0,
                wait_for_idle_fn=waiter, read_snapshot_fn=lambda: snapshot(),
                fetch_merge_sha_fn=lambda pr, token: "merge",
                verify_merge_sha_fn=lambda source, expected: expected,
                lock_fn=lock, emit=lambda line: None, resource_now=lambda: clock[0],
            )
        waiter.assert_not_called()
        self.assertEqual(phases, ["build"])

    def test_snapshot_at_end_of_shared_budget_cannot_launch_hardware(self):
        clock = [0.0]
        phases = []
        observed = {}

        @contextlib.contextmanager
        def lock(**kwargs):
            observed["lock_budget"] = kwargs["max_wait_seconds"]
            clock[0] = 21599.0
            yield

        def queue_snapshot(*, deadline, now):
            observed["deadline"] = deadline
            clock[0] = deadline
            return snapshot()

        with self.assertRaises(npu_state.ResourceTimeout):
            gate.orchestrate(
                gate.Config(pathlib.Path("/source"), pathlib.Path("/artifacts"), "merge", "LingquLab/TileXR", 42),
                control_dir=pathlib.Path("/trusted"), token="token", cancellation=gate.CancellationState(),
                run_phase_fn=lambda name, *args, **kwargs: phases.append(name) or 0,
                read_snapshot_fn=lambda: snapshot(), queue_snapshot_fn=queue_snapshot,
                fetch_merge_sha_fn=lambda pr, token: "merge",
                verify_merge_sha_fn=lambda source, expected: expected,
                lock_fn=lock, sleep_fn=lambda seconds: None,
                emit=lambda line: None, resource_now=lambda: clock[0],
            )

        self.assertEqual(observed, {"lock_budget": 21600.0, "deadline": 21600.0})
        self.assertEqual(clock[0], 21600.0)
        self.assertEqual(phases, ["build"])

    def test_cancellation_interrupts_resource_wait_and_maps_to_130(self):
        phases = []
        cancellation = gate.CancellationState()

        @contextlib.contextmanager
        def lock(**kwargs):
            yield

        def wait(**kwargs):
            cancellation.cancel()
            kwargs["sleep"](60)

        with self.assertRaises(gate.Cancelled) as raised:
            gate.orchestrate(
                gate.Config(pathlib.Path("/source"), pathlib.Path("/artifacts"), "merge", "LingquLab/TileXR", 42),
                control_dir=pathlib.Path("/trusted"), token="token", cancellation=cancellation,
                run_phase_fn=lambda name, *args, **kwargs: phases.append(name) or 0,
                wait_for_idle_fn=wait, read_snapshot_fn=lambda: snapshot(),
                fetch_merge_sha_fn=lambda pr, token: "merge",
                verify_merge_sha_fn=lambda source, expected: expected,
                lock_fn=lock, emit=lambda line: None,
            )

        self.assertEqual(gate.exit_code_for(raised.exception), 130)
        self.assertEqual(phases, ["build"])

    def test_cancellation_after_merge_refetch_prevents_hardware_launch(self):
        phases = []
        cancellation = gate.CancellationState()

        @contextlib.contextmanager
        def lock(**kwargs):
            yield

        def fetch(pr_number, token):
            cancellation.cancel()
            return "merge"

        with self.assertRaises(gate.Cancelled):
            gate.orchestrate(
                gate.Config(pathlib.Path("/source"), pathlib.Path("/artifacts"), "merge", "LingquLab/TileXR", 42),
                control_dir=pathlib.Path("/trusted"), token="token", cancellation=cancellation,
                run_phase_fn=lambda name, *args, **kwargs: phases.append(name) or 0,
                wait_for_idle_fn=lambda **kwargs: snapshot(), read_snapshot_fn=lambda: snapshot(),
                fetch_merge_sha_fn=fetch, verify_merge_sha_fn=lambda source, expected: expected,
                lock_fn=lock, emit=lambda line: None,
            )
        self.assertEqual(phases, ["build"])


class LockTests(unittest.TestCase):
    def test_lock_contention_has_a_finite_resource_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            lock_path = pathlib.Path(directory) / "npu.lock"
            with mock.patch.object(gate.fcntl, "flock", side_effect=BlockingIOError):
                with self.assertRaises(npu_state.ResourceTimeout):
                    with gate.npu_lock(
                        lock_path,
                        cancellation=gate.CancellationState(),
                        max_wait_seconds=2,
                        now=iter((0.0, 1.0, 2.0)).__next__,
                        sleep=lambda seconds: None,
                    ):
                        self.fail("contended lock must not be entered")

    def test_lock_contention_sleep_is_cancellation_aware(self):
        cancellation = gate.CancellationState()
        cancellation.cancel()
        with tempfile.TemporaryDirectory() as directory:
            lock_path = pathlib.Path(directory) / "npu.lock"
            with mock.patch.object(gate.fcntl, "flock", side_effect=BlockingIOError):
                with self.assertRaises(gate.Cancelled):
                    with gate.npu_lock(
                        lock_path,
                        cancellation=cancellation,
                        max_wait_seconds=20,
                        now=lambda: 0.0,
                    ):
                        self.fail("cancelled lock must not be entered")


class SummaryTests(unittest.TestCase):
    def test_initial_report_reads_trusted_pr_identity_environment(self):
        config = gate.Config(
            pathlib.Path("/source"), pathlib.Path("/artifacts"), "merge", "LingquLab/TileXR", 42
        )
        report = gate.initial_report(
            config,
            {"TILEXR_CI_HEAD_SHA": "head", "TILEXR_CI_BASE_SHA": "base"},
        )
        self.assertEqual(
            (report.pr_number, report.head_sha, report.base_sha, report.merge_sha),
            (42, "head", "base", "merge"),
        )

    def test_cases_and_summary_include_identity_outcomes_and_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            artifacts = pathlib.Path(directory)
            (artifacts / "cases.tsv").write_text(
                "passing case\tPASS\t0\t1.25\n"
                "failing case\tFAIL\t7\t2.5\n"
                "malformed\trow\n",
                encoding="utf-8",
            )
            step_summary = io.StringIO()
            report = gate.GateReport(
                pr_number=42, head_sha="head", base_sha="base", merge_sha="merge",
                build_outcome="passed", queue_outcome="passed", hardware_outcome="failed",
                wait_seconds=65.0, execution_seconds=3.75, failure_class="code-or-test-failure",
            )

            text = gate.write_summary(report, artifacts, step_summary=step_summary)

            self.assertEqual((artifacts / "summary.md").read_text(encoding="utf-8"), text)
            self.assertEqual(step_summary.getvalue(), text + "\n")
            for expected in (
                "PR #42", "head", "base", "merge", "Build", "Queue", "Hardware",
                "65.0", "3.75", "Passed cases", "1", "Failed cases", "failing case",
                "code-or-test-failure", "Malformed cases.tsv rows", "1",
                "UDMA data plane: out of scope on 910B3",
                "Multi-host validation: out of scope for this gate",
            ):
                self.assertIn(expected, text)

    def test_summary_keeps_scope_and_failure_details_without_cases_file(self):
        with tempfile.TemporaryDirectory() as directory:
            report = gate.GateReport(
                pr_number=7, merge_sha="merge", failure_class="runner-or-toolchain-failure",
                failure_detail="npu-smi unhealthy",
            )
            text = gate.write_summary(report, pathlib.Path(directory))
            self.assertIn("runner-or-toolchain-failure", text)
            self.assertIn("npu-smi unhealthy", text)
            self.assertIn("UDMA data plane: out of scope on 910B3", text)
            self.assertIn("Multi-host validation: out of scope for this gate", text)


class CaseEvidenceTests(unittest.TestCase):
    def test_success_evidence_requires_cases_file(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(gate.InfrastructureFailure):
                gate.validate_success_cases(pathlib.Path(directory) / "cases.tsv")

    def test_rejects_malformed_nonfinite_contradictory_and_duplicate_rows(self):
        invalid_contents = {
            "malformed": "only\ttwo\n",
            "nan": "case\tPASS\t0\tnan\n",
            "negative duration": "case\tPASS\t0\t-1\n",
            "negative exit": "case\tFAIL\t-1\t1\n",
            "unsupported": "case\tSKIP\t0\t1\n",
            "pass nonzero": "case\tPASS\t1\t1\n",
            "fail zero": "case\tFAIL\t0\t1\n",
            "duplicate": "case\tPASS\t0\t1\ncase\tPASS\t0\t2\n",
            "empty name": "\tPASS\t0\t1\n",
        }
        for label, contents in invalid_contents.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "cases.tsv"
                path.write_text(contents, encoding="utf-8")
                with self.assertRaises(gate.InfrastructureFailure):
                    gate.validate_success_cases(path)

    def test_valid_passing_manifest_is_authoritative_success_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "cases.tsv"
            path.write_text(
                "build checks\tPASS\t0\t1.25\nhardware checks\tPASS\t0\t2\n",
                encoding="utf-8",
            )
            cases = gate.validate_success_cases(path)
        self.assertEqual([case[0] for case in cases], ["build checks", "hardware checks"])


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

    def test_collector_failure_is_not_overwritten_by_late_cancellation(self):
        collector = gate._collector_failure("collector timed out")
        selected = gate.select_final_failure(collector, gate.Cancelled("late signal"))
        self.assertIs(selected, collector)
        self.assertEqual(gate.exit_code_for(selected), 23)

    def test_outer_cleanup_runs_after_lock_failure_until_cleanup_is_checked(self):
        report = gate.GateReport(pr_number=1)
        self.assertTrue(gate.requires_outer_cleanup(report))
        report.cleanup_checked = True
        self.assertFalse(gate.requires_outer_cleanup(report))

    def test_lock_failure_outer_cleanup_precedes_collection_and_overrides_timeout(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            config = gate.Config(
                root / "source", root / "artifacts", "merge", "LingquLab/TileXR", 42
            )

            def cleanup():
                events.append("cleanup")
                raise gate.InfrastructureFailure("cleanup could not be verified")

            with mock.patch.object(
                gate,
                "orchestrate",
                side_effect=npu_state.ResourceTimeout("lock timed out"),
            ), mock.patch.object(gate, "verify_final_cleanup", side_effect=cleanup), \
                 mock.patch.object(
                     gate, "write_summary", side_effect=lambda *args, **kwargs: events.append("summary")
                 ), mock.patch.object(
                     gate, "invoke_collector", side_effect=lambda *args, **kwargs: events.append("collector")
                 ):
                result = gate._run_controller_body(
                    config,
                    {"TILEXR_CI_GITHUB_TOKEN": "token"},
                    root / "trusted",
                    gate.CancellationState(),
                )

        self.assertEqual(result, 23)
        self.assertLess(events.index("cleanup"), events.index("collector"))


class CliValidationTests(unittest.TestCase):
    def test_cancelled_collector_attempt_is_capped_at_thirty_seconds(self):
        config = gate.Config(
            pathlib.Path("/source"), pathlib.Path("/artifacts"), "sha", "LingquLab/TileXR", 1
        )
        cancellation = gate.CancellationState()
        cancellation.cancel()
        clock = [0.0]
        child = FakeProcess(polls=(), returncode=None)
        terminate = mock.Mock()

        def advance(seconds):
            clock[0] += seconds

        with tempfile.TemporaryDirectory() as directory:
            collector = pathlib.Path(directory) / "collect_artifacts.sh"
            collector.write_text("#!/bin/sh\n", encoding="utf-8")
            collector.chmod(0o700)
            with self.assertRaises(gate.InfrastructureFailure):
                gate.invoke_collector(
                    collector, config, {}, cancellation=cancellation,
                    popen=lambda *args, **kwargs: child, now=lambda: clock[0],
                    sleep=advance, terminate=terminate,
                )

        self.assertEqual(clock[0], 30.0)
        terminate.assert_called_once_with(child)

    def test_over_deadline_collector_completion_cannot_succeed(self):
        config = gate.Config(
            pathlib.Path("/source"), pathlib.Path("/artifacts"), "sha", "LingquLab/TileXR", 1
        )
        cancellation = gate.CancellationState()
        cancellation.cancel()
        child = FakeProcess(polls=(0,), returncode=0)
        clock = iter((0.0, 30.0))
        with tempfile.TemporaryDirectory() as directory:
            collector = pathlib.Path(directory) / "collect_artifacts.sh"
            collector.write_text("#!/bin/sh\n", encoding="utf-8")
            collector.chmod(0o700)
            with self.assertRaises(gate.InfrastructureFailure):
                gate.invoke_collector(
                    collector, config, {}, cancellation=cancellation,
                    popen=lambda *args, **kwargs: child, now=lambda: next(clock),
                    sleep=lambda seconds: None, terminate=lambda process: None,
                )
        self.assertEqual(child._polls, [0])

    def test_collector_termination_failure_preserves_collector_primary(self):
        config = gate.Config(
            pathlib.Path("/source"), pathlib.Path("/artifacts"), "sha", "LingquLab/TileXR", 1
        )
        child = FakeProcess(polls=(9,), returncode=9)
        with tempfile.TemporaryDirectory() as directory:
            collector = pathlib.Path(directory) / "collect_artifacts.sh"
            collector.write_text("#!/bin/sh\n", encoding="utf-8")
            collector.chmod(0o700)
            with self.assertRaises(gate.InfrastructureFailure) as raised:
                gate.invoke_collector(
                    collector,
                    config,
                    {},
                    popen=lambda *args, **kwargs: child,
                    terminate=lambda process: (_ for _ in ()).throw(OSError("cannot reap collector")),
                )

        self.assertTrue(getattr(raised.exception, "collector_failure", False))
        self.assertIn("status 9", str(raised.exception))
        self.assertIn("cannot reap collector", str(raised.exception))

    def test_preexisting_cancellation_does_not_skip_bounded_collector_attempt(self):
        config = gate.Config(
            pathlib.Path("/source"), pathlib.Path("/artifacts"), "sha", "LingquLab/TileXR", 1
        )
        cancellation = gate.CancellationState()
        cancellation.cancel()
        child = FakeProcess(polls=(0,), returncode=0)
        popen = mock.Mock(return_value=child)
        terminate = mock.Mock()
        with tempfile.TemporaryDirectory() as directory:
            collector = pathlib.Path(directory) / "collect_artifacts.sh"
            collector.write_text("#!/bin/sh\n", encoding="utf-8")
            collector.chmod(0o700)

            gate.invoke_collector(
                collector,
                config,
                {},
                cancellation=cancellation,
                popen=popen,
                now=lambda: 0.0,
                sleep=lambda seconds: None,
                terminate=terminate,
            )

        popen.assert_called_once()
        terminate.assert_called_once_with(child)

    def test_missing_collector_overrides_primary_code_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            config = gate.Config(
                root / "source", root / "artifacts", "merge", "LingquLab/TileXR", 42
            )
            with mock.patch.object(
                gate, "orchestrate", side_effect=gate.CodeFailure("tests failed")
            ), mock.patch.object(gate, "verify_final_cleanup"), mock.patch.object(
                gate, "write_summary", return_value="summary"
            ):
                result = gate._run_controller_body(
                    config,
                    {"TILEXR_CI_GITHUB_TOKEN": "token"},
                    root / "trusted-without-collector",
                    gate.CancellationState(),
                )
        self.assertEqual(result, 23)

    def test_repository_and_pr_number_are_strictly_validated(self):
        with self.assertRaises(gate.InfrastructureFailure):
            gate.validate_config(
                gate.Config(pathlib.Path("/s"), pathlib.Path("/a"), "sha", "other/repo", 1)
            )
        with self.assertRaises(gate.InfrastructureFailure):
            gate.validate_config(
                gate.Config(pathlib.Path("/s"), pathlib.Path("/a"), "sha", "LingquLab/TileXR", 0)
            )

    def test_missing_or_unexecutable_collector_is_infrastructure_failure(self):
        config = gate.Config(
            pathlib.Path("/source"), pathlib.Path("/artifacts"), "sha", "LingquLab/TileXR", 1
        )
        with self.assertRaises(gate.InfrastructureFailure):
            gate.invoke_collector(pathlib.Path("/does/not/exist"), config, {})

    def test_collector_cancellation_during_launch_still_collects_and_reaps(self):
        config = gate.Config(
            pathlib.Path("/source"), pathlib.Path("/artifacts"), "sha", "LingquLab/TileXR", 1
        )
        cancellation = gate.CancellationState()
        child = FakeProcess(polls=(0,), returncode=0)
        terminate = mock.Mock()
        popen = mock.Mock(side_effect=lambda *args, **kwargs: cancellation.cancel() or child)
        with tempfile.TemporaryDirectory() as directory:
            collector = pathlib.Path(directory) / "collect_artifacts.sh"
            collector.write_text("#!/bin/sh\n", encoding="utf-8")
            collector.chmod(0o700)
            gate.invoke_collector(
                collector,
                config,
                {"PATH": "/usr/bin"},
                cancellation=cancellation,
                popen=popen,
                now=lambda: 0.0,
                sleep=lambda seconds: None,
                terminate=terminate,
            )
        popen.assert_called_once_with(
            [str(collector), "/source", "/artifacts"],
            cwd="/source",
            env={"PATH": "/usr/bin"},
            start_new_session=True,
        )
        terminate.assert_called_once_with(child)


if __name__ == "__main__":
    unittest.main()
