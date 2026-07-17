#!/usr/bin/env python3
import pathlib
import pwd
import sys
import unittest
from unittest import mock


CONTROL_DIR = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "ci" / "control"
sys.path.insert(0, str(CONTROL_DIR))

import npu_state


FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def fixture(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def completed(command, stdout, returncode=0, stderr=""):
    return npu_state.subprocess.CompletedProcess(command, returncode, stdout, stderr)


def healthy_responses(process_output):
    commands = [["npu-smi", "info"]] + [
        ["npu-smi", "info", "-t", "health", "-i", str(device)]
        for device in range(8)
    ]
    health = fixture("npu_smi_health_ok.txt")
    return [completed(commands[0], process_output)] + [
        completed(command, health.replace(": 0", ": %d" % device, 1))
        for device, command in enumerate(commands[1:])
    ]


class ParseNpuStateTests(unittest.TestCase):
    def test_busy_process_table_returns_devices_pids_and_names(self):
        processes = npu_state.parse_process_table(fixture("npu_smi_busy.txt"))

        self.assertEqual([(process.device, process.pid) for process in processes],
                         [(0, 3558608), (7, 3558615)])
        self.assertEqual([process.name for process in processes], ["python3.11", "python3.11"])

    def test_idle_process_table_returns_no_processes(self):
        self.assertEqual(npu_state.parse_process_table(fixture("npu_smi_idle.txt")), [])

    def test_health_returns_the_first_health_field(self):
        self.assertEqual(npu_state.parse_health(fixture("npu_smi_health_ok.txt")), "OK")


class WaitForIdleTests(unittest.TestCase):
    def test_waiter_resets_stability_until_two_consecutive_idle_samples(self):
        busy = npu_state.Snapshot(
            healthy=True,
            processes=(npu_state.NpuProcess(0, 3558608, "python3.11", "alice"),),
        )
        unhealthy = npu_state.Snapshot(healthy=False, processes=())
        idle = npu_state.Snapshot(healthy=True, processes=())
        snapshots = iter((idle, busy, unhealthy, idle, idle))
        clock = iter((0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0))
        emitted = []
        sleeps = []

        result = npu_state.wait_for_idle(
            read_snapshot=lambda: next(snapshots),
            sleep=sleeps.append,
            now=lambda: next(clock),
            emit=emitted.append,
            max_wait_seconds=10,
            poll_seconds=1,
            stable_samples=2,
        )

        self.assertIs(result, idle)
        self.assertEqual(sleeps, [1, 1, 1, 1])
        self.assertEqual(len(emitted), 5)
        self.assertIn("device=0 pid=3558608 owner=alice", emitted[1])
        self.assertIn("healthy=false", emitted[2])
        self.assertIn("stable=0/2", emitted[2])
        self.assertIn("stable=2/2", emitted[-1])

    def test_waiter_raises_resource_timeout_at_deadline(self):
        busy = npu_state.Snapshot(
            healthy=True,
            processes=(npu_state.NpuProcess(7, 3558615, "python3.11", "bob"),),
        )
        clock = iter((0.0, 0.0, 0.0, 1.0, 1.0, 2.0))
        emitted = []
        read_snapshot = mock.Mock(return_value=busy)

        with self.assertRaises(npu_state.ResourceTimeout):
            npu_state.wait_for_idle(
                read_snapshot=read_snapshot,
                sleep=lambda seconds: None,
                now=lambda: next(clock),
                emit=emitted.append,
                max_wait_seconds=2,
                poll_seconds=1,
                stable_samples=2,
            )

        self.assertEqual(read_snapshot.call_count, 2)
        self.assertEqual(len(emitted), 2)
        self.assertIn("remaining=1", emitted[-1])

    def test_waiter_rejects_idle_sample_that_crosses_the_deadline(self):
        clock = iter((0.0, 0.0, 2.0))
        idle = npu_state.Snapshot(healthy=True, processes=())

        with self.assertRaises(npu_state.ResourceTimeout):
            npu_state.wait_for_idle(
                read_snapshot=lambda: idle,
                sleep=lambda seconds: None,
                now=lambda: next(clock),
                emit=lambda line: None,
                max_wait_seconds=2,
                poll_seconds=1,
                stable_samples=1,
            )


class ReadSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.process_output = fixture("npu_smi_busy.txt")
        self.health_output = fixture("npu_smi_health_ok.txt")

    def test_read_snapshot_runs_expected_commands_and_resolves_owners(self):
        expected_commands = [["npu-smi", "info"]] + [
            ["npu-smi", "info", "-t", "health", "-i", str(device)]
            for device in range(8)
        ]
        responses = [completed(expected_commands[0], self.process_output)] + [
            completed(command, self.health_output.replace(": 0", ": %d" % device, 1))
            for device, command in enumerate(expected_commands[1:])
        ]
        stat_one = mock.Mock(st_uid=1001)
        stat_two = mock.Mock(st_uid=1002)

        with mock.patch.object(npu_state.subprocess, "run", side_effect=responses) as run, \
             mock.patch.object(npu_state.os, "stat", side_effect=[stat_one, stat_two]) as stat, \
             mock.patch.object(
                 npu_state.pwd,
                 "getpwuid",
                 side_effect=[
                     pwd.struct_passwd(("alice", "x", 1001, 1001, "", "/home/alice", "/bin/bash")),
                     pwd.struct_passwd(("bob", "x", 1002, 1002, "", "/home/bob", "/bin/bash")),
                 ],
             ):
            snapshot = npu_state.read_snapshot()

        self.assertTrue(snapshot.healthy)
        self.assertEqual(
            snapshot.processes,
            (
                npu_state.NpuProcess(0, 3558608, "python3.11", "alice"),
                npu_state.NpuProcess(7, 3558615, "python3.11", "bob"),
            ),
        )
        self.assertEqual([call.args[0] for call in run.call_args_list], expected_commands)
        self.assertEqual([call.args[0] for call in stat.call_args_list],
                         ["/proc/3558608", "/proc/3558615"])
        self.assertTrue(all(
            call.kwargs["timeout"] == npu_state.NPU_SMI_TIMEOUT_SECONDS
            for call in run.call_args_list
        ))

    def test_read_snapshot_is_unhealthy_when_npu_smi_times_out(self):
        commands = [["npu-smi", "info"]] + [
            ["npu-smi", "info", "-t", "health", "-i", str(device)]
            for device in range(8)
        ]
        responses = healthy_responses(fixture("npu_smi_idle.txt"))

        def run(command, **kwargs):
            if command == commands[4]:
                raise npu_state.subprocess.TimeoutExpired(
                    command, npu_state.NPU_SMI_TIMEOUT_SECONDS
                )
            return responses[commands.index(command)]

        with mock.patch.object(npu_state.subprocess, "run", side_effect=run) as mocked_run:
            snapshot = npu_state.read_snapshot()

        self.assertFalse(snapshot.healthy)
        self.assertEqual(len(mocked_run.call_args_list), 9)
        self.assertTrue(all(
            call.kwargs["timeout"] == npu_state.NPU_SMI_TIMEOUT_SECONDS
            for call in mocked_run.call_args_list
        ))

    def test_read_snapshot_caps_commands_to_absolute_deadline(self):
        clock = [100.0]

        def run(command, **kwargs):
            self.assertEqual(kwargs["timeout"], 5.0)
            clock[0] = 105.0
            raise npu_state.subprocess.TimeoutExpired(command, kwargs["timeout"])

        with mock.patch.object(npu_state.subprocess, "run", side_effect=run) as mocked_run:
            state = npu_state.read_snapshot(deadline=105.0, now=lambda: clock[0])

        self.assertFalse(state.healthy)
        self.assertEqual(mocked_run.call_count, 1)

    def test_read_snapshot_is_unhealthy_when_a_command_fails(self):
        commands = [["npu-smi", "info"]] + [
            ["npu-smi", "info", "-t", "health", "-i", str(device)]
            for device in range(8)
        ]
        responses = [completed(commands[0], self.process_output)] + [
            completed(command, self.health_output, returncode=(1 if device == 3 else 0))
            for device, command in enumerate(commands[1:])
        ]

        with mock.patch.object(npu_state.subprocess, "run", side_effect=responses), \
             mock.patch.object(npu_state.os, "stat", side_effect=FileNotFoundError):
            snapshot = npu_state.read_snapshot()

        self.assertFalse(snapshot.healthy)
        self.assertEqual([(process.device, process.owner) for process in snapshot.processes],
                         [(0, "unknown"), (7, "unknown")])

    def test_read_snapshot_is_unhealthy_for_non_ok_or_missing_health(self):
        commands = [["npu-smi", "info"]] + [
            ["npu-smi", "info", "-t", "health", "-i", str(device)]
            for device in range(8)
        ]
        responses = [completed(commands[0], fixture("npu_smi_idle.txt"))]
        for device, command in enumerate(commands[1:]):
            output = self.health_output
            if device == 2:
                output = output.replace("Health                         : OK", "Health                         : Warning", 1)
            if device == 5:
                output = "NPU ID                         : 5\nChip Count                     : 1\n"
            responses.append(completed(command, output))

        with mock.patch.object(npu_state.subprocess, "run", side_effect=responses):
            snapshot = npu_state.read_snapshot()

        self.assertFalse(snapshot.healthy)
        self.assertEqual(snapshot.processes, ())

    def test_read_snapshot_is_unhealthy_without_process_table_header(self):
        process_output = "NPU status was truncated before the process table\n"

        with mock.patch.object(
            npu_state.subprocess, "run", side_effect=healthy_responses(process_output)
        ):
            snapshot = npu_state.read_snapshot()

        self.assertFalse(snapshot.healthy)

    def test_read_snapshot_is_unhealthy_for_header_only_process_table(self):
        process_output = (
            "| NPU     Chip              | Process id    | Process name             |\n"
        )

        with mock.patch.object(
            npu_state.subprocess, "run", side_effect=healthy_responses(process_output)
        ):
            snapshot = npu_state.read_snapshot()

        self.assertFalse(snapshot.healthy)

    def test_read_snapshot_is_unhealthy_for_incomplete_device_coverage(self):
        process_output = fixture("npu_smi_idle.txt").replace(
            "| No running processes found in NPU 7                                                        |\n",
            "",
        )

        with mock.patch.object(
            npu_state.subprocess, "run", side_effect=healthy_responses(process_output)
        ):
            snapshot = npu_state.read_snapshot()

        self.assertFalse(snapshot.healthy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
