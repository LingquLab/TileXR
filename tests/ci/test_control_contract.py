#!/usr/bin/env python3

import pathlib
import shutil
import stat
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTROL = ROOT / "scripts" / "ci" / "control"


class ControlSourceContractTests(unittest.TestCase):
    def read(self, relative_path):
        return (ROOT / relative_path).read_text(encoding="utf-8")

    def test_control_shell_entrypoints_are_executable(self):
        for name in [
            "build_blue.sh",
            "run_hardware.sh",
            "collect_artifacts.sh",
            "job_completed.sh",
        ]:
            mode = (CONTROL / name).stat().st_mode
            self.assertTrue(mode & stat.S_IXUSR, name)

    def test_build_manifest_enables_and_runs_all_non_hardware_coverage(self):
        text = self.read("scripts/ci/control/build_blue.sh")
        for token in [
            "/home/tilexr-ci/toolchains/cann/9.1.0",
            "-DTILEXR_BUILD_COLLECTIVES=ON",
            "-DTILEXR_BUILD_EP=ON",
            "-DTILEXR_BUILD_CHECKER=ON",
            "-DTILEXR_BUILD_TESTS=ON",
            "-DBUILD_TESTING=ON",
            "ctest --test-dir",
            "test_tilexr_log",
            "test_tilexr_log_spdlog_compile",
            "test_tilexr_source_guards",
            "test_tilexr_udma_transport_layout",
            "test_tilexr_udma_registry",
            "test_tilexr_udma_demo_sources",
            "test_tilexr_udma_source_guard",
            "test_tilexr_sdma_metadata",
            "test_tilexr_sdma_api_invalid",
            "test_tilexr_sdma_transport_disabled",
            "test_tilexr_sdma_comm_wiring",
            "test_tilexr_sdma_source_guard",
            "test_tilexr_sdma_header_compile",
            "test_tilexr_memory_demo_sources",
            "test_tilexr_ep_layout",
            "test_tilexr_ep_api_sources",
            "test_tilexr_ep_kernel_sources",
            "test_tilexr_ep_host_validation",
            "libascend_hal.so",
            "readelf",
            "RPATH|RUNPATH",
            "devlib",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("run_case sdma-disabled-comm", text)

    def test_manifests_never_source_pull_request_code_into_trusted_shell(self):
        for relative in [
            "scripts/ci/control/build_blue.sh",
            "scripts/ci/control/run_hardware.sh",
        ]:
            text = self.read(relative)
            self.assertNotIn('source "${SOURCE_DIR}/', text)
            sealed_source = (
                "source /home/tilexr-ci/toolchains/cann/9.1.0/cann/set_env.sh"
            )
            self.assertIn(sealed_source, text)
            self.assertLess(text.index(sealed_source), text.index("run_case()"))
            self.assertIn("trusted-environment", text)

    def test_manifests_prioritize_freshly_built_libraries(self):
        for relative in [
            "scripts/ci/control/build_blue.sh",
            "scripts/ci/control/run_hardware.sh",
        ]:
            text = self.read(relative)
            assignment = (
                'SOURCE_LIBRARY_PATH="${SOURCE_DIR}/install/lib64:'
                '${SOURCE_DIR}/install/lib"'
            )
            self.assertIn(assignment, text)
            export_line = next(
                line for line in text.splitlines()
                if line.startswith("export LD_LIBRARY_PATH=")
            )
            self.assertLess(
                export_line.index("${SOURCE_LIBRARY_PATH}"),
                export_line.index("${ASCEND_DRIVER_PATH}"),
            )
            self.assertIn("for test_install in comm udma sdma ep memory", text)
            self.assertIn('${SOURCE_DIR}/tests/${test_install}/install/lib64', text)
            self.assertIn('${SOURCE_DIR}/tests/${test_install}/install/lib', text)

    def test_build_manifest_requires_every_hardware_binary(self):
        text = self.read("scripts/ci/control/build_blue.sh")
        for token in [
            "test_tilexr_udma",
            "tilexr_memory_demo",
            "tilexr_ep_dispatch_demo",
            "tilexr_sdma_demo",
            "test_tilexr_collectives_correctness",
            "tilexr_collective_perf",
        ]:
            self.assertIn(token, text)
        self.assertIn("test -x", text)

    def test_hardware_manifest_has_required_8_card_coverage(self):
        text = self.read("scripts/ci/control/run_hardware.sh")
        for token in [
            "TILEXR_TEST_DEVICES=0,1,2,3,4,5,6,7",
            "TILEXR_AVAILABLE_NPUS=8",
            "TILEXR_SKIP_IF_INSUFFICIENT_NPUS=0",
            "TILEXR_COLLECTIVES_RUN_TIMEOUT_SEC=600",
            "test_tilexr_udma",
            "run_tilexr_memory_demo.sh",
            "run_tilexr_ep_dispatch_demo.sh",
            "allgather alltoall allreduce reducescatter",
            "broadcast --root 0",
            "broadcast --root 7",
            "64 4096 1048576",
            "--check 1",
            "test_tilexr_sdma_disabled_comm",
            "npu-smi info -t health -i",
            "/usr/bin/timeout --signal=TERM --kill-after=2 10",
            "Health",
            "OK",
        ]:
            self.assertIn(token, text)

    def test_every_multirank_launch_has_external_ten_minute_timeout(self):
        text = self.read("scripts/ci/control/run_hardware.sh")
        timeout = "/usr/bin/timeout --signal=TERM --kill-after=10 600"
        # UDMA MPI, memory, EP, correctness loop, two broadcasts, and perf loop.
        self.assertGreaterEqual(text.count(timeout), 7)
        for marker in [
            "mpirun -n 8",
            "run_tilexr_memory_demo.sh",
            "run_tilexr_ep_dispatch_demo.sh",
            "run_collectives_correctness.sh",
            "run_collective_perf.sh",
        ]:
            position = text.find(marker)
            self.assertNotEqual(-1, position)
            prefix = text[max(0, position - 500) : position]
            self.assertIn(timeout, prefix)

    def test_hardware_manifest_records_stable_cases_and_scope(self):
        text = self.read("scripts/ci/control/run_hardware.sh")
        for token in [
            "run_case()",
            "cases.tsv",
            "udma-single-rank-fallback",
            "udma-eight-rank-fallback",
            "memory-eight-rank",
            "ep-eight-rank",
            "collectives-correctness-${op}",
            "collectives-correctness-broadcast-root-0",
            "collectives-correctness-broadcast-root-7",
            "collectives-perf-${op}",
            "sdma-device-${device}",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("run_tilexr_udma_demo.sh", text)
        self.assertIn("UDMA data plane: out of scope on 910B3", text)
        self.assertIn("Multi-host validation: out of scope for this gate", text)
        self.assertIn("vLLM model inference: out of scope for this gate", text)
        self.assertIn("Performance regression thresholds: out of scope for this gate", text)

    def test_build_manifest_resets_cases_and_records_each_safe_test(self):
        text = self.read("scripts/ci/control/build_blue.sh")
        self.assertIn(': > "${CASES_FILE}"', text)
        self.assertIn("run_case()", text)
        self.assertGreaterEqual(text.count("run_case "), 19)
        self.assertIn("printf '%s\\t%s\\t%s\\t%s\\n'", text)

    def test_build_manifest_captures_configuration_and_build_logs(self):
        text = self.read("scripts/ci/control/build_blue.sh")
        self.assertIn("run_logged_step()", text)
        for name in [
            "top-level-configure",
            "top-level-build",
            "comm-build",
            "udma-build",
            "sdma-build",
            "ep-build",
            "memory-configure",
            "memory-build",
        ]:
            self.assertIn('run_logged_step "{}"'.format(name), text)

    def test_job_hook_uses_fixed_workspace_and_absolute_tools(self):
        text = self.read("scripts/ci/control/job_completed.sh")
        self.assertIn("/home/tilexr-ci/actions-runner/_work/TileXR/TileXR", text)
        self.assertIn("/home/tilexr-ci/actions-runner/_work", text)
        self.assertIn("/usr/bin/find", text)
        self.assertIn("/usr/bin/rm", text)
        self.assertNotIn("TILEXR_CI_WORKSPACE", text)

    def test_collector_records_environment_and_tool_versions(self):
        text = self.read("scripts/ci/control/collect_artifacts.sh")
        for token in [
            "environment.txt",
            "TILEXR_CANN_HOME",
            "cmake --version",
            "c++ --version",
            "bisheng -v",
            "mpirun --version",
            "npu-smi info",
            "git -C",
        ]:
            self.assertIn(token, text)


class ArtifactCollectorBehaviorTests(unittest.TestCase):
    def run_collector(self, source, artifacts):
        return subprocess.run(
            [str(CONTROL / "collect_artifacts.sh"), str(source), str(artifacts)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_collector_preserves_paths_and_rejects_unsafe_payloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            artifacts = source / ".artifacts"
            external = root / "external"
            (source / "logs" / "one").mkdir(parents=True)
            (source / "logs" / "two").mkdir(parents=True)
            (source / "build-ci").mkdir()
            (source / "install" / "lib").mkdir(parents=True)
            external.mkdir()

            (source / "logs" / "one" / "same.log").write_text("one\n")
            (source / "logs" / "two" / "same.log").write_text("two\n")
            (source / "results.xml").write_text("<testsuite/>\n")
            (source / "summary.csv").write_text("name,value\na,1\n")
            (source / "trace.json").write_text("{}\n")
            (source / "ldd-tile-comm.txt").write_text("libascend_hal.so\n")
            (source / "build-ci" / "payload.o").write_bytes(b"object")
            (source / "install" / "lib" / "libbad.so").write_bytes(b"library")
            (source / "model.om").write_bytes(b"model")
            (source / "cann.run").write_bytes(b"package")
            (source / "notes.txt").write_text("not an artifact\n")
            (external / "outside.log").write_text("outside\n")
            (source / "outside.log").symlink_to(external / "outside.log")
            (source / "linked-dir").symlink_to(external, target_is_directory=True)

            artifacts.mkdir()
            (artifacts / "ctest-top-level.xml").write_text("<testsuite/>\n")
            result = self.run_collector(source, artifacts)
            self.assertEqual(0, result.returncode, result.stderr)

            collected = artifacts / "source"
            self.assertEqual("one\n", (collected / "logs/one/same.log").read_text())
            self.assertEqual("two\n", (collected / "logs/two/same.log").read_text())
            for relative in [
                "results.xml",
                "summary.csv",
                "trace.json",
                "ldd-tile-comm.txt",
            ]:
                self.assertTrue((collected / relative).is_file(), relative)
            for relative in [
                "build-ci/payload.o",
                "install/lib/libbad.so",
                "model.om",
                "cann.run",
                "notes.txt",
                "outside.log",
                "linked-dir/outside.log",
            ]:
                self.assertFalse((collected / relative).exists(), relative)
            self.assertFalse((artifacts / "source/.artifacts").exists())
            self.assertTrue((artifacts / "environment.txt").is_file())

            manifest = (artifacts / "manifest.txt").read_text().splitlines()
            self.assertEqual(sorted(manifest), manifest)
            self.assertTrue(any(line.startswith("source/logs/one/same.log\t") for line in manifest))
            self.assertTrue(any(line.startswith("source/logs/two/same.log\t") for line in manifest))
            for line in manifest:
                relative, size = line.split("\t")
                self.assertEqual((artifacts / relative).stat().st_size, int(size))
            for path in artifacts.rglob("*"):
                self.assertFalse(path.is_symlink(), str(path))

    def test_collector_refuses_files_larger_than_100_mib(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            artifacts = root / "artifacts"
            source.mkdir()
            huge = source / "huge.log"
            with huge.open("wb") as output:
                output.truncate(100 * 1024 * 1024 + 1)

            result = self.run_collector(source, artifacts)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertFalse((artifacts / "source/huge.log").exists())
            rejected = (artifacts / "artifact-rejections.txt").read_text()
            self.assertIn("source/huge.log", rejected)

    def test_collector_sanitizes_preexisting_upload_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            artifacts = root / "artifacts"
            external = root / "external.log"
            source.mkdir()
            artifacts.mkdir()
            external.write_text("external\n")
            (artifacts / "keep.xml").write_text("<testsuite/>\n")
            (artifacts / "payload.o").write_bytes(b"object")
            (artifacts / "library.so").write_bytes(b"library")
            (artifacts / "outside.log").symlink_to(external)
            with (artifacts / "huge.log").open("wb") as output:
                output.truncate(100 * 1024 * 1024 + 1)

            result = self.run_collector(source, artifacts)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue((artifacts / "keep.xml").is_file())
            self.assertFalse((artifacts / "payload.o").exists())
            self.assertFalse((artifacts / "library.so").exists())
            self.assertFalse((artifacts / "outside.log").exists())
            self.assertFalse((artifacts / "huge.log").exists())
            self.assertEqual("external\n", external.read_text())
            rejected = (artifacts / "artifact-rejections.txt").read_text()
            for token in ["payload.o", "library.so", "outside.log", "huge.log"]:
                self.assertIn(token, rejected)

    def test_collector_rejects_symlink_source_and_artifact_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source_real = root / "source-real"
            artifact_real = root / "artifact-real"
            source_real.mkdir()
            artifact_real.mkdir()
            source_link = root / "source-link"
            artifact_link = root / "artifact-link"
            source_link.symlink_to(source_real, target_is_directory=True)
            artifact_link.symlink_to(artifact_real, target_is_directory=True)

            source_result = self.run_collector(source_link, root / "artifacts")
            self.assertNotEqual(0, source_result.returncode)
            artifact_result = self.run_collector(source_real, artifact_link)
            self.assertNotEqual(0, artifact_result.returncode)


class JobHookBehaviorTests(unittest.TestCase):
    def materialize_hook(self, root, runner_work_root):
        source = (CONTROL / "job_completed.sh").read_text(encoding="utf-8")
        source = source.replace(
            "/home/tilexr-ci/actions-runner/_work", str(runner_work_root)
        )
        if not pathlib.Path("/usr/bin/rm").exists():
            source = source.replace("/usr/bin/rm", shutil.which("rm") or "/bin/rm")
        hook = root / "job_completed-test.sh"
        hook.write_text(source, encoding="utf-8")
        hook.chmod(hook.stat().st_mode | stat.S_IXUSR)
        return hook

    def test_job_hook_cleans_children_without_following_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runner_root = root / "runner" / "_work"
            workspace = runner_root / "TileXR" / "TileXR"
            external = root / "external"
            artifacts = root / "artifacts"
            (workspace / "nested").mkdir(parents=True)
            external.mkdir()
            artifacts.mkdir()
            (workspace / "nested" / "file").write_text("delete")
            (workspace / "outside-link").symlink_to(external, target_is_directory=True)
            (external / "keep").write_text("keep")
            (artifacts / "keep.xml").write_text("keep")
            hook = self.materialize_hook(root, runner_root)

            result = subprocess.run(
                [str(hook)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(workspace.is_dir())
            self.assertEqual([], list(workspace.iterdir()))
            self.assertEqual("keep", (external / "keep").read_text())
            self.assertEqual("keep", (artifacts / "keep.xml").read_text())

    def test_job_hook_refuses_a_symlink_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runner_root = root / "runner" / "_work"
            workspace_parent = runner_root / "TileXR"
            target = root / "target"
            workspace_parent.mkdir(parents=True)
            target.mkdir()
            (target / "keep").write_text("keep")
            (workspace_parent / "TileXR").symlink_to(target, target_is_directory=True)
            hook = self.materialize_hook(root, runner_root)

            result = subprocess.run(
                [str(hook)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertNotEqual(0, result.returncode)
            self.assertEqual("keep", (target / "keep").read_text())

    def test_job_hook_refuses_a_dangling_symlink_workspace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runner_root = root / "runner" / "_work"
            workspace_parent = runner_root / "TileXR"
            workspace_parent.mkdir(parents=True)
            (workspace_parent / "TileXR").symlink_to(root / "missing")
            hook = self.materialize_hook(root, runner_root)

            result = subprocess.run(
                [str(hook)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertNotEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
