#!/usr/bin/env python3

import pathlib
import shutil
import shlex
import stat
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
CONTROL = ROOT / "scripts" / "ci" / "control"


def extract_shell_function(text, name):
    marker = "{}() {{".format(name)
    start = text.index(marker)
    end = text.index("\n}\n", start) + len("\n}\n")
    return text[start:end]


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

    def test_build_manifest_validates_installed_headers_libraries_and_checker(self):
        text = self.read("scripts/ci/control/build_blue.sh")
        for token in [
            "tilexr_api.h",
            "tilexr_types.h",
            "comm_args.h",
            "tilexr_sync.h",
            "tilexr_udma.h",
            "tilexr_sdma.h",
            "tilexr_ep.h",
            "tilexr_collectives.h",
            "tilexr_collectives_perf.h",
            "libtile-comm.so",
            "libtilexr-ep.so",
            "libtilexr-collectives.so",
            "install/bin/tilexr_checker",
            "libtilexr-checker-core.a",
            "validate_dynamic_output",
            "ldd-${label}.txt",
            "readelf-${label}.txt",
            "top-level-reinstall",
        ]:
            self.assertIn(token, text)
        self.assertNotIn("build-ci/tools/checker/tilexr_checker", text)
        checker_cmake = self.read("tools/checker/CMakeLists.txt")
        for token in [
            "install(TARGETS",
            "tilexr_checker",
            "tilexr-checker-core",
            "${CMAKE_INSTALL_BINDIR}",
            "${CMAKE_INSTALL_LIBDIR}",
        ]:
            self.assertIn(token, checker_cmake)

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

    def test_hardware_manifest_captures_bounded_state_topology_and_cann_evidence(self):
        hardware = self.read("scripts/ci/control/run_hardware.sh")
        build = self.read("scripts/ci/control/build_blue.sh")
        for token in [
            "npu-state-before.txt",
            "npu-state-after.txt",
            "npu-topology.txt",
            "trap finalize_hardware EXIT",
            "/usr/bin/timeout --signal=TERM --kill-after=2 10",
        ]:
            self.assertIn(token, hardware)
        for token in [
            "version-cann.txt",
            "${ASCEND_HOME_PATH}/${ARCH}-linux/ascend_toolkit_install.info",
            "ascend_toolkit_install.info",
            "ascend_ops_install.info",
            "Ascend-cann-toolkit",
            "Ascend-cann-910b-ops",
            "version=9.1.0",
        ]:
            self.assertIn(token, build)
        self.assertNotIn("9\\.1([.]0)?", build)

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

    def test_collector_requires_authoritative_root_evidence(self):
        text = self.read("scripts/ci/control/collect_artifacts.sh")
        self.assertIn("check_root_evidence", text)
        self.assertIn('check_root_evidence "cases.tsv"', text)
        self.assertIn('check_root_evidence "summary.md"', text)
        self.assertIn("EVIDENCE_ERROR", text)

    def test_job_hook_requires_exact_canonical_fixed_paths(self):
        text = self.read("scripts/ci/control/job_completed.sh")
        self.assertTrue(text.startswith("#!/bin/bash\n"))
        self.assertIn(
            '[[ "${RUNNER_WORK_ROOT_REAL}" != "${RUNNER_WORK_ROOT}" ]]', text
        )
        self.assertIn(
            '[[ "${WORKSPACE_PARENT_REAL}" != "${WORKSPACE_PARENT}" ]]', text
        )
        self.assertIn('[[ "${WORKSPACE_REAL}" != "${WORKSPACE}" ]]', text)

    def test_host_checks_manifest_runs_complete_fast_host_coverage(self):
        text = self.read("scripts/ci/host_checks.sh")
        for token in [
            "set -euo pipefail",
            "TILEXR_CI_BUILD_ROOT",
            'git -C "${ROOT_DIR}" ls-files -z --',
            'bash -n "${tracked_shell_files[@]}"',
            'cmake -S "${ROOT_DIR}/tests/ci"',
            "ctest --output-on-failure -T Test --no-compress-output",
            "ctest-ci.xml",
            'cmake -S "${ROOT_DIR}/tests/comm"',
            "test_tilexr_log",
            "test_tilexr_log_spdlog_compile",
            "test_tilexr_source_guards",
            'cmake -S "${ROOT_DIR}/tests/ep"',
            "-DBUILD_TILEXR_EP_DEMO=OFF",
            "ctest-ep.xml",
            'cmake -S "${ROOT_DIR}/tests/data_as_flag"',
            "ctest-data-as-flag.xml",
            "test_vllm_collectives_patch.py",
            "test_vllm_collectives_integration_sources.py",
            "test_collective_profile_report.py",
            "run_case()",
            "cases.tsv",
            "trap finalize_host_checks EXIT",
            "GITHUB_STEP_SUMMARY",
            "TILEXR_CI_PR_NUMBER",
            "TILEXR_CI_HEAD_SHA",
            "TILEXR_CI_BASE_SHA",
            "TILEXR_CI_EXPECTED_MERGE_SHA",
            "NPU-only coverage is not run by Host Checks",
        ]:
            self.assertIn(token, text)
        for suite in [
            "shell-syntax",
            "ci-ctest",
            "comm-host",
            "ep-source-only",
            "data-as-flag",
            "collectives-vllm-patch",
            "collectives-vllm-integration-sources",
            "collectives-profile-report",
        ]:
            self.assertIn("run_case {} ".format(suite), text)
        self.assertGreaterEqual(text.count(':-local}'), 4)


class HardwareHelperBehaviorTests(unittest.TestCase):
    def hardware_function(self, name):
        text = (CONTROL / "run_hardware.sh").read_text(encoding="utf-8")
        return extract_shell_function(text, name)

    def test_required_no_skip_wrapper_rejects_skip_and_accepts_real_pass(self):
        hardware = (CONTROL / "run_hardware.sh").read_text(encoding="utf-8")
        self.assertIn("run_case sdma-disabled-comm run_required_no_skip", hardware)
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            trusted = root / "trusted"
            trusted.mkdir()
            skipped = root / "skipped.sh"
            skipped.write_text(
                "#!/bin/bash\necho 'Skip test_tilexr_sdma_disabled_comm: aclInit failed'\nexit 0\n"
            )
            passed = root / "passed.sh"
            passed.write_text("#!/bin/bash\necho passed\nexit 0\n")
            skipped.chmod(0o755)
            passed.chmod(0o755)
            function = self.hardware_function("run_required_no_skip")

            def invoke(command):
                harness = "set -euo pipefail\nTRUSTED_ENV_ROOT={}\n{}\n".format(
                    shlex.quote(str(trusted)), function
                )
                harness += "run_required_no_skip 'Skip test_tilexr_sdma_disabled_comm:' {}\n".format(
                    shlex.quote(str(command))
                )
                return subprocess.run(
                    ["bash", "-c", harness],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            self.assertNotEqual(0, invoke(skipped).returncode)
            self.assertEqual(0, invoke(passed).returncode)

    def test_health_parser_uses_first_exact_top_level_field(self):
        hardware = (CONTROL / "run_hardware.sh").read_text(encoding="utf-8")
        self.assertIn("| parse_top_level_health)", hardware)
        function = self.hardware_function("parse_top_level_health")

        def parse(output):
            harness = "{}\nprintf '%s' {} | parse_top_level_health\n".format(
                function, shlex.quote(output)
            )
            return subprocess.run(
                ["bash", "-c", harness],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        ok = "Health : OK\nMCU Health : Warning\n"
        misleading = "Health : Warning\nMCU Health : OK\n"
        self.assertEqual("OK", parse(ok).stdout.strip())
        self.assertEqual("Warning", parse(misleading).stdout.strip())

    def test_exit_evidence_failure_preserves_primary_and_fails_success(self):
        function = self.hardware_function("finalize_hardware")
        with tempfile.TemporaryDirectory() as temporary:
            prefix = "ARTIFACT_DIR={}\n{}\n".format(
                shlex.quote(temporary), function
            )
            prefix += "capture_npu_evidence() { return 9; }\n"
            failed = subprocess.run(["bash", "-c", prefix + "false; finalize_hardware"])
            passed = subprocess.run(["bash", "-c", prefix + "true; finalize_hardware"])
            self.assertEqual(1, failed.returncode)
            self.assertEqual(9, passed.returncode)

    def test_run_directory_rejects_file_and_symlink_without_running_command(self):
        hardware = (CONTROL / "run_hardware.sh").read_text(encoding="utf-8")
        prepare = extract_shell_function(hardware, "prepare_ci_run_root")
        run_in_dir = extract_shell_function(hardware, "run_in_dir")
        with tempfile.TemporaryDirectory() as temporary:
            source = pathlib.Path(temporary).resolve()
            marker = source / "command-ran"

            def invoke(setup, directory, after_prepare=""):
                harness = "set -euo pipefail\nSOURCE_DIR={}\n{}\n{}\n".format(
                    shlex.quote(str(source)), prepare, run_in_dir
                )
                harness += setup + "\n"
                harness += "prepare_ci_run_root\n"
                harness += after_prepare + "\n"
                harness += "run_in_dir {} /usr/bin/touch {}\n".format(
                    shlex.quote(str(directory)), shlex.quote(str(marker))
                )
                return subprocess.run(
                    ["bash", "-c", harness],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            ci_root = source / ".ci-run"
            symlink_setup = "mkdir -p {0}; ln -s {0} {1}".format(
                shlex.quote(str(source / "outside")), shlex.quote(str(ci_root))
            )
            self.assertNotEqual(0, invoke(symlink_setup, ci_root / "case").returncode)
            self.assertFalse(marker.exists())

            file_case = ci_root / "case-file"
            file_setup = "touch {}".format(shlex.quote(str(file_case)))
            self.assertNotEqual(
                0,
                invoke(
                    "mkdir -p {}".format(shlex.quote(str(ci_root))),
                    file_case,
                    file_setup,
                ).returncode,
            )
            self.assertFalse(marker.exists())

            linked_case = ci_root / "case-link"
            link_setup = "mkdir -p {0}; ln -s {0} {1}".format(
                shlex.quote(str(source / "outside-case")),
                shlex.quote(str(linked_case)),
            )
            self.assertNotEqual(
                0,
                invoke(
                    "mkdir -p {}".format(shlex.quote(str(ci_root))),
                    linked_case,
                    link_setup,
                ).returncode,
            )
            self.assertFalse(marker.exists())


class HostChecksBehaviorTests(unittest.TestCase):
    def test_run_case_stops_at_and_records_an_intermediate_failure(self):
        host_checks = (ROOT / "scripts/ci/host_checks.sh").read_text(
            encoding="utf-8"
        )
        function = extract_shell_function(host_checks, "run_case")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            cases = root / "cases.tsv"
            after_failure = root / "after-failure"
            harness = "set -euo pipefail\nCASES_FILE={}\n{}\n".format(
                shlex.quote(str(cases)), function
            )
            harness += "suite() {{ false; touch {}; }}\n".format(
                shlex.quote(str(after_failure))
            )
            harness += "run_case sample suite\n"
            result = subprocess.run(
                ["bash", "-c", harness],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertNotEqual(0, result.returncode)
            self.assertFalse(after_failure.exists())
            fields = cases.read_text(encoding="utf-8").strip().split("\t")
            self.assertEqual(["sample", "FAIL", "1"], fields[:3])


class BuildHelperBehaviorTests(unittest.TestCase):
    def test_cann_metadata_requires_exact_package_and_version(self):
        build = (CONTROL / "build_blue.sh").read_text(encoding="utf-8")
        function = extract_shell_function(build, "validate_cann_metadata_file")
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)

            def validate(contents, package):
                metadata = root / "metadata.info"
                metadata.write_text(contents)
                harness = "{}\nvalidate_cann_metadata_file {} {}\n".format(
                    function, shlex.quote(str(metadata)), shlex.quote(package)
                )
                return subprocess.run(
                    ["bash", "-c", harness],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            valid = "package_name=Ascend-cann-toolkit\nversion=9.1.0\n"
            wrong_version = "package_name=Ascend-cann-toolkit\nversion=9.10.0\n"
            wrong_package = "package_name=Ascend-cann-910b-ops\nversion=9.1.0\n"
            conflicting = (
                "package_name=Ascend-cann-toolkit\n"
                "version=9.1.0\nversion=9.10.0\n"
            )
            self.assertEqual(0, validate(valid, "Ascend-cann-toolkit").returncode)
            self.assertNotEqual(
                0, validate(wrong_version, "Ascend-cann-toolkit").returncode
            )
            self.assertNotEqual(
                0, validate(wrong_package, "Ascend-cann-toolkit").returncode
            )
            self.assertNotEqual(
                0, validate(conflicting, "Ascend-cann-toolkit").returncode
            )


class ArtifactCollectorBehaviorTests(unittest.TestCase):
    def seed_evidence(self, artifacts):
        artifacts.mkdir(parents=True, exist_ok=True)
        (artifacts / "cases.tsv").write_text("case\tpass\t0\t1\n")
        (artifacts / "summary.md").write_text("# Summary\n")

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

            self.seed_evidence(artifacts)
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
            self.assertEqual("case\tpass\t0\t1\n", (artifacts / "cases.tsv").read_text())
            self.assertEqual("# Summary\n", (artifacts / "summary.md").read_text())

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
            self.seed_evidence(artifacts)
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
            self.seed_evidence(artifacts)
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

    def test_collector_fails_closed_on_missing_or_unsafe_root_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            source = root / "source"
            source.mkdir()

            cases_missing = root / "cases-missing"
            cases_missing.mkdir()
            (cases_missing / "summary.md").write_text("# Summary\n")
            cases_result = self.run_collector(source, cases_missing)
            self.assertNotEqual(0, cases_result.returncode)
            self.assertTrue((cases_missing / "environment.txt").is_file())
            self.assertTrue((cases_missing / "manifest.txt").is_file())

            summary_missing = root / "summary-missing"
            summary_missing.mkdir()
            (summary_missing / "cases.tsv").write_text("case\tpass\t0\t1\n")
            summary_result = self.run_collector(source, summary_missing)
            self.assertNotEqual(0, summary_result.returncode)
            self.assertTrue((summary_missing / "environment.txt").is_file())
            self.assertTrue((summary_missing / "manifest.txt").is_file())

            symlinked = root / "symlinked"
            symlinked.mkdir()
            real_cases = root / "real-cases.tsv"
            real_cases.write_text("case\tpass\t0\t1\n")
            (symlinked / "cases.tsv").symlink_to(real_cases)
            (symlinked / "summary.md").write_text("# Summary\n")
            symlink_result = self.run_collector(source, symlinked)
            self.assertNotEqual(0, symlink_result.returncode)
            self.assertFalse((symlinked / "cases.tsv").exists())
            self.assertTrue((symlinked / "environment.txt").is_file())
            self.assertTrue((symlinked / "manifest.txt").is_file())

            oversized = root / "oversized"
            oversized.mkdir()
            (oversized / "cases.tsv").write_text("case\tpass\t0\t1\n")
            with (oversized / "summary.md").open("wb") as output:
                output.truncate(100 * 1024 * 1024 + 1)
            oversized_result = self.run_collector(source, oversized)
            self.assertNotEqual(0, oversized_result.returncode)
            self.assertFalse((oversized / "summary.md").exists())
            self.assertTrue((oversized / "environment.txt").is_file())
            self.assertTrue((oversized / "manifest.txt").is_file())

            unsafe_directory = root / "unsafe-directory"
            (unsafe_directory / "summary.md").mkdir(parents=True)
            (unsafe_directory / "summary.md" / "payload.so").write_bytes(b"library")
            (unsafe_directory / "cases.tsv").write_text("case\tpass\t0\t1\n")
            directory_result = self.run_collector(source, unsafe_directory)
            self.assertNotEqual(0, directory_result.returncode)
            self.assertFalse((unsafe_directory / "summary.md").exists())
            self.assertTrue((unsafe_directory / "environment.txt").is_file())
            self.assertTrue((unsafe_directory / "manifest.txt").is_file())

            for upload in [
                cases_missing,
                summary_missing,
                symlinked,
                oversized,
                unsafe_directory,
            ]:
                for path in upload.rglob("*"):
                    self.assertFalse(path.is_symlink(), str(path))
                    if path.is_file():
                        self.assertLessEqual(path.stat().st_size, 100 * 1024 * 1024)

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
        runner_work_root = runner_work_root.resolve()
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

    def test_job_hook_refuses_an_intermediate_symlink_inside_work_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runner_root = root / "runner" / "_work"
            redirected = runner_root / "redirected" / "TileXR"
            redirected.mkdir(parents=True)
            (redirected / "keep").write_text("keep")
            (runner_root / "TileXR").symlink_to(
                runner_root / "redirected", target_is_directory=True
            )
            hook = self.materialize_hook(root, runner_root)

            result = subprocess.run(
                [str(hook)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertNotEqual(0, result.returncode)
            self.assertEqual("keep", (redirected / "keep").read_text())

    def test_job_hook_refuses_a_dangling_intermediate_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            runner_root = root / "runner" / "_work"
            runner_root.mkdir(parents=True)
            (runner_root / "TileXR").symlink_to(
                runner_root / "missing", target_is_directory=True
            )
            hook = self.materialize_hook(root, runner_root)

            result = subprocess.run(
                [str(hook)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            self.assertNotEqual(0, result.returncode)


if __name__ == "__main__":
    unittest.main()
