#!/usr/bin/env python3
#
# Copyright (c) 2026 TileXR Project
#

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_C = REPO_ROOT / "tests" / "ccu" / "ccu_public_direct_api_compile_probe.c"
INCLUDE_DIR = REPO_ROOT / "src" / "include"


class TileXRCcuPublicApiCompileProbeTest(unittest.TestCase):
    def test_probe_source_uses_only_public_api_header(self):
        source = PROBE_C.read_text(encoding="utf-8")

        self.assertIn('#include "tilexr_api.h"', source)
        for needle in [
            '#include "tilexr_comm.h"',
            '#include "ccu/',
            "#include <hcomm/",
            "#include <hccl/",
            "runtime/kernel.h",
            "rtCcuTaskInfo_t",
            "rtCCULaunch",
            "TileXRCcuDirectInstallAttempt",
            "TileXRCcuTask",
        ]:
            with self.subTest(needle=needle):
                self.assertNotIn(needle, source)

    def test_probe_compiles_as_external_c_translation_unit(self):
        compiler = shutil.which("gcc") or shutil.which("cc")
        if compiler is None:
            self.skipTest("no C compiler found")

        with tempfile.TemporaryDirectory() as temp_dir:
            object_file = Path(temp_dir) / "ccu_public_direct_api_compile_probe.o"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-I",
                    str(INCLUDE_DIR),
                    "-c",
                    str(PROBE_C),
                    "-o",
                    str(object_file),
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_probe_compiles_as_external_cpp_translation_unit(self):
        compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("no C++ compiler found")

        with tempfile.TemporaryDirectory() as temp_dir:
            object_file = Path(temp_dir) / "ccu_public_direct_api_compile_probe.o"
            result = subprocess.run(
                [
                    compiler,
                    "-std=c++14",
                    "-I",
                    str(INCLUDE_DIR),
                    "-x",
                    "c++",
                    "-c",
                    str(PROBE_C),
                    "-o",
                    str(object_file),
                ],
                cwd=REPO_ROOT,
                check=False,
                text=True,
                capture_output=True,
            )

        self.assertEqual(0, result.returncode, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
