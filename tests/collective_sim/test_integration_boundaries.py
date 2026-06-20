import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "collective_sim"
EXAMPLE = TOOL / "examples" / "allgather_1d_clos"


class IntegrationBoundaryTest(unittest.TestCase):
    def test_tool_sources_do_not_import_runtime_or_hardware_modules(self):
        forbidden = ("import acl", "import torch", "torch_npu", "import hccl", "ctypes.CDLL", "libtile")
        for path in (TOOL / "tilexr_collective_sim").glob("*.py"):
            source = path.read_text(encoding="utf-8")
            for token in forbidden:
                self.assertNotIn(token, source, f"{token} found in {path}")

    def test_sweep_command_writes_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "sweep"
            result = subprocess.run(
                [sys.executable, "-m", "tilexr_collective_sim.cli", "sweep", str(EXAMPLE / "sweep.yaml"), "--out", str(out)],
                cwd=str(EXAMPLE),
                env={**os.environ, "PYTHONPATH": str(TOOL)},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / "results.json").exists())
            self.assertTrue((out / "summary.csv").exists())
            self.assertTrue((out / "report.html").exists())

    def test_readme_lists_core_commands_and_constraints(self):
        readme = (TOOL / "README.md").read_text(encoding="utf-8")
        self.assertIn("collective-sim validate", readme)
        self.assertIn("collective-sim run", readme)
        self.assertIn("collective-sim sweep", readme)
        self.assertIn("communication buffer", readme)
        self.assertIn("SDMA", readme)
        self.assertIn("800 GB/s", readme)
        self.assertIn("no Ascend hardware", readme)


if __name__ == "__main__":
    unittest.main()
