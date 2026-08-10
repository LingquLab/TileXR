from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.moonep import test_npu_e2e


def test_topk_is_duplicate_free_and_creates_prefetch_pressure():
    test_npu_e2e.validate_topk_routing()


def test_direct_python_entry_builds_static_torchrun_command():
    command = test_npu_e2e.build_launch_command(
        python_executable="/usr/bin/python3",
        script_path="/repo/tools/moonep/test_npu_e2e.py",
        nproc_per_node=8,
        master_addr="127.0.0.1",
        master_port=29601,
    )

    assert command == [
        "/usr/bin/python3",
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--node-rank=0",
        "--nproc-per-node=8",
        "--master-addr=127.0.0.1",
        "--master-port=29601",
        "/repo/tools/moonep/test_npu_e2e.py",
    ]


def test_direct_main_launches_workers_with_current_python(monkeypatch):
    captured = {}

    class Result:
        returncode = 7

    def run(command, *, check):
        captured["command"] = command
        captured["check"] = check
        return Result()

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(test_npu_e2e.subprocess, "run", run)

    result = test_npu_e2e.main(
        [
            "--nproc-per-node=6",
            "--master-addr=127.0.0.2",
            "--master-port=29602",
        ]
    )

    assert result == 7
    assert captured["check"] is False
    assert captured["command"][0] == sys.executable
    assert "--nproc-per-node=6" in captured["command"]
    assert "--master-addr=127.0.0.2" in captured["command"]
    assert "--master-port=29602" in captured["command"]
    assert Path(captured["command"][-1]).as_posix().endswith(
        "tools/moonep/test_npu_e2e.py"
    )
