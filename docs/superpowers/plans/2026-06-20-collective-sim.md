# Collective Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first offline collective simulation tool for generic DAG-based algorithms, with AllGather on 1D-Clos as the runnable example.

**Architecture:** Add a standalone Python tool under `tools/collective_sim/` that parses JSON/YAML-like inputs, validates algorithm DAGs, runs symbolic correctness checks, simulates resource timing with a discrete-event model, and writes JSON/CSV/HTML reports. Keep the tool independent from `op-simulator`, ACL/CANN runtime APIs, and compiled TileXR libraries.

**Tech Stack:** Python 3 stdlib, `unittest`, optional PyYAML for YAML parsing, JSON/CSV/HTML file output, shell commands run from the repository root.

## Global Constraints

- `op-simulator/` is for operator/AICore simulation and must not be used as the topology-level simulator surface.
- The first tool version lives under `tools/collective_sim/`.
- The first complete example is AllGather on simplified 1D-Clos.
- Algorithms are represented as a generic scheduling DAG, not as hard-coded collective implementations.
- DAG buffers must distinguish `user_input`, `user_output`, `comm_buffer`, and optional `registered_comm_buffer`.
- Cross-rank `send`, `recv`, `put`, or `get` operations must use a communication-buffer endpoint.
- DataCopy-mode and UDMA-mode transfers must have at least one endpoint in a communication buffer.
- Local copy-in and copy-out stages are explicit DAG `copy` operations.
- 1D-Clos defaults to `8` ranks per server.
- Intra-server P2P uses a message-size bandwidth curve with upper asymptote `50 GB/s`.
- Device uplink uses a message-size bandwidth curve with upper asymptote `300 GB/s`.
- 64P can be configured as non-blocking.
- 128P and larger configurations can enable `2:1` oversubscription.
- SDMA-backed local movement defaults to an `800 GB/s` bandwidth curve.
- The tool must not call real ACL, CANN, NPU runtime, `libtile-comm.so`, or `libtilexr-collectives.so`.
- Unit and example tests must run without Ascend hardware.

---

## File Map

- Create `tools/collective_sim/tilexr_collective_sim/__init__.py`: package marker and version string.
- Create `tools/collective_sim/tilexr_collective_sim/model.py`: dataclasses, constants, result structures, and serialization helpers shared by the tool.
- Create `tools/collective_sim/tilexr_collective_sim/io.py`: JSON/YAML-compatible document loading and typed parsing into `model.py` dataclasses.
- Create `tools/collective_sim/tilexr_collective_sim/validation.py`: static DAG validation and topological sorting.
- Create `tools/collective_sim/tilexr_collective_sim/semantics.py`: symbolic DAG execution and AllGather semantic validation.
- Create `tools/collective_sim/tilexr_collective_sim/topology.py`: bandwidth curve interpolation, transfer duration helpers, and simplified 1D-Clos resource selection.
- Create `tools/collective_sim/tilexr_collective_sim/simulator.py`: discrete-event simulation, resource contention, validation gate, and result generation.
- Create `tools/collective_sim/tilexr_collective_sim/dsl.py`: Python helpers that generate standard DAG dictionaries.
- Create `tools/collective_sim/tilexr_collective_sim/report.py`: `result.json`, `summary.csv`, and static `report.html` generation.
- Create `tools/collective_sim/tilexr_collective_sim/cli.py`: `validate`, `run`, `sweep`, and `report` command entry points.
- Create `tools/collective_sim/examples/allgather_1d_clos/generate_allgather.py`: reproducible AllGather DAG generator.
- Create `tools/collective_sim/examples/allgather_1d_clos/algorithm.json`: generated AllGather DAG fixture.
- Create `tools/collective_sim/examples/allgather_1d_clos/topology_64p.yaml`: non-blocking 64P topology fixture using JSON-compatible YAML.
- Create `tools/collective_sim/examples/allgather_1d_clos/topology_128p_2to1.yaml`: 128P oversubscribed topology fixture using JSON-compatible YAML.
- Create `tools/collective_sim/examples/allgather_1d_clos/calibration.yaml`: P2P 50G, uplink 300G, and SDMA 800G curve fixture.
- Create `tools/collective_sim/examples/allgather_1d_clos/case.yaml`: single-case run fixture.
- Create `tools/collective_sim/examples/allgather_1d_clos/sweep.yaml`: message-size sweep fixture.
- Create `tools/collective_sim/README.md`: usage, file formats, constraints, and verification commands.
- Create `tests/collective_sim/test_model_io.py`: parser and dataclass tests.
- Create `tests/collective_sim/test_validation.py`: static validation tests.
- Create `tests/collective_sim/test_semantics.py`: symbolic AllGather correctness tests.
- Create `tests/collective_sim/test_topology.py`: bandwidth curve and 1D-Clos resource tests.
- Create `tests/collective_sim/test_simulator.py`: discrete-event and contention tests.
- Create `tests/collective_sim/test_dsl_examples.py`: AllGather DSL and checked-in fixture tests.
- Create `tests/collective_sim/test_report_cli.py`: report and CLI tests.
- Create `tests/collective_sim/test_integration_boundaries.py`: no-hardware and no-runtime dependency guard tests.

## Core Interfaces

All tasks must use these names and signatures.

```python
# tools/collective_sim/tilexr_collective_sim/model.py
COMM_BUFFER_ROLES = {"comm_buffer", "registered_comm_buffer"}
USER_BUFFER_ROLES = {"user_input", "user_output"}
TRANSFER_OPS = {"send", "recv", "put", "get"}
COPY_OPS = {"copy"}

@dataclass(frozen=True)
class BufferSpec:
    id: str
    rank: int
    role: str
    chunks: Sequence[str]

@dataclass(frozen=True)
class OpSpec:
    id: str
    type: str
    rank: int
    bytes: int = 0
    src_buffer: Optional[str] = None
    dst_buffer: Optional[str] = None
    src_rank: Optional[int] = None
    dst_rank: Optional[int] = None
    deps: Sequence[str] = ()
    mode: str = "local"
    resources: Sequence[str] = ()

@dataclass(frozen=True)
class AlgorithmSpec:
    name: str
    collective: str
    rank_count: int
    buffers: Dict[str, BufferSpec]
    ops: Dict[str, OpSpec]
    metadata: Dict[str, Any]

@dataclass(frozen=True)
class CurvePoint:
    bytes: int
    gbps: float

@dataclass(frozen=True)
class BandwidthCurve:
    id: str
    points: Sequence[CurvePoint]
    startup_latency_us: float

@dataclass(frozen=True)
class CalibrationSpec:
    curves: Dict[str, BandwidthCurve]

@dataclass(frozen=True)
class TopologySpec:
    type: str
    rank_count: int
    ranks_per_server: int
    p2p_curve: str
    uplink_curve: str
    non_blocking_until_ranks: int
    oversubscription_enabled_from_ranks: int
    oversubscription_ratio: str
    congestion_model: str

@dataclass(frozen=True)
class CaseSpec:
    algorithm: str
    topology: str
    calibration: str
    message_bytes: Sequence[int]
    validate: bool = True

@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    op_id: Optional[str] = None
    buffer_id: Optional[str] = None
    rank: Optional[int] = None

@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    issues: Sequence[ValidationIssue]

@dataclass(frozen=True)
class SimEvent:
    op_id: str
    op_type: str
    rank: int
    start_us: float
    end_us: float
    wait_us: float
    transfer_us: float
    message_bytes: int
    resources: Sequence[str]
    effective_gbps: float
    bottleneck_resource: str
    data_source: str

@dataclass(frozen=True)
class SimulationResult:
    schema: str
    algorithm: str
    collective: str
    rank_count: int
    message_bytes: int
    validation: ValidationReport
    events: Sequence[SimEvent]
    latency_us: float
    algbw_gbps: float
    busbw_gbps: float
    bottleneck_resource: str
    data_source: str
```

```text
Function signatures shared across modules:
- load_document(path: Path) -> Dict[str, Any]
- load_algorithm(path: Path) -> AlgorithmSpec
- load_topology(path: Path) -> TopologySpec
- load_calibration(path: Path) -> CalibrationSpec
- load_case(path: Path) -> CaseSpec
- validate_static(algorithm: AlgorithmSpec, topology: Optional[TopologySpec] = None) -> ValidationReport
- topological_ops(algorithm: AlgorithmSpec) -> List[OpSpec]
- validate_allgather_semantics(algorithm: AlgorithmSpec) -> ValidationReport
- interpolate_curve(curve: BandwidthCurve, message_bytes: int) -> float
- transfer_duration_us(curve: BandwidthCurve, message_bytes: int, active_flows: int = 1, scale: float = 1.0) -> float
- resource_ids_for_transfer(topology: TopologySpec, src_rank: int, dst_rank: int) -> tuple
- simulate_algorithm(algorithm: AlgorithmSpec, topology: TopologySpec, calibration: CalibrationSpec, message_bytes: int, validate: bool = True) -> SimulationResult
- allgather_direct_algorithm(rank_count: int, message_bytes: int) -> Dict[str, Any]
- write_result(result: SimulationResult, out_dir: Path) -> None
- write_summary(results: Sequence[SimulationResult], path: Path) -> None
- write_html_report(results: Sequence[SimulationResult], path: Path) -> None
- write_html_report_from_plain(result_data: Any, path: Path) -> None
```

## Task 1: Data Model and Input Loading

**Files:**
- Create: `tools/collective_sim/tilexr_collective_sim/__init__.py`
- Create: `tools/collective_sim/tilexr_collective_sim/model.py`
- Create: `tools/collective_sim/tilexr_collective_sim/io.py`
- Create: `tests/collective_sim/test_model_io.py`

**Interfaces:**
- Produces all dataclasses and loader functions listed in "Core Interfaces".
- Consumes no project code outside Python stdlib.

- [ ] **Step 1: Write failing model and loader tests**

Create `tests/collective_sim/test_model_io.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

from tilexr_collective_sim.io import load_algorithm, load_calibration, load_case, load_topology
from tilexr_collective_sim.model import COMM_BUFFER_ROLES, TRANSFER_OPS


class ModelIoTest(unittest.TestCase):
    def test_load_algorithm_with_buffer_roles_and_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithm.json"
            path.write_text(json.dumps({
                "name": "direct_allgather",
                "collective": "allgather",
                "rank_count": 2,
                "buffers": [
                    {"id": "r0_in", "rank": 0, "role": "user_input", "chunks": ["rank0.chunk0"]},
                    {"id": "r0_comm", "rank": 0, "role": "comm_buffer", "chunks": []},
                    {"id": "r1_comm", "rank": 1, "role": "comm_buffer", "chunks": []}
                ],
                "ops": [
                    {"id": "copy_r0", "type": "copy", "rank": 0, "bytes": 1024,
                     "src_buffer": "r0_in", "dst_buffer": "r0_comm", "mode": "sdma"},
                    {"id": "send_r0_to_r1", "type": "send", "rank": 0, "bytes": 1024,
                     "src_rank": 0, "dst_rank": 1, "src_buffer": "r0_comm",
                     "dst_buffer": "r1_comm", "deps": ["copy_r0"], "mode": "datacopy"}
                ],
                "metadata": {"chunk_bytes": 1024}
            }), encoding="utf-8")

            algorithm = load_algorithm(path)

            self.assertEqual(algorithm.name, "direct_allgather")
            self.assertEqual(algorithm.collective, "allgather")
            self.assertEqual(algorithm.rank_count, 2)
            self.assertEqual(algorithm.buffers["r0_comm"].role, "comm_buffer")
            self.assertEqual(algorithm.ops["send_r0_to_r1"].deps, ("copy_r0",))
            self.assertIn("comm_buffer", COMM_BUFFER_ROLES)
            self.assertIn("send", TRANSFER_OPS)

    def test_load_topology_calibration_and_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "topology.yaml").write_text(json.dumps({
                "topology": {
                    "type": "one_d_clos",
                    "rank_count": 128,
                    "ranks_per_server": 8,
                    "intra_server": {"kind": "full_mesh", "bandwidth_curve": "p2p_50g"},
                    "uplink": {"bandwidth_curve": "uplink_300g"},
                    "clos": {
                        "mode": "simplified_pool",
                        "non_blocking_until_ranks": 64,
                        "oversubscription": {"enabled_from_ranks": 128, "ratio": "2:1"},
                        "congestion": {"model": "destination_pool_fair_share"}
                    }
                }
            }), encoding="utf-8")
            (root / "calibration.yaml").write_text(json.dumps({
                "calibration": {
                    "curves": {
                        "p2p_50g": {"kind": "table", "points": [{"bytes": 1024, "gbps": 5}, {"bytes": 67108864, "gbps": 50}], "startup_latency_us": 2.0},
                        "uplink_300g": {"kind": "table", "points": [{"bytes": 1024, "gbps": 10}, {"bytes": 67108864, "gbps": 300}], "startup_latency_us": 3.0},
                        "sdma_800g": {"kind": "table", "points": [{"bytes": 1024, "gbps": 40}, {"bytes": 67108864, "gbps": 800}], "startup_latency_us": 1.0}
                    }
                }
            }), encoding="utf-8")
            (root / "case.yaml").write_text(json.dumps({
                "case": {
                    "algorithm": "algorithm.json",
                    "topology": "topology.yaml",
                    "calibration": "calibration.yaml",
                    "message_bytes": [1024, 4096],
                    "validate": True
                }
            }), encoding="utf-8")

            topology = load_topology(root / "topology.yaml")
            calibration = load_calibration(root / "calibration.yaml")
            case = load_case(root / "case.yaml")

            self.assertEqual(topology.rank_count, 128)
            self.assertEqual(topology.ranks_per_server, 8)
            self.assertEqual(topology.oversubscription_ratio, "2:1")
            self.assertIn("sdma_800g", calibration.curves)
            self.assertEqual(case.message_bytes, (1024, 4096))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_model_io -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tilexr_collective_sim'`.

- [ ] **Step 3: Add the package marker**

Create `tools/collective_sim/tilexr_collective_sim/__init__.py`:

```python
"""Offline collective simulation tooling for TileXR."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Add dataclasses and conversion helpers**

Create `tools/collective_sim/tilexr_collective_sim/model.py` using the exact dataclasses from "Core Interfaces". Add these helper functions at the end of the file:

```python
def issue(code: str, message: str, op_id: Optional[str] = None,
          buffer_id: Optional[str] = None, rank: Optional[int] = None) -> ValidationIssue:
    return ValidationIssue(code=code, message=message, op_id=op_id, buffer_id=buffer_id, rank=rank)


def ok_report() -> ValidationReport:
    return ValidationReport(ok=True, issues=())


def report_from_issues(issues: Iterable[ValidationIssue]) -> ValidationReport:
    items = tuple(issues)
    return ValidationReport(ok=not items, issues=items)


def dataclass_to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: dataclass_to_plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [dataclass_to_plain(item) for item in value]
    if isinstance(value, list):
        return [dataclass_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): dataclass_to_plain(item) for key, item in value.items()}
    return value
```

Imports required at the top of `model.py`:

```python
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence
```

- [ ] **Step 5: Add document loading and typed parsers**

Create `tools/collective_sim/tilexr_collective_sim/io.py`:

```python
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .model import (
    AlgorithmSpec,
    BandwidthCurve,
    BufferSpec,
    CalibrationSpec,
    CaseSpec,
    CurvePoint,
    OpSpec,
    TopologySpec,
)


def load_document(path: Path) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise ValueError(f"{path}: YAML input requires PyYAML or JSON-compatible syntax") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top-level document must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context}: missing required field '{key}'")
    return mapping[key]


def _as_tuple(values: Iterable[Any]) -> tuple:
    return tuple(values)


def load_algorithm(path: Path) -> AlgorithmSpec:
    data = load_document(path)
    buffers = {}
    for item in _required(data, "buffers", str(path)):
        buffer = BufferSpec(
            id=str(_required(item, "id", "buffer")),
            rank=int(_required(item, "rank", "buffer")),
            role=str(_required(item, "role", "buffer")),
            chunks=tuple(str(chunk) for chunk in item.get("chunks", [])),
        )
        buffers[buffer.id] = buffer
    ops = {}
    for item in _required(data, "ops", str(path)):
        op = OpSpec(
            id=str(_required(item, "id", "op")),
            type=str(_required(item, "type", "op")),
            rank=int(_required(item, "rank", "op")),
            bytes=int(item.get("bytes", 0)),
            src_buffer=item.get("src_buffer"),
            dst_buffer=item.get("dst_buffer"),
            src_rank=item.get("src_rank"),
            dst_rank=item.get("dst_rank"),
            deps=tuple(str(dep) for dep in item.get("deps", [])),
            mode=str(item.get("mode", "local")),
            resources=tuple(str(resource) for resource in item.get("resources", [])),
        )
        ops[op.id] = op
    return AlgorithmSpec(
        name=str(_required(data, "name", str(path))),
        collective=str(_required(data, "collective", str(path))),
        rank_count=int(_required(data, "rank_count", str(path))),
        buffers=buffers,
        ops=ops,
        metadata=dict(data.get("metadata", {})),
    )


def load_topology(path: Path) -> TopologySpec:
    root = load_document(path)
    data = root.get("topology", root)
    clos = data.get("clos", {})
    oversub = clos.get("oversubscription", {})
    congestion = clos.get("congestion", {})
    return TopologySpec(
        type=str(_required(data, "type", str(path))),
        rank_count=int(_required(data, "rank_count", str(path))),
        ranks_per_server=int(data.get("ranks_per_server", 8)),
        p2p_curve=str(data.get("intra_server", {}).get("bandwidth_curve", "p2p_50g")),
        uplink_curve=str(data.get("uplink", {}).get("bandwidth_curve", "uplink_300g")),
        non_blocking_until_ranks=int(clos.get("non_blocking_until_ranks", 64)),
        oversubscription_enabled_from_ranks=int(oversub.get("enabled_from_ranks", 128)),
        oversubscription_ratio=str(oversub.get("ratio", "2:1")),
        congestion_model=str(congestion.get("model", "destination_pool_fair_share")),
    )


def load_calibration(path: Path) -> CalibrationSpec:
    root = load_document(path)
    data = root.get("calibration", root)
    curves = {}
    for curve_id, item in _required(data, "curves", str(path)).items():
        points = tuple(
            CurvePoint(bytes=int(point["bytes"]), gbps=float(point["gbps"]))
            for point in item.get("points", [])
        )
        curves[str(curve_id)] = BandwidthCurve(
            id=str(curve_id),
            points=points,
            startup_latency_us=float(item.get("startup_latency_us", 0.0)),
        )
    return CalibrationSpec(curves=curves)


def load_case(path: Path) -> CaseSpec:
    root = load_document(path)
    data = root.get("case", root)
    return CaseSpec(
        algorithm=str(_required(data, "algorithm", str(path))),
        topology=str(_required(data, "topology", str(path))),
        calibration=str(_required(data, "calibration", str(path))),
        message_bytes=tuple(int(value) for value in _required(data, "message_bytes", str(path))),
        validate=bool(data.get("validate", True)),
    )
```

- [ ] **Step 6: Run the test and verify it passes**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_model_io -v
```

Expected: PASS, `Ran 2 tests`.

- [ ] **Step 7: Commit Task 1**

```bash
git add tools/collective_sim/tilexr_collective_sim/__init__.py \
        tools/collective_sim/tilexr_collective_sim/model.py \
        tools/collective_sim/tilexr_collective_sim/io.py \
        tests/collective_sim/test_model_io.py
git commit -m "feat: add collective simulator data model"
```

## Task 2: Static DAG Validation

**Files:**
- Create: `tools/collective_sim/tilexr_collective_sim/validation.py`
- Create: `tests/collective_sim/test_validation.py`

**Interfaces:**
- Consumes: `AlgorithmSpec`, `TopologySpec`, `ValidationReport`, `issue`, `report_from_issues`.
- Produces: `validate_static(algorithm, topology=None) -> ValidationReport` and `topological_ops(algorithm) -> List[OpSpec]`.

- [ ] **Step 1: Write failing validation tests**

Create `tests/collective_sim/test_validation.py`:

```python
import unittest

from tilexr_collective_sim.model import AlgorithmSpec, BufferSpec, OpSpec, TopologySpec
from tilexr_collective_sim.validation import topological_ops, validate_static


def topology(rank_count=2):
    return TopologySpec(
        type="one_d_clos",
        rank_count=rank_count,
        ranks_per_server=8,
        p2p_curve="p2p_50g",
        uplink_curve="uplink_300g",
        non_blocking_until_ranks=64,
        oversubscription_enabled_from_ranks=128,
        oversubscription_ratio="2:1",
        congestion_model="destination_pool_fair_share",
    )


def valid_algorithm():
    buffers = {
        "r0_in": BufferSpec("r0_in", 0, "user_input", ("rank0.chunk0",)),
        "r0_comm": BufferSpec("r0_comm", 0, "comm_buffer", ()),
        "r1_comm": BufferSpec("r1_comm", 1, "comm_buffer", ()),
    }
    ops = {
        "copy_r0": OpSpec("copy_r0", "copy", 0, 1024, "r0_in", "r0_comm", mode="sdma"),
        "send_r0": OpSpec("send_r0", "send", 0, 1024, "r0_comm", "r1_comm", 0, 1, ("copy_r0",), "datacopy"),
    }
    return AlgorithmSpec("valid", "allgather", 2, buffers, ops, {})


class StaticValidationTest(unittest.TestCase):
    def test_valid_dag_passes_and_topological_order_respects_deps(self):
        report = validate_static(valid_algorithm(), topology())
        self.assertTrue(report.ok, report.issues)
        order = [op.id for op in topological_ops(valid_algorithm())]
        self.assertEqual(order, ["copy_r0", "send_r0"])

    def test_direct_user_to_user_send_fails(self):
        algorithm = valid_algorithm()
        buffers = dict(algorithm.buffers)
        buffers["r1_out"] = BufferSpec("r1_out", 1, "user_output", ())
        ops = {
            "bad_send": OpSpec("bad_send", "send", 0, 1024, "r0_in", "r1_out", 0, 1, (), "datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad", "allgather", 2, buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("comm_buffer_endpoint_required", {issue.code for issue in report.issues})

    def test_datacopy_without_comm_endpoint_fails(self):
        algorithm = valid_algorithm()
        buffers = dict(algorithm.buffers)
        buffers["r0_tmp"] = BufferSpec("r0_tmp", 0, "user_output", ())
        ops = {
            "bad_copy": OpSpec("bad_copy", "copy", 0, 1024, "r0_in", "r0_tmp", mode="datacopy"),
        }
        report = validate_static(AlgorithmSpec("bad_copy", "allgather", 2, buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("comm_buffer_endpoint_required", {issue.code for issue in report.issues})

    def test_missing_dep_and_cycle_fail(self):
        algorithm = valid_algorithm()
        ops = {
            "a": OpSpec("a", "wait", 0, deps=("b",)),
            "b": OpSpec("b", "wait", 0, deps=("a",)),
            "c": OpSpec("c", "wait", 0, deps=("missing",)),
        }
        report = validate_static(AlgorithmSpec("cycle", "allgather", 2, algorithm.buffers, ops, {}), topology())
        self.assertFalse(report.ok)
        self.assertIn("missing_dependency", {issue.code for issue in report.issues})
        self.assertIn("cycle_detected", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_validation -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tilexr_collective_sim.validation'`.

- [ ] **Step 3: Add static validation implementation**

Create `tools/collective_sim/tilexr_collective_sim/validation.py`:

```python
from typing import Dict, List, Optional, Set

from .model import (
    COMM_BUFFER_ROLES,
    COPY_OPS,
    TRANSFER_OPS,
    AlgorithmSpec,
    OpSpec,
    TopologySpec,
    ValidationIssue,
    issue,
    report_from_issues,
)


VALID_BUFFER_ROLES = {"user_input", "user_output", "comm_buffer", "registered_comm_buffer"}
VALID_OP_TYPES = {"send", "recv", "put", "get", "copy", "reduce", "wait", "barrier"}
COMM_REQUIRED_MODES = {"datacopy", "udma"}


def _has_comm_endpoint(algorithm: AlgorithmSpec, op: OpSpec) -> bool:
    roles = []
    if op.src_buffer is not None and op.src_buffer in algorithm.buffers:
        roles.append(algorithm.buffers[op.src_buffer].role)
    if op.dst_buffer is not None and op.dst_buffer in algorithm.buffers:
        roles.append(algorithm.buffers[op.dst_buffer].role)
    return any(role in COMM_BUFFER_ROLES for role in roles)


def validate_static(algorithm: AlgorithmSpec, topology: Optional[TopologySpec] = None):
    issues: List[ValidationIssue] = []
    if topology is not None and topology.rank_count != algorithm.rank_count:
        issues.append(issue("rank_count_mismatch",
                            f"algorithm rank_count={algorithm.rank_count} topology rank_count={topology.rank_count}"))
    for buffer in algorithm.buffers.values():
        if buffer.role not in VALID_BUFFER_ROLES:
            issues.append(issue("invalid_buffer_role", f"invalid buffer role {buffer.role}", buffer_id=buffer.id, rank=buffer.rank))
        if buffer.rank < 0 or buffer.rank >= algorithm.rank_count:
            issues.append(issue("invalid_buffer_rank", f"invalid buffer rank {buffer.rank}", buffer_id=buffer.id, rank=buffer.rank))
    for op in algorithm.ops.values():
        if op.type not in VALID_OP_TYPES:
            issues.append(issue("invalid_op_type", f"invalid op type {op.type}", op_id=op.id, rank=op.rank))
        if op.rank < 0 or op.rank >= algorithm.rank_count:
            issues.append(issue("invalid_op_rank", f"invalid op rank {op.rank}", op_id=op.id, rank=op.rank))
        for dep in op.deps:
            if dep not in algorithm.ops:
                issues.append(issue("missing_dependency", f"missing dependency {dep}", op_id=op.id, rank=op.rank))
        for attr in ("src_buffer", "dst_buffer"):
            buffer_id = getattr(op, attr)
            if buffer_id is not None and buffer_id not in algorithm.buffers:
                issues.append(issue("missing_buffer", f"missing {attr} {buffer_id}", op_id=op.id, buffer_id=buffer_id, rank=op.rank))
        if op.type in TRANSFER_OPS and not _has_comm_endpoint(algorithm, op):
            issues.append(issue("comm_buffer_endpoint_required",
                                f"{op.type} requires a communication-buffer endpoint",
                                op_id=op.id, rank=op.rank))
        if op.type in COPY_OPS and op.mode in COMM_REQUIRED_MODES and not _has_comm_endpoint(algorithm, op):
            issues.append(issue("comm_buffer_endpoint_required",
                                f"{op.mode} copy requires a communication-buffer endpoint",
                                op_id=op.id, rank=op.rank))
    if _has_cycle(algorithm):
        issues.append(issue("cycle_detected", "DAG dependencies contain a cycle"))
    return report_from_issues(issues)


def _has_cycle(algorithm: AlgorithmSpec) -> bool:
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(op_id: str) -> bool:
        if op_id in visited:
            return False
        if op_id in visiting:
            return True
        if op_id not in algorithm.ops:
            return False
        visiting.add(op_id)
        for dep in algorithm.ops[op_id].deps:
            if visit(dep):
                return True
        visiting.remove(op_id)
        visited.add(op_id)
        return False

    return any(visit(op_id) for op_id in algorithm.ops)


def topological_ops(algorithm: AlgorithmSpec) -> List[OpSpec]:
    ordered: List[OpSpec] = []
    visited: Set[str] = set()

    def visit(op_id: str) -> None:
        if op_id in visited or op_id not in algorithm.ops:
            return
        for dep in sorted(algorithm.ops[op_id].deps):
            visit(dep)
        visited.add(op_id)
        ordered.append(algorithm.ops[op_id])

    for op_id in sorted(algorithm.ops):
        visit(op_id)
    return ordered
```

- [ ] **Step 4: Run validation tests and Task 1 tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest \
  tests.collective_sim.test_model_io \
  tests.collective_sim.test_validation -v
```

Expected: PASS, `Ran 6 tests`.

- [ ] **Step 5: Commit Task 2**

```bash
git add tools/collective_sim/tilexr_collective_sim/validation.py \
        tests/collective_sim/test_validation.py
git commit -m "feat: validate collective simulator DAGs"
```

## Task 3: Symbolic Correctness Execution

**Files:**
- Create: `tools/collective_sim/tilexr_collective_sim/semantics.py`
- Create: `tests/collective_sim/test_semantics.py`

**Interfaces:**
- Consumes: `topological_ops()`, `validate_static()`, `AlgorithmSpec`.
- Produces: `validate_allgather_semantics(algorithm) -> ValidationReport`.

- [ ] **Step 1: Write failing semantic tests**

Create `tests/collective_sim/test_semantics.py`:

```python
import unittest

from tilexr_collective_sim.model import AlgorithmSpec, BufferSpec, OpSpec
from tilexr_collective_sim.semantics import validate_allgather_semantics


def two_rank_allgather(include_remote_copy=True):
    buffers = {
        "r0_in": BufferSpec("r0_in", 0, "user_input", ("rank0.chunk0",)),
        "r1_in": BufferSpec("r1_in", 1, "user_input", ("rank1.chunk0",)),
        "r0_c0": BufferSpec("r0_c0", 0, "comm_buffer", ()),
        "r0_c1": BufferSpec("r0_c1", 0, "comm_buffer", ()),
        "r1_c0": BufferSpec("r1_c0", 1, "comm_buffer", ()),
        "r1_c1": BufferSpec("r1_c1", 1, "comm_buffer", ()),
        "r0_out": BufferSpec("r0_out", 0, "user_output", ()),
        "r1_out": BufferSpec("r1_out", 1, "user_output", ()),
    }
    ops = {
        "copy_r0_in": OpSpec("copy_r0_in", "copy", 0, 1024, "r0_in", "r0_c0", mode="sdma"),
        "copy_r1_in": OpSpec("copy_r1_in", "copy", 1, 1024, "r1_in", "r1_c1", mode="sdma"),
        "send_r0_to_r1": OpSpec("send_r0_to_r1", "send", 0, 1024, "r0_c0", "r1_c0", 0, 1, ("copy_r0_in",), "datacopy"),
        "send_r1_to_r0": OpSpec("send_r1_to_r0", "send", 1, 1024, "r1_c1", "r0_c1", 1, 0, ("copy_r1_in",), "datacopy"),
        "out_r0_local": OpSpec("out_r0_local", "copy", 0, 1024, "r0_c0", "r0_out", deps=("copy_r0_in",), mode="sdma"),
        "out_r1_local": OpSpec("out_r1_local", "copy", 1, 1024, "r1_c1", "r1_out", deps=("copy_r1_in",), mode="sdma"),
    }
    if include_remote_copy:
        ops.update({
            "out_r0_remote": OpSpec("out_r0_remote", "copy", 0, 1024, "r0_c1", "r0_out", deps=("send_r1_to_r0",), mode="sdma"),
            "out_r1_remote": OpSpec("out_r1_remote", "copy", 1, 1024, "r1_c0", "r1_out", deps=("send_r0_to_r1",), mode="sdma"),
        })
    return AlgorithmSpec("two_rank", "allgather", 2, buffers, ops, {})


class SemanticsTest(unittest.TestCase):
    def test_allgather_symbolic_execution_passes(self):
        report = validate_allgather_semantics(two_rank_allgather())
        self.assertTrue(report.ok, report.issues)

    def test_missing_remote_chunk_fails(self):
        report = validate_allgather_semantics(two_rank_allgather(include_remote_copy=False))
        self.assertFalse(report.ok)
        self.assertIn("allgather_missing_chunk", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_semantics -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tilexr_collective_sim.semantics'`.

- [ ] **Step 3: Add symbolic execution**

Create `tools/collective_sim/tilexr_collective_sim/semantics.py`:

```python
from typing import Dict, List, Set

from .model import AlgorithmSpec, ValidationIssue, issue, report_from_issues
from .validation import topological_ops, validate_static


def _append_chunks(dst: List[str], chunks: List[str]) -> None:
    for chunk in chunks:
        if chunk not in dst:
            dst.append(chunk)


def _execute_symbolic(algorithm: AlgorithmSpec) -> Dict[str, List[str]]:
    state: Dict[str, List[str]] = {
        buffer_id: list(buffer.chunks)
        for buffer_id, buffer in algorithm.buffers.items()
    }
    for op in topological_ops(algorithm):
        if op.type in {"copy", "send", "recv", "put", "get"}:
            if op.src_buffer is None or op.dst_buffer is None:
                continue
            _append_chunks(state.setdefault(op.dst_buffer, []), list(state.get(op.src_buffer, [])))
        elif op.type == "reduce":
            if op.src_buffer is not None and op.dst_buffer is not None:
                _append_chunks(state.setdefault(op.dst_buffer, []), list(state.get(op.src_buffer, [])))
        elif op.type in {"wait", "barrier"}:
            continue
    return state


def validate_allgather_semantics(algorithm: AlgorithmSpec):
    static_report = validate_static(algorithm)
    issues: List[ValidationIssue] = list(static_report.issues)
    if algorithm.collective != "allgather":
        issues.append(issue("unsupported_semantic_collective", f"semantic validator only supports allgather, got {algorithm.collective}"))
        return report_from_issues(issues)
    state = _execute_symbolic(algorithm)
    expected: Set[str] = {f"rank{rank}.chunk0" for rank in range(algorithm.rank_count)}
    output_buffers = [buffer for buffer in algorithm.buffers.values() if buffer.role == "user_output"]
    ranks_with_output = {buffer.rank for buffer in output_buffers}
    for rank in range(algorithm.rank_count):
        if rank not in ranks_with_output:
            issues.append(issue("allgather_missing_output", f"rank {rank} has no user_output buffer", rank=rank))
    for buffer in output_buffers:
        observed = set(state.get(buffer.id, []))
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            issues.append(issue("allgather_missing_chunk",
                                f"{buffer.id} missing chunks {missing}",
                                buffer_id=buffer.id, rank=buffer.rank))
        if extra:
            issues.append(issue("allgather_extra_chunk",
                                f"{buffer.id} has unexpected chunks {extra}",
                                buffer_id=buffer.id, rank=buffer.rank))
    return report_from_issues(issues)
```

- [ ] **Step 4: Run semantic, validation, and model tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest \
  tests.collective_sim.test_model_io \
  tests.collective_sim.test_validation \
  tests.collective_sim.test_semantics -v
```

Expected: PASS, `Ran 8 tests`.

- [ ] **Step 5: Commit Task 3**

```bash
git add tools/collective_sim/tilexr_collective_sim/semantics.py \
        tests/collective_sim/test_semantics.py
git commit -m "feat: add symbolic allgather validation"
```

## Task 4: Bandwidth Curves and 1D-Clos Topology

**Files:**
- Create: `tools/collective_sim/tilexr_collective_sim/topology.py`
- Create: `tests/collective_sim/test_topology.py`

**Interfaces:**
- Consumes: `BandwidthCurve`, `CalibrationSpec`, `TopologySpec`.
- Produces: `interpolate_curve()`, `transfer_duration_us()`, and `resource_ids_for_transfer()`.

- [ ] **Step 1: Write failing topology tests**

Create `tests/collective_sim/test_topology.py`:

```python
import unittest

from tilexr_collective_sim.model import BandwidthCurve, CurvePoint, TopologySpec
from tilexr_collective_sim.topology import interpolate_curve, oversubscription_scale, resource_ids_for_transfer, transfer_duration_us


def curve():
    return BandwidthCurve("test", (CurvePoint(1024, 10.0), CurvePoint(4096, 40.0)), 2.0)


def topology(rank_count):
    return TopologySpec("one_d_clos", rank_count, 8, "p2p_50g", "uplink_300g", 64, 128, "2:1", "destination_pool_fair_share")


class TopologyTest(unittest.TestCase):
    def test_curve_interpolation_and_duration(self):
        self.assertEqual(interpolate_curve(curve(), 1024), 10.0)
        self.assertEqual(interpolate_curve(curve(), 4096), 40.0)
        self.assertAlmostEqual(interpolate_curve(curve(), 2560), 25.0)
        duration = transfer_duration_us(curve(), 1024, active_flows=1)
        self.assertGreater(duration, 2.0)

    def test_sdma_active_flow_scaling(self):
        duration_one = transfer_duration_us(curve(), 4096, active_flows=1)
        duration_two = transfer_duration_us(curve(), 4096, active_flows=2)
        self.assertGreater(duration_two, duration_one)

    def test_resource_selection_same_server_and_cross_server(self):
        self.assertEqual(resource_ids_for_transfer(topology(64), 0, 7), ("p2p:s0:0->7",))
        self.assertEqual(
            resource_ids_for_transfer(topology(64), 0, 8),
            ("uplink:src:0", "clos:nonblocking", "uplink:dst:8"),
        )
        self.assertIn("clos:dst_server:1:oversub_2to1", resource_ids_for_transfer(topology(128), 0, 8))

    def test_oversubscription_scale(self):
        self.assertEqual(oversubscription_scale(topology(64)), 1.0)
        self.assertEqual(oversubscription_scale(topology(128)), 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_topology -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tilexr_collective_sim.topology'`.

- [ ] **Step 3: Implement topology helpers**

Create `tools/collective_sim/tilexr_collective_sim/topology.py`:

```python
from .model import BandwidthCurve, TopologySpec


def interpolate_curve(curve: BandwidthCurve, message_bytes: int) -> float:
    if not curve.points:
        raise ValueError(f"curve {curve.id} has no points")
    points = sorted(curve.points, key=lambda point: point.bytes)
    if message_bytes <= points[0].bytes:
        return points[0].gbps
    if message_bytes >= points[-1].bytes:
        return points[-1].gbps
    for left, right in zip(points, points[1:]):
        if left.bytes <= message_bytes <= right.bytes:
            ratio = float(message_bytes - left.bytes) / float(right.bytes - left.bytes)
            return left.gbps + (right.gbps - left.gbps) * ratio
    return points[-1].gbps


def transfer_duration_us(curve: BandwidthCurve, message_bytes: int, active_flows: int = 1, scale: float = 1.0) -> float:
    if message_bytes < 0:
        raise ValueError("message_bytes must be non-negative")
    flows = max(1, int(active_flows))
    base_gbps = interpolate_curve(curve, message_bytes)
    effective_gbps = max(1e-9, base_gbps * scale / flows)
    bytes_per_us = effective_gbps * 1000.0
    return curve.startup_latency_us + (float(message_bytes) / bytes_per_us)


def _server(topology: TopologySpec, rank: int) -> int:
    return rank // topology.ranks_per_server


def oversubscription_scale(topology: TopologySpec) -> float:
    if topology.rank_count < topology.oversubscription_enabled_from_ranks:
        return 1.0
    if topology.oversubscription_ratio == "2:1":
        return 0.5
    if ":" in topology.oversubscription_ratio:
        left, right = topology.oversubscription_ratio.split(":", 1)
        return float(right) / float(left)
    return 1.0


def resource_ids_for_transfer(topology: TopologySpec, src_rank: int, dst_rank: int):
    src_server = _server(topology, src_rank)
    dst_server = _server(topology, dst_rank)
    if src_server == dst_server:
        return (f"p2p:s{src_server}:{src_rank}->{dst_rank}",)
    if topology.rank_count <= topology.non_blocking_until_ranks:
        clos = "clos:nonblocking"
    else:
        ratio = topology.oversubscription_ratio.replace(":", "to")
        clos = f"clos:dst_server:{dst_server}:oversub_{ratio}"
    return (f"uplink:src:{src_rank}", clos, f"uplink:dst:{dst_rank}")
```

- [ ] **Step 4: Run topology and previous tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
```

Expected: PASS, `Ran 12 tests`.

- [ ] **Step 5: Commit Task 4**

```bash
git add tools/collective_sim/tilexr_collective_sim/topology.py \
        tests/collective_sim/test_topology.py
git commit -m "feat: add 1d clos topology model"
```

## Task 5: Discrete-Event Simulator

**Files:**
- Create: `tools/collective_sim/tilexr_collective_sim/simulator.py`
- Create: `tests/collective_sim/test_simulator.py`

**Interfaces:**
- Consumes: loaders, validators, semantics, topology helpers.
- Produces: `simulate_algorithm(algorithm: AlgorithmSpec, topology: TopologySpec, calibration: CalibrationSpec, message_bytes: int, validate: bool = True) -> SimulationResult`.

- [ ] **Step 1: Write failing simulator tests**

Create `tests/collective_sim/test_simulator.py`:

```python
import unittest

from tests.collective_sim.test_semantics import two_rank_allgather
from tilexr_collective_sim.model import BandwidthCurve, CalibrationSpec, CurvePoint, OpSpec, TopologySpec
from tilexr_collective_sim.simulator import simulate_algorithm


def calibration():
    return CalibrationSpec(curves={
        "p2p_50g": BandwidthCurve("p2p_50g", (CurvePoint(1024, 5), CurvePoint(67108864, 50)), 2.0),
        "uplink_300g": BandwidthCurve("uplink_300g", (CurvePoint(1024, 10), CurvePoint(67108864, 300)), 3.0),
        "sdma_800g": BandwidthCurve("sdma_800g", (CurvePoint(1024, 40), CurvePoint(67108864, 800)), 1.0),
    })


def topology(rank_count=2):
    return TopologySpec("one_d_clos", rank_count, 8, "p2p_50g", "uplink_300g", 64, 128, "2:1", "destination_pool_fair_share")


class SimulatorTest(unittest.TestCase):
    def test_simulates_valid_allgather_and_records_copy_events(self):
        result = simulate_algorithm(two_rank_allgather(), topology(2), calibration(), 1024, validate=True)
        self.assertTrue(result.validation.ok, result.validation.issues)
        self.assertGreater(result.latency_us, 0.0)
        event_ids = {event.op_id for event in result.events}
        self.assertIn("copy_r0_in", event_ids)
        self.assertIn("send_r0_to_r1", event_ids)
        copy_event = next(event for event in result.events if event.op_id == "copy_r0_in")
        self.assertEqual(copy_event.resources, ("sdma:rank0",))
        self.assertEqual(copy_event.data_source, "estimate")

    def test_validation_gate_blocks_bad_algorithm(self):
        algorithm = two_rank_allgather()
        ops = dict(algorithm.ops)
        ops["bad"] = OpSpec("bad", "send", 0, 1024, "r0_in", "r1_out", 0, 1, (), "datacopy")
        bad = algorithm.__class__(algorithm.name, algorithm.collective, algorithm.rank_count, algorithm.buffers, ops, algorithm.metadata)
        result = simulate_algorithm(bad, topology(2), calibration(), 1024, validate=True)
        self.assertFalse(result.validation.ok)
        self.assertEqual(result.events, ())

    def test_same_ready_time_flows_share_resource(self):
        result = simulate_algorithm(two_rank_allgather(), topology(2), calibration(), 1024, validate=True)
        send_events = [event for event in result.events if event.op_type == "send"]
        self.assertEqual(len(send_events), 2)
        self.assertTrue(all(event.effective_gbps <= 5.0 for event in send_events))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the simulator test and verify it fails**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_simulator -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tilexr_collective_sim.simulator'`.

- [ ] **Step 3: Add simulator implementation**

Create `tools/collective_sim/tilexr_collective_sim/simulator.py`:

```python
from typing import Dict, List, Tuple

from .model import AlgorithmSpec, CalibrationSpec, SimEvent, SimulationResult, TopologySpec, ok_report
from .semantics import validate_allgather_semantics
from .topology import interpolate_curve, oversubscription_scale, resource_ids_for_transfer, transfer_duration_us
from .validation import validate_static, topological_ops


def _curve_for_event(op_type: str, resources: tuple, topology: TopologySpec, calibration: CalibrationSpec):
    if resources and resources[0].startswith("sdma:"):
        return calibration.curves["sdma_800g"]
    if resources and resources[0].startswith("p2p:"):
        return calibration.curves[topology.p2p_curve]
    return calibration.curves[topology.uplink_curve]


def _resources_for_op(op, topology: TopologySpec) -> tuple:
    if op.type == "copy":
        return (f"sdma:rank{op.rank}",)
    if op.src_rank is not None and op.dst_rank is not None:
        return resource_ids_for_transfer(topology, int(op.src_rank), int(op.dst_rank))
    return tuple(op.resources)


def _semantic_report(algorithm: AlgorithmSpec):
    if algorithm.collective == "allgather":
        return validate_allgather_semantics(algorithm)
    return validate_static(algorithm)


def simulate_algorithm(algorithm: AlgorithmSpec, topology: TopologySpec, calibration: CalibrationSpec,
                       message_bytes: int, validate: bool = True) -> SimulationResult:
    validation = _semantic_report(algorithm) if validate else ok_report()
    if not validation.ok:
        return SimulationResult("tilexr_collective_sim_result.v1", algorithm.name, algorithm.collective,
                                algorithm.rank_count, message_bytes, validation, (), 0.0, 0.0, 0.0, "", "estimate")

    finish_by_op: Dict[str, float] = {}
    resource_available: Dict[str, float] = {}
    events: List[SimEvent] = []
    ordered = topological_ops(algorithm)

    for op in ordered:
        dep_finish = max((finish_by_op.get(dep, 0.0) for dep in op.deps), default=0.0)
        resources = _resources_for_op(op, topology)
        resource_ready = max((resource_available.get(resource, 0.0) for resource in resources), default=0.0)
        start = max(dep_finish, resource_ready)
        active_count = 1
        for other in ordered:
            if other.id == op.id or other.id in finish_by_op:
                continue
            if set(_resources_for_op(other, topology)) & set(resources) and set(other.deps) == set(op.deps):
                active_count += 1
        curve = _curve_for_event(op.type, resources, topology, calibration)
        scale = oversubscription_scale(topology) if any(resource.startswith("clos:") for resource in resources) else 1.0
        duration = transfer_duration_us(curve, op.bytes or message_bytes, active_flows=active_count, scale=scale)
        end = start + duration
        for resource in resources:
            resource_available[resource] = end
        finish_by_op[op.id] = end
        effective_gbps = interpolate_curve(curve, op.bytes or message_bytes) * scale / max(1, active_count)
        bottleneck = resources[0] if resources else "none"
        events.append(SimEvent(op.id, op.type, op.rank, start, end, start - dep_finish, duration,
                               op.bytes or message_bytes, resources, effective_gbps, bottleneck, "estimate"))

    latency = max((event.end_us for event in events), default=0.0)
    algbw = (float(message_bytes) / latency / 1000.0) if latency > 0 else 0.0
    busbw = algbw * float(max(0, algorithm.rank_count - 1)) / float(max(1, algorithm.rank_count))
    bottleneck = max(events, key=lambda event: event.transfer_us).bottleneck_resource if events else ""
    return SimulationResult("tilexr_collective_sim_result.v1", algorithm.name, algorithm.collective,
                            algorithm.rank_count, message_bytes, validation, tuple(events),
                            latency, algbw, busbw, bottleneck, "estimate")
```

- [ ] **Step 4: Run all simulator tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
```

Expected: PASS, `Ran 15 tests`.

- [ ] **Step 5: Commit Task 5**

```bash
git add tools/collective_sim/tilexr_collective_sim/simulator.py \
        tests/collective_sim/test_simulator.py
git commit -m "feat: simulate collective DAG timing"
```

## Task 6: Python DSL and AllGather Example

**Files:**
- Create: `tools/collective_sim/tilexr_collective_sim/dsl.py`
- Create: `tools/collective_sim/examples/allgather_1d_clos/generate_allgather.py`
- Create: `tools/collective_sim/examples/allgather_1d_clos/algorithm.json`
- Create: `tools/collective_sim/examples/allgather_1d_clos/topology_64p.yaml`
- Create: `tools/collective_sim/examples/allgather_1d_clos/topology_128p_2to1.yaml`
- Create: `tools/collective_sim/examples/allgather_1d_clos/calibration.yaml`
- Create: `tools/collective_sim/examples/allgather_1d_clos/case.yaml`
- Create: `tools/collective_sim/examples/allgather_1d_clos/sweep.yaml`
- Create: `tests/collective_sim/test_dsl_examples.py`

**Interfaces:**
- Consumes: `load_algorithm()`, `validate_allgather_semantics()`.
- Produces: `allgather_direct_algorithm(rank_count, message_bytes) -> Dict[str, Any]` and checked-in fixtures.

- [ ] **Step 1: Write failing DSL/example tests**

Create `tests/collective_sim/test_dsl_examples.py`:

```python
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tilexr_collective_sim.dsl import allgather_direct_algorithm
from tilexr_collective_sim.io import load_algorithm, load_calibration, load_case, load_topology
from tilexr_collective_sim.semantics import validate_allgather_semantics


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tools" / "collective_sim" / "examples" / "allgather_1d_clos"


class DslExampleTest(unittest.TestCase):
    def test_direct_allgather_uses_comm_buffers_and_copy_stages(self):
        data = allgather_direct_algorithm(rank_count=4, message_bytes=1024)
        roles = {buffer["role"] for buffer in data["buffers"]}
        op_types = {op["type"] for op in data["ops"]}
        self.assertIn("comm_buffer", roles)
        self.assertIn("user_input", roles)
        self.assertIn("user_output", roles)
        self.assertIn("copy", op_types)
        self.assertIn("send", op_types)

    def test_generated_algorithm_passes_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithm.json"
            path.write_text(json.dumps(allgather_direct_algorithm(4, 1024), indent=2), encoding="utf-8")
            report = validate_allgather_semantics(load_algorithm(path))
            self.assertTrue(report.ok, report.issues)

    def test_checked_in_example_files_load(self):
        algorithm = load_algorithm(EXAMPLE / "algorithm.json")
        self.assertTrue(validate_allgather_semantics(algorithm).ok)
        self.assertEqual(load_topology(EXAMPLE / "topology_64p.yaml").rank_count, 64)
        self.assertEqual(load_topology(EXAMPLE / "topology_128p_2to1.yaml").rank_count, 128)
        self.assertIn("sdma_800g", load_calibration(EXAMPLE / "calibration.yaml").curves)
        self.assertEqual(load_case(EXAMPLE / "case.yaml").message_bytes, (1024, 1048576))

    def test_generator_rewrites_algorithm_file(self):
        result = subprocess.run(
            [sys.executable, str(EXAMPLE / "generate_allgather.py"), "--rank-count", "4", "--message-bytes", "1024"],
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")},
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("algorithm.json", result.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_dsl_examples -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tilexr_collective_sim.dsl'`.

- [ ] **Step 3: Implement the AllGather DSL**

Create `tools/collective_sim/tilexr_collective_sim/dsl.py`:

```python
from typing import Any, Dict, List


def allgather_direct_algorithm(rank_count: int, message_bytes: int) -> Dict[str, Any]:
    buffers: List[Dict[str, Any]] = []
    ops: List[Dict[str, Any]] = []
    for rank in range(rank_count):
        buffers.append({"id": f"r{rank}_in", "rank": rank, "role": "user_input", "chunks": [f"rank{rank}.chunk0"]})
        buffers.append({"id": f"r{rank}_out", "rank": rank, "role": "user_output", "chunks": []})
        for src in range(rank_count):
            buffers.append({"id": f"r{rank}_comm_from_{src}", "rank": rank, "role": "comm_buffer", "chunks": []})
    for rank in range(rank_count):
        ops.append({
            "id": f"copy_in_r{rank}",
            "type": "copy",
            "rank": rank,
            "bytes": message_bytes,
            "src_buffer": f"r{rank}_in",
            "dst_buffer": f"r{rank}_comm_from_{rank}",
            "mode": "sdma",
        })
    for src in range(rank_count):
        for dst in range(rank_count):
            if src == dst:
                continue
            ops.append({
                "id": f"send_r{src}_to_r{dst}",
                "type": "send",
                "rank": src,
                "bytes": message_bytes,
                "src_rank": src,
                "dst_rank": dst,
                "src_buffer": f"r{src}_comm_from_{src}",
                "dst_buffer": f"r{dst}_comm_from_{src}",
                "deps": [f"copy_in_r{src}"],
                "mode": "datacopy",
            })
    for rank in range(rank_count):
        for src in range(rank_count):
            dep = f"copy_in_r{rank}" if src == rank else f"send_r{src}_to_r{rank}"
            ops.append({
                "id": f"copy_out_r{rank}_from_{src}",
                "type": "copy",
                "rank": rank,
                "bytes": message_bytes,
                "src_buffer": f"r{rank}_comm_from_{src}",
                "dst_buffer": f"r{rank}_out",
                "deps": [dep],
                "mode": "sdma",
            })
    return {
        "name": "direct_allgather",
        "collective": "allgather",
        "rank_count": rank_count,
        "buffers": buffers,
        "ops": ops,
        "metadata": {"chunk_bytes": message_bytes, "description": "direct all-to-all AllGather over communication buffers"},
    }
```

- [ ] **Step 4: Add the generator and fixture files**

Create `tools/collective_sim/examples/allgather_1d_clos/generate_allgather.py`:

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from tilexr_collective_sim.dsl import allgather_direct_algorithm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-count", type=int, default=4)
    parser.add_argument("--message-bytes", type=int, default=1024)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent / "algorithm.json"
    out.write_text(json.dumps(allgather_direct_algorithm(args.rank_count, args.message_bytes), indent=2) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create the JSON-compatible YAML fixtures with these top-level fields:

```json
{"topology":{"type":"one_d_clos","rank_count":64,"ranks_per_server":8,"intra_server":{"kind":"full_mesh","bandwidth_curve":"p2p_50g"},"uplink":{"bandwidth_curve":"uplink_300g"},"clos":{"mode":"simplified_pool","non_blocking_until_ranks":64,"oversubscription":{"enabled_from_ranks":128,"ratio":"2:1"},"congestion":{"model":"destination_pool_fair_share"}}}}
```

Use that content for `topology_64p.yaml`, changing only `"rank_count":128` for `topology_128p_2to1.yaml`.

Create `calibration.yaml`:

```json
{"calibration":{"curves":{"p2p_50g":{"kind":"table","points":[{"bytes":1024,"gbps":5},{"bytes":1048576,"gbps":35},{"bytes":67108864,"gbps":50}],"startup_latency_us":2.0},"uplink_300g":{"kind":"table","points":[{"bytes":1024,"gbps":10},{"bytes":1048576,"gbps":120},{"bytes":67108864,"gbps":300}],"startup_latency_us":3.0},"sdma_800g":{"kind":"table","points":[{"bytes":1024,"gbps":40},{"bytes":1048576,"gbps":500},{"bytes":67108864,"gbps":800}],"startup_latency_us":1.0}}}}
```

Create `case.yaml`:

```json
{"case":{"algorithm":"algorithm.json","topology":"topology_64p.yaml","calibration":"calibration.yaml","message_bytes":[1024,1048576],"validate":true}}
```

Create `sweep.yaml`:

```json
{"sweep":{"algorithms":["algorithm.json"],"topologies":["topology_64p.yaml","topology_128p_2to1.yaml"],"calibration":"calibration.yaml","message_bytes":[1024,4096,1048576,67108864],"validate":true}}
```

Generate `algorithm.json`:

```bash
PYTHONPATH=tools/collective_sim python3 tools/collective_sim/examples/allgather_1d_clos/generate_allgather.py --rank-count 4 --message-bytes 1024
```

Expected: stdout contains `algorithm.json`.

- [ ] **Step 5: Run all tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
```

Expected: PASS, `Ran 19 tests`.

- [ ] **Step 6: Commit Task 6**

```bash
git add tools/collective_sim/tilexr_collective_sim/dsl.py \
        tools/collective_sim/examples/allgather_1d_clos \
        tests/collective_sim/test_dsl_examples.py
git commit -m "feat: add allgather simulator example"
```

## Task 7: Reports and CLI

**Files:**
- Create: `tools/collective_sim/tilexr_collective_sim/report.py`
- Create: `tools/collective_sim/tilexr_collective_sim/cli.py`
- Create: `tests/collective_sim/test_report_cli.py`

**Interfaces:**
- Consumes: loader, validator, simulator, `SimulationResult`.
- Produces: CLI commands `validate`, `run`, and `report`.

- [ ] **Step 1: Write failing report and CLI tests**

Create `tests/collective_sim/test_report_cli.py`:

```python
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.collective_sim.test_simulator import calibration, topology
from tests.collective_sim.test_semantics import two_rank_allgather
from tilexr_collective_sim.report import write_html_report, write_result, write_summary
from tilexr_collective_sim.simulator import simulate_algorithm


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "tools" / "collective_sim" / "examples" / "allgather_1d_clos"


class ReportCliTest(unittest.TestCase):
    def test_report_writes_json_csv_and_html_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            result = simulate_algorithm(two_rank_allgather(), topology(2), calibration(), 1024)
            write_result(result, out)
            write_summary([result], out / "summary.csv")
            write_html_report([result], out / "report.html")
            data = json.loads((out / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(data["schema"], "tilexr_collective_sim_result.v1")
            summary = (out / "summary.csv").read_text(encoding="utf-8")
            self.assertIn("algorithm,collective,rank_count,message_bytes,latency_us", summary)
            html = (out / "report.html").read_text(encoding="utf-8")
            self.assertIn("Algorithm Selection", html)
            self.assertIn("Congestion Drilldown", html)
            self.assertIn("Rank Timeline", html)
            self.assertIn("estimate", html)

    def test_cli_validate_and_run_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            validate = subprocess.run(
                [sys.executable, "-m", "tilexr_collective_sim.cli", "validate",
                 str(EXAMPLE / "algorithm.json"), "--topology", str(EXAMPLE / "topology_64p.yaml")],
                cwd=str(ROOT),
                env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(validate.returncode, 0, validate.stderr)
            self.assertIn("validation ok", validate.stdout)
            run = subprocess.run(
                [sys.executable, "-m", "tilexr_collective_sim.cli", "run", str(EXAMPLE / "case.yaml"), "--out", str(out)],
                cwd=str(EXAMPLE),
                env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue((out / "result.json").exists())
            self.assertTrue((out / "summary.csv").exists())
            self.assertTrue((out / "report.html").exists())

    def test_cli_report_regenerates_html_from_results_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            regenerated = Path(tmp) / "regenerated.html"
            run = subprocess.run(
                [sys.executable, "-m", "tilexr_collective_sim.cli", "run", str(EXAMPLE / "case.yaml"), "--out", str(out)],
                cwd=str(EXAMPLE),
                env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            report = subprocess.run(
                [sys.executable, "-m", "tilexr_collective_sim.cli", "report", str(out / "results.json"), "--out", str(regenerated)],
                cwd=str(EXAMPLE),
                env={**os.environ, "PYTHONPATH": str(ROOT / "tools" / "collective_sim")},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(report.returncode, 0, report.stderr)
            self.assertIn("Algorithm Selection", regenerated.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_report_cli -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tilexr_collective_sim.report'`.

- [ ] **Step 3: Add report writer**

Create `tools/collective_sim/tilexr_collective_sim/report.py`:

```python
import csv
import html
import json
from pathlib import Path
from typing import Sequence

from .model import SimulationResult, dataclass_to_plain


def write_result(result: SimulationResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "result.json").write_text(json.dumps(dataclass_to_plain(result), indent=2) + "\n", encoding="utf-8")


def write_summary(results: Sequence[SimulationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "collective", "rank_count", "message_bytes", "latency_us",
                         "algbw_gbps", "busbw_gbps", "bottleneck_resource", "data_source", "validation_ok"])
        for result in results:
            writer.writerow([result.algorithm, result.collective, result.rank_count, result.message_bytes,
                             f"{result.latency_us:.6f}", f"{result.algbw_gbps:.6f}", f"{result.busbw_gbps:.6f}",
                             result.bottleneck_resource, result.data_source, str(result.validation.ok).lower()])


def write_html_report(results: Sequence[SimulationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_html_report_from_plain([dataclass_to_plain(result) for result in results], path)


def write_html_report_from_plain(result_data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = result_data if isinstance(result_data, list) else [result_data]
    rows = []
    for result in items:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result.get('algorithm', '')))}</td>"
            f"<td>{result.get('rank_count', '')}</td>"
            f"<td>{result.get('message_bytes', '')}</td>"
            f"<td>{float(result.get('latency_us', 0.0)):.3f}</td>"
            f"<td>{float(result.get('algbw_gbps', 0.0)):.3f}</td>"
            f"<td>{html.escape(str(result.get('bottleneck_resource', '')))}</td>"
            f"<td>{html.escape(str(result.get('data_source', '')))}</td>"
            "</tr>"
        )
    event_rows = []
    for result in items:
        for event in result.get("events", []):
            event_rows.append(
                "<tr>"
                f"<td>{html.escape(str(event.get('op_id', '')))}</td>"
                f"<td>{event.get('rank', '')}</td>"
                f"<td>{float(event.get('start_us', 0.0)):.3f}</td>"
                f"<td>{float(event.get('end_us', 0.0)):.3f}</td>"
                f"<td>{html.escape(','.join(event.get('resources', [])))}</td>"
                "</tr>"
            )
    document = (
        "<!doctype html><html><head><meta charset='utf-8'><title>TileXR Collective Simulation</title>"
        "<style>body{font-family:sans-serif;margin:24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}</style>"
        "</head><body>"
        "<h1>Algorithm Selection</h1>"
        "<table><thead><tr><th>algorithm</th><th>ranks</th><th>bytes</th><th>latency_us</th><th>algbw_gbps</th><th>bottleneck</th><th>source</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<h1>Congestion Drilldown</h1><p>Resource bottlenecks and effective bandwidth are listed per event.</p>"
        "<h1>Rank Timeline</h1>"
        "<table><thead><tr><th>op</th><th>rank</th><th>start_us</th><th>end_us</th><th>resources</th></tr></thead>"
        f"<tbody>{''.join(event_rows)}</tbody></table>"
        "</body></html>\n"
    )
    path.write_text(document, encoding="utf-8")
```

- [ ] **Step 4: Add CLI**

Create `tools/collective_sim/tilexr_collective_sim/cli.py`:

```python
import argparse
import json
from pathlib import Path

from .io import load_algorithm, load_calibration, load_case, load_topology
from .model import dataclass_to_plain
from .report import write_html_report, write_html_report_from_plain, write_result, write_summary
from .semantics import validate_allgather_semantics
from .simulator import simulate_algorithm
from .validation import validate_static


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def cmd_validate(args) -> int:
    algorithm = load_algorithm(Path(args.algorithm))
    topology = load_topology(Path(args.topology)) if args.topology else None
    static_report = validate_static(algorithm, topology)
    semantic_report = validate_allgather_semantics(algorithm) if algorithm.collective == "allgather" else static_report
    report = semantic_report if not static_report.issues else static_report
    if report.ok:
        print("validation ok")
        return 0
    for item in report.issues:
        print(f"{item.code}: {item.message}")
    return 1


def cmd_run(args) -> int:
    case_path = Path(args.case).resolve()
    base = case_path.parent
    case = load_case(case_path)
    algorithm = load_algorithm(_resolve(base, case.algorithm))
    topology = load_topology(_resolve(base, case.topology))
    calibration = load_calibration(_resolve(base, case.calibration))
    results = [simulate_algorithm(algorithm, topology, calibration, size, validate=case.validate)
               for size in case.message_bytes]
    out_dir = Path(args.out)
    write_result(results[0], out_dir)
    (out_dir / "results.json").write_text(json.dumps([dataclass_to_plain(result) for result in results], indent=2) + "\n",
                                     encoding="utf-8")
    write_summary(results, out_dir / "summary.csv")
    write_html_report(results, out_dir / "report.html")
    print(f"wrote {out_dir}")
    return 0 if all(result.validation.ok for result in results) else 1


def cmd_report(args) -> int:
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    write_html_report_from_plain(data, Path(args.out))
    print(f"wrote {args.out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="collective-sim")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("algorithm")
    validate.add_argument("--topology")
    validate.set_defaults(func=cmd_validate)
    run = sub.add_parser("run")
    run.add_argument("case")
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)
    report = sub.add_parser("report")
    report.add_argument("result")
    report.add_argument("--out", required=True)
    report.set_defaults(func=cmd_report)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run report and CLI tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_report_cli -v
```

Expected: PASS, `Ran 3 tests`.

- [ ] **Step 6: Run all tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
```

Expected: PASS, `Ran 22 tests`.

- [ ] **Step 7: Commit Task 7**

```bash
git add tools/collective_sim/tilexr_collective_sim/report.py \
        tools/collective_sim/tilexr_collective_sim/cli.py \
        tests/collective_sim/test_report_cli.py
git commit -m "feat: add collective simulator reports and cli"
```

## Task 8: Boundary Tests, Sweep Support, and Documentation

**Files:**
- Create: `tools/collective_sim/README.md`
- Create: `tests/collective_sim/test_integration_boundaries.py`
- Modify: `tools/collective_sim/tilexr_collective_sim/cli.py`
- Modify: `tools/collective_sim/tilexr_collective_sim/io.py`

**Interfaces:**
- Consumes all previous task outputs.
- Produces final no-hardware verification command and documented usage.

- [ ] **Step 1: Write failing boundary tests**

Create `tests/collective_sim/test_integration_boundaries.py`:

```python
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
```

- [ ] **Step 2: Run boundary tests and verify they fail**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_integration_boundaries -v
```

Expected: FAIL because the `sweep` subcommand is not registered yet and `README.md` does not exist.

- [ ] **Step 3: Add sweep loading and CLI behavior**

In `tools/collective_sim/tilexr_collective_sim/io.py`, add:

```python
def load_sweep(path: Path) -> Dict[str, Any]:
    root = load_document(path)
    data = root.get("sweep", root)
    for key in ("algorithms", "topologies", "calibration", "message_bytes"):
        _required(data, key, str(path))
    return dict(data)
```

In `tools/collective_sim/tilexr_collective_sim/cli.py`, import `load_sweep`:

```python
from .io import load_algorithm, load_calibration, load_case, load_sweep, load_topology
```

Add the sweep command handler before `main()`:

```python
def cmd_sweep(args) -> int:
    sweep_path = Path(args.sweep).resolve()
    base = sweep_path.parent
    sweep = load_sweep(sweep_path)
    calibration = load_calibration(_resolve(base, sweep["calibration"]))
    results = []
    for algorithm_path in sweep["algorithms"]:
        algorithm = load_algorithm(_resolve(base, algorithm_path))
        for topology_path in sweep["topologies"]:
            topology = load_topology(_resolve(base, topology_path))
            for size in sweep["message_bytes"]:
                results.append(simulate_algorithm(algorithm, topology, calibration, int(size),
                                                  validate=bool(sweep.get("validate", True))))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps([dataclass_to_plain(result) for result in results], indent=2) + "\n",
                                      encoding="utf-8")
    if results:
        write_result(results[0], out_dir)
    write_summary(results, out_dir / "summary.csv")
    write_html_report(results, out_dir / "report.html")
    print(f"wrote {out_dir}")
    return 0 if all(result.validation.ok for result in results) else 1
```

Then add the sweep subcommand in `main()` after the `run` subcommand:

```python
sweep = sub.add_parser("sweep")
sweep.add_argument("sweep")
sweep.add_argument("--out", required=True)
sweep.set_defaults(func=cmd_sweep)
```

- [ ] **Step 4: Add README**

Create `tools/collective_sim/README.md`:

```markdown
# TileXR Collective Simulation Tool

This directory contains the offline collective simulation tool. It evaluates generic collective algorithm DAGs without Ascend hardware, ACL, CANN runtime calls, or compiled TileXR libraries.

## Constraints

- Algorithms are standard JSON DAGs.
- Cross-rank communication acts on communication buffer roles.
- DataCopy-mode and UDMA-mode transfers require at least one communication buffer endpoint.
- Local copy-in and copy-out are explicit DAG `copy` operations.
- The first topology model is simplified 1D-Clos.
- P2P uses a message-size curve that approaches 50 GB/s.
- Uplink uses a message-size curve that approaches 300 GB/s.
- SDMA local copy uses a default 800 GB/s curve until calibration overrides it.

## Commands

Run from the repository root with:

```bash
export PYTHONPATH=tools/collective_sim
python3 -m tilexr_collective_sim.cli validate tools/collective_sim/examples/allgather_1d_clos/algorithm.json --topology tools/collective_sim/examples/allgather_1d_clos/topology_64p.yaml
python3 -m tilexr_collective_sim.cli run tools/collective_sim/examples/allgather_1d_clos/case.yaml --out run/collective_sim/allgather
python3 -m tilexr_collective_sim.cli sweep tools/collective_sim/examples/allgather_1d_clos/sweep.yaml --out run/collective_sim/allgather_sweep
```

The run and sweep commands write:

- `result.json`
- `results.json`
- `summary.csv`
- `report.html`

## Test

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
```
```

- [ ] **Step 5: Run boundary tests**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest tests.collective_sim.test_integration_boundaries -v
```

Expected: PASS, `Ran 3 tests`.

- [ ] **Step 6: Run full test suite for the tool**

Run:

```bash
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
```

Expected: PASS, `Ran 25 tests`.

- [ ] **Step 7: Run the example CLI manually**

Run:

```bash
rm -rf run/collective_sim/allgather run/collective_sim/allgather_sweep
PYTHONPATH=tools/collective_sim python3 -m tilexr_collective_sim.cli validate \
  tools/collective_sim/examples/allgather_1d_clos/algorithm.json \
  --topology tools/collective_sim/examples/allgather_1d_clos/topology_64p.yaml
PYTHONPATH=tools/collective_sim python3 -m tilexr_collective_sim.cli run \
  tools/collective_sim/examples/allgather_1d_clos/case.yaml \
  --out run/collective_sim/allgather
PYTHONPATH=tools/collective_sim python3 -m tilexr_collective_sim.cli sweep \
  tools/collective_sim/examples/allgather_1d_clos/sweep.yaml \
  --out run/collective_sim/allgather_sweep
```

Expected:

```text
validation ok
wrote run/collective_sim/allgather
wrote run/collective_sim/allgather_sweep
```

Confirm:

```bash
test -f run/collective_sim/allgather/result.json
test -f run/collective_sim/allgather/summary.csv
test -f run/collective_sim/allgather/report.html
test -f run/collective_sim/allgather_sweep/results.json
test -f run/collective_sim/allgather_sweep/summary.csv
test -f run/collective_sim/allgather_sweep/report.html
```

Expected: all `test -f` commands exit with status 0.

- [ ] **Step 8: Commit Task 8**

```bash
git add tools/collective_sim/README.md \
        tools/collective_sim/tilexr_collective_sim/io.py \
        tools/collective_sim/tilexr_collective_sim/cli.py \
        tests/collective_sim/test_integration_boundaries.py
git commit -m "docs: document collective simulator workflow"
```

## Final Verification

Run this from the repository root after all tasks are complete:

```bash
git status --short
PYTHONPATH=tools/collective_sim python3 -m unittest discover -s tests/collective_sim -v
rm -rf run/collective_sim/allgather run/collective_sim/allgather_sweep
PYTHONPATH=tools/collective_sim python3 -m tilexr_collective_sim.cli validate \
  tools/collective_sim/examples/allgather_1d_clos/algorithm.json \
  --topology tools/collective_sim/examples/allgather_1d_clos/topology_64p.yaml
PYTHONPATH=tools/collective_sim python3 -m tilexr_collective_sim.cli run \
  tools/collective_sim/examples/allgather_1d_clos/case.yaml \
  --out run/collective_sim/allgather
PYTHONPATH=tools/collective_sim python3 -m tilexr_collective_sim.cli sweep \
  tools/collective_sim/examples/allgather_1d_clos/sweep.yaml \
  --out run/collective_sim/allgather_sweep
```

Expected evidence:

- `python3 -m unittest discover -s tests/collective_sim -v` reports all tests passing.
- `validate` prints `validation ok`.
- `run` prints `wrote run/collective_sim/allgather`.
- `sweep` prints `wrote run/collective_sim/allgather_sweep`.
- `run/collective_sim/allgather/report.html` exists.
- `run/collective_sim/allgather_sweep/report.html` exists.

## Spec Coverage Self-Review

- Generic DAG model: Tasks 1, 2, 3, and 6 define and exercise the DAG.
- AllGather example: Task 6 creates the DSL and checked-in example.
- Static validation: Task 2 validates ids, dependencies, ranks, references, cycles, and communication-buffer endpoint rules.
- Semantic correctness: Task 3 validates symbolic AllGather output.
- Communication buffer constraints: Tasks 2, 3, 5, 6, and 8 test and document copy-in/copy-out and endpoint rules.
- 1D-Clos topology: Task 4 implements P2P, uplink, non-blocking 64P, and oversubscribed 128P resource selection.
- SDMA 800 GB/s: Tasks 4, 5, 6, and 8 include the default SDMA curve and local-copy timing.
- Discrete-event simulation: Task 5 implements dependency timing, resource timing, fair-share scaling, and result events.
- Reports and visualization: Task 7 writes JSON, CSV, and HTML with algorithm selection, congestion drilldown, and rank timeline sections.
- CLI and file formats: Tasks 1, 6, 7, and 8 implement loaders, examples, `validate`, `run`, `sweep`, and documented commands.
- No-hardware boundary: Task 8 source-checks runtime imports and runs the complete test suite without Ascend hardware.
