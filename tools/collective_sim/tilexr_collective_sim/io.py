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


def _sequence(value: Any, key: str, context: str) -> Iterable[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context}: field '{key}' must be a list")
    return value


def load_algorithm(path: Path) -> AlgorithmSpec:
    data = load_document(path)
    buffers = {}
    for item in _required(data, "buffers", str(path)):
        buffer_id = str(_required(item, "id", "buffer"))
        if buffer_id in buffers:
            raise ValueError(f"{path}: duplicate buffer id {buffer_id}")
        buffer = BufferSpec(
            id=buffer_id,
            rank=int(_required(item, "rank", "buffer")),
            role=str(_required(item, "role", "buffer")),
            chunks=tuple(str(chunk) for chunk in item.get("chunks", [])),
        )
        buffers[buffer.id] = buffer
    ops = {}
    for item in _required(data, "ops", str(path)):
        op_id = str(_required(item, "id", "op"))
        if op_id in ops:
            raise ValueError(f"{path}: duplicate op id {op_id}")
        op = OpSpec(
            id=op_id,
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


def load_sweep(path: Path) -> Dict[str, Any]:
    root = load_document(path)
    data = root.get("sweep", root)
    for key in ("calibration", "message_bytes"):
        _required(data, key, str(path))
    sweep = dict(data)
    if "cases" in data:
        cases = []
        for index, case in enumerate(_sequence(data["cases"], "cases", str(path))):
            context = f"{path}: cases[{index}]"
            if not isinstance(case, Mapping):
                raise ValueError(f"{context}: case must be a mapping")
            cases.append({
                "algorithm": str(_required(case, "algorithm", context)),
                "topology": str(_required(case, "topology", context)),
            })
        sweep["cases"] = tuple(cases)
        return sweep

    algorithms = _sequence(_required(data, "algorithms", str(path)), "algorithms", str(path))
    topologies = _sequence(_required(data, "topologies", str(path)), "topologies", str(path))
    sweep["cases"] = tuple(
        {"algorithm": str(algorithm), "topology": str(topology)}
        for algorithm in algorithms
        for topology in topologies
    )
    return sweep
