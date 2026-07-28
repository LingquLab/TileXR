from typing import Dict, List, Sequence, Tuple

from .model import (
    AlgorithmSpec,
    BandwidthCurve,
    CalibrationSpec,
    SimEvent,
    SimulationResult,
    TopologySpec,
    ok_report,
    report_from_issues,
)
from .semantics import validate_allgather_semantics
from .topology import interpolate_curve, oversubscription_scale, resource_ids_for_transfer
from .validation import topological_ops, validate_static


def _resource_curve_and_scale(
    resource: str, topology: TopologySpec, calibration: CalibrationSpec
) -> Tuple[BandwidthCurve, float]:
    if resource.startswith("sdma:"):
        return calibration.curves["sdma_800g"], 1.0
    if resource.startswith("p2p:"):
        return calibration.curves[topology.p2p_curve], 1.0
    if resource.startswith("clos:"):
        return calibration.curves[topology.uplink_curve], oversubscription_scale(topology)
    return calibration.curves[topology.uplink_curve], 1.0


def _resources_for_op(op, topology: TopologySpec) -> tuple:
    if op.type == "copy":
        return (f"sdma:rank{op.rank}",)
    if op.src_rank is not None and op.dst_rank is not None:
        return resource_ids_for_transfer(topology, int(op.src_rank), int(op.dst_rank))
    return tuple(op.resources)


def _semantic_report(algorithm: AlgorithmSpec, topology: TopologySpec):
    if algorithm.collective == "allgather":
        static_report = validate_static(algorithm, topology)
        semantic_report = validate_allgather_semantics(algorithm)
        if static_report.ok:
            return semantic_report
        issues = list(static_report.issues)
        for semantic_issue in semantic_report.issues:
            if semantic_issue not in issues:
                issues.append(semantic_issue)
        return report_from_issues(issues)
    return validate_static(algorithm, topology)


def _event_transfer(resources: Sequence[str], message_bytes: int, topology: TopologySpec,
                    calibration: CalibrationSpec, active_by_resource: Dict[str, int]) -> Tuple[float, float, str]:
    if not resources:
        curve = calibration.curves[topology.uplink_curve]
        effective_gbps = interpolate_curve(curve, message_bytes)
        duration = curve.startup_latency_us + (float(message_bytes) / max(1e-9, effective_gbps * 1000.0))
        return duration, effective_gbps, "none"

    duration = 0.0
    bottleneck = resources[0]
    bottleneck_gbps = float("inf")
    for resource in resources:
        curve, scale = _resource_curve_and_scale(resource, topology, calibration)
        active_count = max(1, active_by_resource.get(resource, 1))
        effective_gbps = interpolate_curve(curve, message_bytes) * scale / float(active_count)
        resource_duration = curve.startup_latency_us + (float(message_bytes) / max(1e-9, effective_gbps * 1000.0))
        duration = max(duration, resource_duration)
        if effective_gbps < bottleneck_gbps:
            bottleneck_gbps = effective_gbps
            bottleneck = resource
    return duration, bottleneck_gbps, bottleneck


def simulate_algorithm(algorithm: AlgorithmSpec, topology: TopologySpec, calibration: CalibrationSpec,
                       message_bytes: int, validate: bool = True) -> SimulationResult:
    validation = _semantic_report(algorithm, topology) if validate else ok_report()
    if not validation.ok:
        return SimulationResult("tilexr_collective_sim_result.v1", algorithm.name, algorithm.collective,
                                algorithm.rank_count, message_bytes, validation, (), 0.0, 0.0, 0.0, "", "estimate")

    finish_by_op: Dict[str, float] = {}
    resource_available: Dict[str, float] = {}
    events: List[SimEvent] = []
    ordered = topological_ops(algorithm)

    while len(finish_by_op) < len(ordered):
        ready = []
        for op in ordered:
            if op.id in finish_by_op or not all(dep in finish_by_op for dep in op.deps):
                continue
            dep_finish = max((finish_by_op.get(dep, 0.0) for dep in op.deps), default=0.0)
            resources = _resources_for_op(op, topology)
            resource_ready = max((resource_available.get(resource, 0.0) for resource in resources), default=0.0)
            start = max(dep_finish, resource_ready)
            ready.append((op, resources, dep_finish, start))
        if not ready:
            break

        batch_start = min(start for _, _, _, start in ready)
        batch = [item for item in ready if item[3] == batch_start]
        active_by_resource: Dict[str, int] = {}
        for _, resources, _, _ in batch:
            for resource in resources:
                active_by_resource[resource] = active_by_resource.get(resource, 0) + 1

        resource_updates: Dict[str, float] = {}
        for op, resources, dep_finish, start in batch:
            op_bytes = op.bytes or message_bytes
            duration, effective_gbps, bottleneck = _event_transfer(resources, op_bytes, topology, calibration, active_by_resource)
            end = start + duration
            for resource in resources:
                resource_updates[resource] = max(resource_updates.get(resource, 0.0), end)
            finish_by_op[op.id] = end
            events.append(SimEvent(op.id, op.type, op.rank, start, end, start - dep_finish, duration,
                                   op_bytes, resources, effective_gbps, bottleneck, "estimate"))
        for resource, available_at in resource_updates.items():
            resource_available[resource] = max(resource_available.get(resource, 0.0), available_at)

    latency = max((event.end_us for event in events), default=0.0)
    algbw = (float(message_bytes) / latency / 1000.0) if latency > 0 else 0.0
    busbw = algbw * float(max(0, algorithm.rank_count - 1)) / float(max(1, algorithm.rank_count))
    bottleneck = max(events, key=lambda event: event.transfer_us).bottleneck_resource if events else ""
    return SimulationResult("tilexr_collective_sim_result.v1", algorithm.name, algorithm.collective,
                            algorithm.rank_count, message_bytes, validation, tuple(events),
                            latency, algbw, busbw, bottleneck, "estimate")
