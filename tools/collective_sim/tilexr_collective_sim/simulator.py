from typing import Dict, List

from .model import AlgorithmSpec, CalibrationSpec, SimEvent, SimulationResult, TopologySpec, ok_report
from .semantics import validate_allgather_semantics
from .topology import interpolate_curve, oversubscription_scale, resource_ids_for_transfer, transfer_duration_us
from .validation import topological_ops, validate_static


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
