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
    elif topology.rank_count < topology.oversubscription_enabled_from_ranks:
        clos = "clos:limited"
    else:
        ratio = topology.oversubscription_ratio.replace(":", "to")
        clos = f"clos:dst_server:{dst_server}:oversub_{ratio}"
    return (f"uplink:src:{src_rank}", clos, f"uplink:dst:{dst_rank}")
