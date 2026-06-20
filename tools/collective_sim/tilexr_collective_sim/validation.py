from typing import List, Optional, Set

from .model import (
    COMM_BUFFER_ROLES,
    COPY_OPS,
    TRANSFER_OPS,
    AlgorithmSpec,
    OpSpec,
    TopologySpec,
    ValidationIssue,
    ValidationReport,
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


def _transfer_uses_only_comm_endpoints(algorithm: AlgorithmSpec, op: OpSpec) -> bool:
    roles = []
    if op.src_buffer is not None and op.src_buffer in algorithm.buffers:
        roles.append(algorithm.buffers[op.src_buffer].role)
    if op.dst_buffer is not None and op.dst_buffer in algorithm.buffers:
        roles.append(algorithm.buffers[op.dst_buffer].role)
    return bool(roles) and all(role in COMM_BUFFER_ROLES for role in roles)


def _validate_transfer_ranks(algorithm: AlgorithmSpec, op: OpSpec) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    for attr in ("src_rank", "dst_rank"):
        value = getattr(op, attr)
        if value is None:
            issues.append(issue("missing_transfer_rank",
                                f"{op.type} missing {attr}",
                                op_id=op.id, rank=op.rank))
            continue
        try:
            rank = int(value)
        except (TypeError, ValueError):
            issues.append(issue("invalid_transfer_rank",
                                f"{op.type} {attr}={value!r} is not an integer rank",
                                op_id=op.id, rank=op.rank))
            continue
        if rank < 0 or rank >= algorithm.rank_count:
            issues.append(issue("invalid_transfer_rank",
                                f"{op.type} {attr}={rank} outside rank_count={algorithm.rank_count}",
                                op_id=op.id, rank=op.rank))

    endpoint_pairs = (
        ("src_rank", op.src_rank, "src_buffer", op.src_buffer),
        ("dst_rank", op.dst_rank, "dst_buffer", op.dst_buffer),
    )
    for rank_attr, rank_value, buffer_attr, buffer_id in endpoint_pairs:
        if rank_value is None or buffer_id is None or buffer_id not in algorithm.buffers:
            continue
        buffer_rank = algorithm.buffers[buffer_id].rank
        try:
            rank = int(rank_value)
        except (TypeError, ValueError):
            continue
        if rank != buffer_rank:
            issues.append(issue("transfer_rank_buffer_mismatch",
                                f"{op.type} {rank_attr}={rank} does not match {buffer_attr} rank={buffer_rank}",
                                op_id=op.id, buffer_id=buffer_id, rank=op.rank))
    return issues


def validate_static(algorithm: AlgorithmSpec, topology: Optional[TopologySpec] = None) -> ValidationReport:
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
        if op.type in TRANSFER_OPS:
            issues.extend(_validate_transfer_ranks(algorithm, op))
        if op.type in TRANSFER_OPS and not _transfer_uses_only_comm_endpoints(algorithm, op):
            issues.append(issue("comm_buffer_endpoint_required",
                                f"{op.type} requires communication-buffer endpoints",
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
