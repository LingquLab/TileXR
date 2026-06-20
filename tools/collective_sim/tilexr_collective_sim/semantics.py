from typing import Dict, List, Set

from .model import AlgorithmSpec, ValidationIssue, ValidationReport, issue, report_from_issues
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


def validate_allgather_semantics(algorithm: AlgorithmSpec) -> ValidationReport:
    static_report = validate_static(algorithm)
    issues: List[ValidationIssue] = list(static_report.issues)
    if algorithm.collective != "allgather":
        issues.append(issue("unsupported_semantic_collective", f"semantic validator only supports allgather, got {algorithm.collective}"))
        return report_from_issues(issues)
    if not static_report.ok:
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
