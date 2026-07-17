from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
