from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import struct
import time
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


CACHE_SCHEMA_VERSION = 1
CAPTURE_SCHEMA_VERSION = 1
ROUTE_REPLAY_SCHEMA_VERSION = 1
MODEL_PERFORMANCE_SCHEMA_VERSION = 1

_REQUIRED_ARTIFACTS = (
    "replay/route_replay.json",
    "model/performance.json",
)
_REQUIRED_PROVENANCE = (
    "tilexr_git_sha",
    "adapter_sha256",
    "runner_sha256",
    "install_artifacts",
    "model_stack",
    "backend",
    "kernel_version",
    "cann",
    "driver",
    "firmware",
    "soc",
    "topology",
    "rank_mapping",
)


class CacheValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ReplayShape:
    tokens_per_rank: int
    topk: int
    hidden_size: int
    ep_size: int
    world_size: int
    expert_count: int
    ffn_hidden_size: int = 2048
    token_padding: int = 1
    forward_calls: int = 10

    def __post_init__(self) -> None:
        values = {
            "S": self.tokens_per_rank,
            "K": self.topk,
            "H": self.hidden_size,
            "EP": self.ep_size,
            "R": self.world_size,
            "E": self.expert_count,
            "Hf": self.ffn_hidden_size,
            "P": self.token_padding,
            "forward_calls": self.forward_calls,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.topk > self.expert_count:
            raise ValueError(f"K must not exceed expert count {self.expert_count}")
        if self.hidden_size % 128 != 0:
            raise ValueError("H must be divisible by 128")
        if self.expert_count % self.ep_size != 0:
            raise ValueError("expert count must be divisible by EP")
        if self.ep_size > self.world_size:
            raise ValueError("EP must not exceed R")

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.ep_size

    def as_dict(self) -> dict[str, int]:
        return {
            "S": self.tokens_per_rank,
            "K": self.topk,
            "H": self.hidden_size,
            "EP": self.ep_size,
            "R": self.world_size,
            "E": self.expert_count,
            "Hf": self.ffn_hidden_size,
            "B": self.experts_per_rank,
            "P": self.token_padding,
            "forward_calls": self.forward_calls,
        }


@dataclass(frozen=True)
class CacheEntry:
    path: Path
    manifest: dict[str, object]


def _canonical_bytes(value: object) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("cache identity must be finite JSON data") from exc
    return encoded.encode("ascii")


def build_cache_identity(
    shape: ReplayShape,
    *,
    provenance: Mapping[str, object],
    operator_order: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    missing = [name for name in _REQUIRED_PROVENANCE if name not in provenance]
    if missing:
        raise ValueError(f"cache provenance is missing: {', '.join(missing)}")
    if not operator_order:
        raise ValueError("model operator order must not be empty")
    normalized_order = [dict(item) for item in operator_order]
    for sequence, item in enumerate(normalized_order):
        if item.get("sequence") != sequence or not isinstance(item.get("stage"), str):
            raise ValueError("model operator order must have contiguous sequences and stages")
    identity = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "capture_schema_version": CAPTURE_SCHEMA_VERSION,
        "shape": shape.as_dict(),
        "operator_order": normalized_order,
        "payload_contract": {
            "captured": (
                "topk,tokens_per_expert,remote_stats,experts_to_copy,"
                "tensor_descriptors"
            ),
            "regenerated": "hidden,route_weights,projections,gradients",
            "regeneration": "deterministic_seed_per_rank_call_stage",
            "dtype": "bf16_payload_fp32_route_weight_int32_topk",
            "layout": "contiguous_model_arena_views",
        },
        "provenance": dict(provenance),
    }
    _canonical_bytes(identity)
    return identity


def cache_key(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_bytes(dict(identity))).hexdigest()


def _safe_capture_id(capture_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", capture_id).strip(".-")
    return safe[:80] or "capture"


def begin_cache_capture(
    cache_root: str | Path,
    identity: Mapping[str, object],
    *,
    capture_id: str,
) -> Path:
    key_root = Path(cache_root) / cache_key(identity)
    key_root.mkdir(parents=True, exist_ok=True)
    staging = key_root / (
        f".staging-{_safe_capture_id(capture_id)}-{os.getpid()}-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    return staging


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CacheValidationError(f"invalid {description}: {path}") from exc
    if not isinstance(value, dict):
        raise CacheValidationError(f"{description} must be a JSON object: {path}")
    return value


def _shape_from_identity(identity: Mapping[str, object]) -> ReplayShape:
    raw = identity.get("shape")
    if not isinstance(raw, Mapping):
        raise CacheValidationError("cache identity is missing shape metadata")
    try:
        return ReplayShape(
            tokens_per_rank=int(raw["S"]),
            topk=int(raw["K"]),
            hidden_size=int(raw["H"]),
            ep_size=int(raw["EP"]),
            world_size=int(raw["R"]),
            expert_count=int(raw["E"]),
            ffn_hidden_size=int(raw["Hf"]),
            token_padding=int(raw["P"]),
            forward_calls=int(raw["forward_calls"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheValidationError("cache identity has invalid shape metadata") from exc


def _missing_rank_message(ranks: Mapping[str, object], world_size: int) -> str | None:
    expected = {str(rank) for rank in range(world_size)}
    actual = set(ranks)
    missing = sorted(expected - actual, key=int)
    extra = sorted(actual - expected)
    if missing:
        return f"missing ranks: {', '.join(missing)}"
    if extra:
        return f"unexpected ranks: {', '.join(extra)}"
    return None


def _validate_route_record(
    record: object,
    *,
    rank: int,
    call: int,
    shape: ReplayShape,
) -> None:
    if not isinstance(record, Mapping):
        raise CacheValidationError(f"rank {rank} call {call} route record is invalid")
    if int(record.get("call", -1)) != call:
        raise CacheValidationError(f"rank {rank} route call ordering is invalid")
    if record.get("topk_shape") != [shape.tokens_per_rank, shape.topk]:
        raise CacheValidationError(f"rank {rank} call {call} TopK shape mismatch")
    if record.get("topk_dtype") != "torch.int32":
        raise CacheValidationError(f"rank {rank} call {call} TopK dtype mismatch")
    if record.get("topk_encoding") != "int32-le-zlib-base64":
        raise CacheValidationError(f"rank {rank} call {call} TopK encoding mismatch")
    try:
        compressed = base64.b64decode(record["topk_zlib_base64"], validate=True)
        raw = zlib.decompress(compressed)
    except (KeyError, TypeError, ValueError, zlib.error) as exc:
        raise CacheValidationError(
            f"rank {rank} call {call} TopK payload is invalid"
        ) from exc
    expected_bytes = shape.tokens_per_rank * shape.topk * 4
    if len(raw) != expected_bytes:
        raise CacheValidationError(f"rank {rank} call {call} TopK byte size mismatch")
    if hashlib.sha256(raw).hexdigest() != record.get("topk_sha256"):
        raise CacheValidationError(f"rank {rank} call {call} TopK checksum mismatch")
    routes = struct.unpack(f"<{shape.tokens_per_rank * shape.topk}i", raw)
    counts = [0] * shape.expert_count
    for token in range(shape.tokens_per_rank):
        row = routes[token * shape.topk : (token + 1) * shape.topk]
        if len(set(row)) != shape.topk:
            raise CacheValidationError(f"rank {rank} call {call} has duplicate TopK routes")
        for expert in row:
            if expert < 0 or expert >= shape.expert_count:
                raise CacheValidationError(
                    f"rank {rank} call {call} has out-of-range expert {expert}"
                )
            counts[expert] += 1
    try:
        recorded_counts = [int(value) for value in record["tokens_per_expert"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise CacheValidationError(
            f"rank {rank} call {call} route histogram is invalid"
        ) from exc
    if recorded_counts != counts:
        raise CacheValidationError(f"rank {rank} call {call} route histogram mismatch")
    remote_stats = record.get("remote_stats")
    if (
        not isinstance(remote_stats, list)
        or len(remote_stats) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in remote_stats
        )
    ):
        raise CacheValidationError(f"rank {rank} call {call} remote_stats mismatch")
    experts_to_copy = record.get("experts_to_copy")
    if (
        not isinstance(experts_to_copy, list)
        or len(experts_to_copy) != shape.world_size
        or any(
            not isinstance(row, list) or len(row) != shape.experts_per_rank
            for row in experts_to_copy
        )
    ):
        raise CacheValidationError(f"rank {rank} call {call} experts_to_copy mismatch")


def validate_route_replay(payload: Mapping[str, object], shape: ReplayShape) -> None:
    if payload.get("schema_version") != ROUTE_REPLAY_SCHEMA_VERSION:
        raise CacheValidationError("route replay schema version mismatch")
    dimensions = payload.get("dimensions")
    expected_dimensions = {
        "world_size": shape.world_size,
        "tokens_per_rank": shape.tokens_per_rank,
        "topk": shape.topk,
        "expert_count": shape.expert_count,
        "forward_calls": shape.forward_calls,
    }
    if not isinstance(dimensions, Mapping) or any(
        dimensions.get(name) != value for name, value in expected_dimensions.items()
    ):
        raise CacheValidationError("route replay dimensions mismatch")
    ranks = payload.get("ranks")
    if not isinstance(ranks, Mapping):
        raise CacheValidationError("route replay ranks are missing")
    rank_error = _missing_rank_message(ranks, shape.world_size)
    if rank_error:
        raise CacheValidationError(rank_error)
    for rank in range(shape.world_size):
        records = ranks[str(rank)]
        if not isinstance(records, list) or len(records) != shape.forward_calls:
            raise CacheValidationError(
                f"rank {rank} must contain {shape.forward_calls} route calls"
            )
        for call, record in enumerate(records):
            _validate_route_record(record, rank=rank, call=call, shape=shape)


def validate_model_performance(
    payload: Mapping[str, object], identity: Mapping[str, object]
) -> None:
    if payload.get("schema_version") != MODEL_PERFORMANCE_SCHEMA_VERSION:
        raise CacheValidationError("model performance schema version mismatch")
    shape = _shape_from_identity(identity)
    if payload.get("world_size") != shape.world_size:
        raise CacheValidationError("model performance world size mismatch")
    ranks = payload.get("ranks")
    if not isinstance(ranks, Mapping):
        raise CacheValidationError("model performance ranks are missing")
    rank_error = _missing_rank_message(ranks, shape.world_size)
    if rank_error:
        raise CacheValidationError(rank_error)
    order = identity.get("operator_order")
    if not isinstance(order, list):
        raise CacheValidationError("cache identity operator order is invalid")
    for rank in range(shape.world_size):
        rank_record = ranks[str(rank)]
        operators = rank_record.get("operators") if isinstance(rank_record, Mapping) else None
        if not isinstance(operators, list) or len(operators) != len(order):
            raise CacheValidationError(f"rank {rank} model operator count mismatch")
        for sequence, (operator, expected) in enumerate(zip(operators, order)):
            if not isinstance(operator, Mapping):
                raise CacheValidationError(f"rank {rank} operator {sequence} is invalid")
            if (
                operator.get("sequence") != sequence
                or operator.get("stage") != expected.get("stage")
            ):
                raise CacheValidationError(f"rank {rank} model operator order mismatch")
            if not isinstance(operator.get("backend"), str) or not isinstance(
                operator.get("kernel_version"), str
            ):
                raise CacheValidationError(f"rank {rank} operator metadata is invalid")
            latency = operator.get("latency_us")
            if (
                isinstance(latency, bool)
                or not isinstance(latency, (int, float))
                or not math.isfinite(float(latency))
                or float(latency) < 0.0
            ):
                raise CacheValidationError(f"rank {rank} operator latency is invalid")
            byte_count = operator.get("algorithm_bytes")
            if byte_count is not None and (
                isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
            ):
                raise CacheValidationError(f"rank {rank} operator bytes are invalid")
            bandwidth = operator.get("algorithm_bandwidth_GBps")
            if bandwidth is not None and (
                isinstance(bandwidth, bool)
                or not isinstance(bandwidth, (int, float))
                or not math.isfinite(float(bandwidth))
                or float(bandwidth) < 0.0
            ):
                raise CacheValidationError(f"rank {rank} operator bandwidth is invalid")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_staging_artifacts(
    staging: Path, identity: Mapping[str, object]
) -> dict[str, str]:
    for relative in _REQUIRED_ARTIFACTS:
        path = staging / relative
        if not path.is_file():
            raise CacheValidationError(f"required cache artifact is missing: {relative}")
    route = _read_json(staging / _REQUIRED_ARTIFACTS[0], "route replay")
    validate_route_replay(route, _shape_from_identity(identity))
    performance = _read_json(
        staging / _REQUIRED_ARTIFACTS[1], "model performance"
    )
    validate_model_performance(performance, identity)
    return {
        relative: _sha256_file(staging / relative) for relative in _REQUIRED_ARTIFACTS
    }


def publish_cache(
    staging: str | Path,
    identity: Mapping[str, object],
    *,
    capture_id: str,
) -> CacheEntry:
    staging_path = Path(staging)
    expected_key_root = staging_path.parent
    expected_key = cache_key(identity)
    if expected_key_root.name != expected_key or not staging_path.name.startswith(
        ".staging-"
    ):
        raise CacheValidationError("staging directory does not belong to cache identity")
    checksums = _validate_staging_artifacts(staging_path, identity)
    created_ns = time.time_ns()
    manifest: dict[str, object] = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": expected_key,
        "capture_id": capture_id,
        "created_ns": created_ns,
        "identity": dict(identity),
        "artifacts": checksums,
    }
    _write_json_atomic(staging_path / "manifest.json", manifest)
    complete = staging_path / "complete"
    with complete.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(f"{expected_key} {capture_id}\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(staging_path)
    destination = expected_key_root / (
        f"generation-{created_ns}-{_safe_capture_id(capture_id)}-{uuid.uuid4().hex[:8]}"
    )
    os.replace(staging_path, destination)
    _fsync_directory(expected_key_root)
    return validate_cache_entry(destination, identity)


def validate_cache_entry(
    path: str | Path, identity: Mapping[str, object]
) -> CacheEntry:
    root = Path(path)
    if not (root / "complete").is_file():
        raise CacheValidationError(f"cache completion marker is missing: {root}")
    manifest = _read_json(root / "manifest.json", "cache manifest")
    expected_key = cache_key(identity)
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise CacheValidationError("cache manifest schema version mismatch")
    if manifest.get("cache_key") != expected_key:
        raise CacheValidationError("cache key mismatch")
    if _canonical_bytes(manifest.get("identity")) != _canonical_bytes(dict(identity)):
        raise CacheValidationError("cache identity mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_REQUIRED_ARTIFACTS):
        raise CacheValidationError("cache artifact manifest mismatch")
    for relative in _REQUIRED_ARTIFACTS:
        artifact = root / relative
        if not artifact.is_file():
            raise CacheValidationError(f"required cache artifact is missing: {relative}")
        if _sha256_file(artifact) != artifacts[relative]:
            raise CacheValidationError(f"cache artifact checksum mismatch: {relative}")
    _validate_staging_artifacts(root, identity)
    return CacheEntry(path=root, manifest=manifest)


def find_cache(
    cache_root: str | Path, identity: Mapping[str, object]
) -> CacheEntry | None:
    key_root = Path(cache_root) / cache_key(identity)
    if not key_root.is_dir():
        return None
    with os.scandir(key_root) as entries:
        candidates = sorted(
            (
                Path(entry.path)
                for entry in entries
                if entry.is_dir() and entry.name.startswith("generation-")
            ),
            key=lambda path: path.name,
            reverse=True,
        )
    for candidate in candidates:
        try:
            return validate_cache_entry(candidate, identity)
        except CacheValidationError:
            continue
    return None
