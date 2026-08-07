from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping


_DTYPES = {"bfloat16", "float16"}
_ROUTE_PATTERNS = {"balanced", "fan_in", "skew"}
_CASE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    tokens_per_rank: int
    topk: int
    expert_count: int
    hidden_size: int
    gate_grad_shape: tuple[int, ...] | None = None
    up_grad_shape: tuple[int, ...] | None = None
    down_grad_shape: tuple[int, ...] | None = None
    dtype: str = "bfloat16"
    seed: int = 1234
    warmup: int = 5
    iterations: int = 20
    correctness: bool = True
    route_pattern: str = "balanced"

    def __post_init__(self) -> None:
        if (
            self.case_id in (".", "..")
            or _CASE_ID_PATTERN.fullmatch(self.case_id) is None
        ):
            raise ValueError(
                "case_id must start with an ASCII letter or digit and contain only "
                "letters, digits, dot, underscore, or hyphen"
            )
        for name in ("tokens_per_rank", "topk", "expert_count", "hidden_size"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype}")
        if self.route_pattern not in _ROUTE_PATTERNS:
            raise ValueError(
                "route_pattern must be one of "
                f"{sorted(_ROUTE_PATTERNS)}, got {self.route_pattern}"
            )
        if self.warmup < 0 or self.iterations <= 0:
            raise ValueError("warmup must be non-negative and iterations must be positive")
        if self.topk > self.expert_count:
            raise ValueError("topk cannot exceed expert_count")
        for name in ("gate_grad_shape", "up_grad_shape", "down_grad_shape"):
            shape = getattr(self, name)
            if shape is None:
                continue
            normalized = tuple(int(value) for value in shape)
            if not normalized or len(normalized) > 3 or any(value <= 0 for value in normalized):
                raise ValueError(f"{name} must contain one to three positive dimensions")
            object.__setattr__(self, name, normalized)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "BenchmarkCase":
        aliases = {
            "id": "case_id",
            "S": "tokens_per_rank",
            "K": "topk",
            "E": "expert_count",
            "H": "hidden_size",
            "iters": "iterations",
        }
        normalized = {aliases.get(key, key): value for key, value in raw.items()}
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(normalized) - allowed)
        if unknown:
            raise ValueError(f"unknown benchmark case fields: {', '.join(unknown)}")
        return cls(**normalized)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def reduce_grad_inner_shape(self, projection: str) -> tuple[int, ...]:
        value = getattr(self, f"{projection}_grad_shape")
        return (self.hidden_size,) if value is None else tuple(value)


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        payload = payload.get("cases")
    if not isinstance(payload, list) or not payload:
        raise ValueError("benchmark JSON must be a non-empty list or {'cases': [...]} object")
    cases = [BenchmarkCase.from_mapping(item) for item in payload]
    ids = [item.case_id for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark case_id values must be unique")
    return cases


def select_cases(cases: Iterable[BenchmarkCase], case_ids: str | None) -> list[BenchmarkCase]:
    values = list(cases)
    if not case_ids:
        return values
    requested = [item.strip() for item in case_ids.split(",") if item.strip()]
    by_id = {item.case_id: item for item in values}
    missing = [item for item in requested if item not in by_id]
    if missing:
        raise ValueError(f"unknown case ids: {', '.join(missing)}")
    return [by_id[item] for item in requested]


def apply_overrides(case: BenchmarkCase, args: argparse.Namespace) -> BenchmarkCase:
    updates = {}
    for name in (
        "tokens_per_rank",
        "topk",
        "expert_count",
        "hidden_size",
        "gate_grad_shape",
        "up_grad_shape",
        "down_grad_shape",
        "route_pattern",
        "dtype",
        "seed",
        "warmup",
        "iterations",
        "correctness",
    ):
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    return replace(case, **updates)


def build_case_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cases", required=True, help="JSON case file")
    parser.add_argument("--case-ids", default=None, help="comma-separated case ids")
    parser.add_argument("--tokens-per-rank", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--expert-count", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--gate-grad-shape", type=_parse_shape, default=None)
    parser.add_argument("--up-grad-shape", type=_parse_shape, default=None)
    parser.add_argument("--down-grad-shape", type=_parse_shape, default=None)
    parser.add_argument("--route-pattern", choices=sorted(_ROUTE_PATTERNS), default=None)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    correctness = parser.add_mutually_exclusive_group()
    correctness.add_argument("--correctness", dest="correctness", action="store_true")
    correctness.add_argument("--no-correctness", dest="correctness", action="store_false")
    parser.set_defaults(correctness=None)


def _parse_shape(value: str) -> tuple[int, ...]:
    parts = tuple(int(item) for item in value.lower().replace("x", ",").split(",") if item)
    if not parts or len(parts) > 3 or any(item <= 0 for item in parts):
        raise argparse.ArgumentTypeError("gradient shape must be 1-3 positive dimensions")
    return parts
