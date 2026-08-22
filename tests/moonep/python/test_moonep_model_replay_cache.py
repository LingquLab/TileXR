from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest

from tools.moonep.model_replay_cache import (
    CacheValidationError,
    ReplayShape,
    begin_cache_capture,
    build_cache_identity,
    cache_key,
    find_cache,
    publish_cache,
    validate_cache_entry,
    validate_route_replay,
)


def _shape() -> ReplayShape:
    return ReplayShape(
        tokens_per_rank=2,
        topk=2,
        hidden_size=128,
        ep_size=2,
        world_size=2,
        expert_count=4,
        ffn_hidden_size=32,
        token_padding=1,
        forward_calls=2,
    )


def _identity(**updates: object) -> dict[str, object]:
    provenance: dict[str, object] = {
        "tilexr_git_sha": "0123456789abcdef",
        "adapter_sha256": "a" * 64,
        "runner_sha256": "b" * 64,
        "install_artifacts": {"libtile-comm.so": "sha256:" + "c" * 64},
        "model_stack": {"mindspeed": "test", "mindspeed_llm": "test"},
        "backend": "tilexr",
        "kernel_version": "test-kernel",
        "cann": "9.1.0-b131",
        "driver": "25.1.rc1",
        "firmware": "test-firmware",
        "soc": "Ascend950PR",
        "topology": {"nodes": ["node-a"], "devices_per_node": 2},
        "rank_mapping": [
            {"rank": 0, "node": 0, "device": 0},
            {"rank": 1, "node": 0, "device": 1},
        ],
    }
    provenance.update(updates)
    order = [
        {"sequence": 0, "phase": "forward", "layer": 0, "stage": "planning"},
        {
            "sequence": 1,
            "phase": "forward",
            "layer": 0,
            "stage": "dispatch_forward",
        },
    ]
    return build_cache_identity(_shape(), provenance=provenance, operator_order=order)


def _route_record(rank: int, call: int) -> dict[str, object]:
    values = (rank, (rank + 1) % 4, (rank + 2) % 4, (rank + 3) % 4)
    raw = struct.pack("<4i", *values)
    counts = [values.count(expert) for expert in range(4)]
    return {
        "call": call,
        "source_call": call,
        "topk_shape": [2, 2],
        "topk_dtype": "torch.int32",
        "topk_encoding": "int32-le-zlib-base64",
        "topk_zlib_base64": base64.b64encode(zlib.compress(raw)).decode("ascii"),
        "topk_sha256": hashlib.sha256(raw).hexdigest(),
        "tokens_per_expert": counts,
        "remote_stats": [1, 1],
        "experts_to_copy": [[0, -1], [1, -1]],
    }


def _write_artifacts(staging: Path, *, ranks: int = 2) -> None:
    route_replay = {
        "schema_version": 1,
        "dimensions": {
            "world_size": 2,
            "tokens_per_rank": 2,
            "topk": 2,
            "expert_count": 4,
            "forward_calls": 2,
        },
        "ranks": {
            str(rank): [_route_record(rank, call) for call in range(2)]
            for rank in range(ranks)
        },
    }
    model_performance = {
        "schema_version": 1,
        "world_size": 2,
        "profiler_enabled": True,
        "ranks": {
            str(rank): {
                "operators": [
                    {
                        "sequence": 0,
                        "stage": "planning",
                        "backend": "tilexr",
                        "kernel_version": "test-kernel",
                        "algorithm_bytes": None,
                        "latency_us": 10.0 + rank,
                        "algorithm_bandwidth_GBps": None,
                    },
                    {
                        "sequence": 1,
                        "stage": "dispatch_forward",
                        "backend": "tilexr",
                        "kernel_version": "test-kernel",
                        "algorithm_bytes": 1024,
                        "latency_us": 20.0 + rank,
                        "algorithm_bandwidth_GBps": 0.0512,
                    },
                ]
            }
            for rank in range(ranks)
        },
    }
    replay = staging / "replay"
    model = staging / "model"
    replay.mkdir(parents=True)
    model.mkdir(parents=True)
    (replay / "route_replay.json").write_text(
        json.dumps(route_replay), encoding="utf-8"
    )
    (model / "performance.json").write_text(
        json.dumps(model_performance), encoding="utf-8"
    )


def test_cache_key_is_deterministic_for_mapping_order() -> None:
    first = _identity()
    reordered = dict(reversed(list(first.items())))

    assert cache_key(first) == cache_key(reordered)
    assert len(cache_key(first)) == 64
    assert cache_key(first) != cache_key(_identity(driver="different"))


def test_remote_stats_remains_two_counters_for_larger_world_sizes() -> None:
    shape = ReplayShape(
        tokens_per_rank=2,
        topk=2,
        hidden_size=128,
        ep_size=4,
        world_size=4,
        expert_count=8,
        ffn_hidden_size=32,
        token_padding=1,
        forward_calls=2,
    )
    ranks = {}
    for rank in range(shape.world_size):
        records = []
        for call in range(shape.forward_calls):
            record = _route_record(rank, call)
            record["tokens_per_expert"] = record["tokens_per_expert"] + [0] * 4
            record["experts_to_copy"] = [[0, -1], [1, -1], [2, -1], [3, -1]]
            records.append(record)
        ranks[str(rank)] = records

    validate_route_replay(
        {
            "schema_version": 1,
            "dimensions": {
                "world_size": shape.world_size,
                "tokens_per_rank": shape.tokens_per_rank,
                "topk": shape.topk,
                "expert_count": shape.expert_count,
                "forward_calls": shape.forward_calls,
            },
            "ranks": ranks,
        },
        shape,
    )


def test_complete_capture_is_published_and_found(tmp_path: Path) -> None:
    identity = _identity()
    staging = begin_cache_capture(tmp_path, identity, capture_id="capture-1")
    assert ".staging-" in staging.name
    _write_artifacts(staging)

    published = publish_cache(staging, identity, capture_id="capture-1")
    assert published.path.is_dir()
    assert (published.path / "manifest.json").is_file()
    assert (published.path / "complete").is_file()
    assert not staging.exists()

    found = find_cache(tmp_path, identity)
    assert found is not None
    assert found.path == published.path
    assert found.manifest["capture_id"] == "capture-1"


def test_partial_all_rank_capture_is_rejected_without_publication(
    tmp_path: Path,
) -> None:
    identity = _identity()
    staging = begin_cache_capture(tmp_path, identity, capture_id="partial")
    _write_artifacts(staging, ranks=1)

    with pytest.raises(CacheValidationError, match="missing ranks.*1"):
        publish_cache(staging, identity, capture_id="partial")

    assert staging.is_dir()
    assert find_cache(tmp_path, identity) is None


def test_corrupt_artifact_is_never_consumed(tmp_path: Path) -> None:
    identity = _identity()
    staging = begin_cache_capture(tmp_path, identity, capture_id="capture-1")
    _write_artifacts(staging)
    published = publish_cache(staging, identity, capture_id="capture-1")

    route_path = published.path / "replay" / "route_replay.json"
    route_path.write_text("corrupt", encoding="utf-8")

    with pytest.raises(CacheValidationError, match="checksum mismatch"):
        validate_cache_entry(published.path, identity)
    assert find_cache(tmp_path, identity) is None


def test_refresh_generation_preserves_previous_valid_cache(tmp_path: Path) -> None:
    identity = _identity()
    first_staging = begin_cache_capture(tmp_path, identity, capture_id="capture-1")
    _write_artifacts(first_staging)
    first = publish_cache(first_staging, identity, capture_id="capture-1")

    failed_staging = begin_cache_capture(tmp_path, identity, capture_id="capture-bad")
    _write_artifacts(failed_staging, ranks=1)
    with pytest.raises(CacheValidationError):
        publish_cache(failed_staging, identity, capture_id="capture-bad")
    assert find_cache(tmp_path, identity).path == first.path

    second_staging = begin_cache_capture(tmp_path, identity, capture_id="capture-2")
    _write_artifacts(second_staging)
    second = publish_cache(second_staging, identity, capture_id="capture-2")

    assert first.path.is_dir()
    assert second.path.is_dir()
    assert first.path != second.path
    assert find_cache(tmp_path, identity).path == second.path


def test_identity_mismatch_is_a_cache_miss(tmp_path: Path) -> None:
    identity = _identity()
    staging = begin_cache_capture(tmp_path, identity, capture_id="capture-1")
    _write_artifacts(staging)
    publish_cache(staging, identity, capture_id="capture-1")

    assert find_cache(tmp_path, _identity(driver="new-driver")) is None
