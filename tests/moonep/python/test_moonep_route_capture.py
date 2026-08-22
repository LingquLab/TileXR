from __future__ import annotations

import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

import pytest

from tools.moonep.mindspeed.build_route_replay import build_route_replay
from tools.moonep.model_replay_cache import ReplayShape, validate_route_replay


ROOT = Path(__file__).resolve().parents[3]
ADAPTER = (
    ROOT / "tools" / "moonep" / "mindspeed" / "tilexr_mindspeed_adapter.py"
).read_text(encoding="utf-8")


def _write_capture(
    root: Path,
    *,
    rank: int,
    call: int,
    capture_id: str = "capture-1",
    corrupt_hash: bool = False,
) -> None:
    routes = (0, 1, 2, 3)
    raw = struct.pack("<4i", *routes)
    payload = {
        "schema_version": 1,
        "complete": True,
        "rank": rank,
        "call": call,
        "source_call": call + 5,
        "capture_id": capture_id,
        "topk_shape": [2, 2],
        "topk_dtype": "torch.int32",
        "topk_encoding": "int32-le-zlib-base64",
        "topk_zlib_base64": base64.b64encode(zlib.compress(raw)).decode("ascii"),
        "topk_sha256": "0" * 64 if corrupt_hash else hashlib.sha256(raw).hexdigest(),
        "tokens_per_expert": [1, 1, 1, 1],
        "remote_stats": [1, 1],
        "experts_to_copy": [[0, 1], [2, 3]],
        "payload_seed": rank * 100 + call,
        "tensor_descriptors": {
            "hidden": {
                "shape": [2, 128],
                "stride": [128, 1],
                "dtype": "torch.bfloat16",
                "contiguous": True,
                "storage_offset": 0,
            }
        },
    }
    (root / f"rank{rank}_call{call:02d}.json").write_text(
        json.dumps(payload), encoding="utf-8"
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
        forward_calls=2,
    )


def test_framework_operator_prewarm_precedes_backend_construction() -> None:
    helper = ADAPTER.index("def _prewarm_framework_ops()")
    factory = ADAPTER.index("def create_tilexr_moonep_backend(**kwargs)")
    call = ADAPTER.index("    _prewarm_framework_ops()", factory)
    buffer_selection = ADAPTER.index(
        '    kwargs["buffer_cls"] = MindSpeedTileXRBuffer', factory
    )

    assert helper < factory < call < buffer_selection
    assert 'TILEXR_MINDSPEED_PREWARM_FRAMEWORK_OPS' in ADAPTER
    assert "torch.npu.synchronize()" in ADAPTER
    assert "TILEXR_MINDSPEED_FRAMEWORK_OPS_PREWARM=complete" in ADAPTER


def test_capture_builder_validates_and_preserves_replay_inputs(tmp_path: Path) -> None:
    for rank in range(2):
        for call in range(2):
            _write_capture(tmp_path, rank=rank, call=call)

    replay = build_route_replay(
        tmp_path,
        capture_id="capture-1",
        source_git_sha="abc123",
        world_size=2,
        forward_calls=2,
        tokens_per_rank=2,
        topk=2,
        expert_count=4,
        hidden_size=128,
        ep_size=2,
    )

    validate_route_replay(replay, _shape())
    assert replay["ranks"]["1"][1]["source_call"] == 6
    assert replay["ranks"]["1"][1]["payload_seed"] == 101
    assert replay["ranks"]["0"][0]["tensor_descriptors"]["hidden"][
        "shape"
    ] == [2, 128]
    assert replay["provenance"]["source_git_sha"] == "abc123"


def test_capture_builder_rejects_missing_rank_call(tmp_path: Path) -> None:
    for rank in range(2):
        for call in range(2):
            if (rank, call) != (1, 1):
                _write_capture(tmp_path, rank=rank, call=call)

    with pytest.raises(FileNotFoundError, match="rank1_call01"):
        build_route_replay(
            tmp_path,
            capture_id="capture-1",
            source_git_sha="abc123",
            world_size=2,
            forward_calls=2,
            tokens_per_rank=2,
            topk=2,
            expert_count=4,
            hidden_size=128,
            ep_size=2,
        )


def test_capture_builder_rejects_corrupt_topk(tmp_path: Path) -> None:
    for rank in range(2):
        for call in range(2):
            _write_capture(
                tmp_path,
                rank=rank,
                call=call,
                corrupt_hash=(rank, call) == (0, 1),
            )

    with pytest.raises(ValueError, match="checksum mismatch.*rank0_call01"):
        build_route_replay(
            tmp_path,
            capture_id="capture-1",
            source_git_sha="abc123",
            world_size=2,
            forward_calls=2,
            tokens_per_rank=2,
            topk=2,
            expert_count=4,
            hidden_size=128,
            ep_size=2,
        )


def test_adapter_capture_is_opt_in_atomic_and_does_not_change_dispatch_return() -> None:
    for token in (
        "TILEXR_MINDSPEED_ROUTE_CAPTURE_DIR",
        "TILEXR_MINDSPEED_ROUTE_CAPTURE_ID",
        "TILEXR_MINDSPEED_ROUTE_CAPTURE_SKIP_CALLS",
        "TILEXR_MINDSPEED_ROUTE_CAPTURE_CALLS",
        "os.replace(temporary, target)",
        "payload_seed",
        "tensor_descriptors",
    ):
        assert token in ADAPTER
    assert "public_result = (hidden, route_weights, cu_seqlens, plan)" in ADAPTER
    assert "return (*public_result, event) if async_finish else public_result" in ADAPTER
