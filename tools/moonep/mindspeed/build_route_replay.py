from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import zlib
from pathlib import Path

from tools.moonep.model_replay_cache import ReplayShape, validate_route_replay


def build_route_replay(
    capture_dir: str | Path,
    *,
    capture_id: str,
    source_git_sha: str,
    world_size: int,
    forward_calls: int,
    tokens_per_rank: int,
    topk: int,
    expert_count: int,
    hidden_size: int,
    ep_size: int,
    ffn_hidden_size: int = 2048,
    token_padding: int = 1,
) -> dict[str, object]:
    shape = ReplayShape(
        tokens_per_rank=tokens_per_rank,
        topk=topk,
        hidden_size=hidden_size,
        ep_size=ep_size,
        world_size=world_size,
        expert_count=expert_count,
        ffn_hidden_size=ffn_hidden_size,
        token_padding=token_padding,
        forward_calls=forward_calls,
    )
    root = Path(capture_dir)
    ranks: dict[str, list[dict[str, object]]] = {}
    topk_hashes: dict[str, list[str]] = {}
    for rank in range(world_size):
        records = []
        hashes = []
        for call in range(forward_calls):
            path = root / f"rank{rank}_call{call:02d}.json"
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            if (
                not isinstance(record, dict)
                or record.get("schema_version") != 1
                or record.get("complete") is not True
                or int(record.get("rank", -1)) != rank
                or int(record.get("call", -1)) != call
                or record.get("topk_shape") != [tokens_per_rank, topk]
                or record.get("topk_dtype") != "torch.int32"
                or record.get("capture_id") != capture_id
            ):
                raise ValueError(f"invalid route capture record: {path}")
            if record.get("topk_encoding") != "int32-le-zlib-base64":
                raise ValueError(f"invalid route encoding: {path}")
            try:
                topk_raw = zlib.decompress(
                    base64.b64decode(record["topk_zlib_base64"], validate=True)
                )
            except (KeyError, TypeError, ValueError, zlib.error) as exc:
                raise ValueError(f"invalid compressed route payload: {path}") from exc
            if (
                len(topk_raw) != tokens_per_rank * topk * 4
                or hashlib.sha256(topk_raw).hexdigest()
                != record.get("topk_sha256")
            ):
                raise ValueError(f"route payload checksum mismatch: {path}")
            try:
                payload_seed = int(record["payload_seed"])
                tensor_descriptors = dict(record["tensor_descriptors"])
                remote_stats = [int(value) for value in record["remote_stats"]]
                experts_to_copy = [
                    [int(value) for value in row]
                    for row in record["experts_to_copy"]
                ]
                counts = [int(value) for value in record["tokens_per_expert"]]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid route capture metadata: {path}") from exc
            records.append(
                {
                    "call": call,
                    "source_call": int(record.get("source_call", call)),
                    "topk_shape": list(record["topk_shape"]),
                    "topk_dtype": record["topk_dtype"],
                    "topk_encoding": record["topk_encoding"],
                    "topk_zlib_base64": record["topk_zlib_base64"],
                    "topk_sha256": record["topk_sha256"],
                    "tokens_per_expert": counts,
                    "remote_stats": remote_stats,
                    "experts_to_copy": experts_to_copy,
                    "payload_seed": payload_seed,
                    "tensor_descriptors": tensor_descriptors,
                }
            )
            hashes.append(str(record["topk_sha256"]))
        ranks[str(rank)] = records
        topk_hashes[str(rank)] = hashes
    replay: dict[str, object] = {
        "schema_version": 1,
        "dimensions": {
            "world_size": world_size,
            "tokens_per_rank": tokens_per_rank,
            "topk": topk,
            "expert_count": expert_count,
            "forward_calls": forward_calls,
        },
        "payload_contract": {
            "captured": "topk,plan_metadata,tensor_descriptors",
            "regenerated": "payload_values_from_payload_seed",
        },
        "provenance": {
            "capture_id": capture_id,
            "source_git_sha": source_git_sha,
            "source": "TileXR MindSpeed model route and plan capture",
            "topk_sha256_by_rank": topk_hashes,
        },
        "ranks": ranks,
    }
    validate_route_replay(replay, shape)
    return replay


def write_route_replay(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a model-flow route replay from a MindSpeed capture"
    )
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--forward-calls", type=int, default=10)
    parser.add_argument("--tokens-per-rank", type=int, required=True)
    parser.add_argument("--topk", type=int, required=True)
    parser.add_argument("--expert-count", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--ep-size", type=int, required=True)
    parser.add_argument("--ffn-hidden-size", type=int, default=2048)
    parser.add_argument("--token-padding", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    replay = build_route_replay(
        args.capture_dir,
        capture_id=args.capture_id,
        source_git_sha=args.source_git_sha,
        world_size=args.world_size,
        forward_calls=args.forward_calls,
        tokens_per_rank=args.tokens_per_rank,
        topk=args.topk,
        expert_count=args.expert_count,
        hidden_size=args.hidden_size,
        ep_size=args.ep_size,
        ffn_hidden_size=args.ffn_hidden_size,
        token_padding=args.token_padding,
    )
    write_route_replay(args.output, replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
