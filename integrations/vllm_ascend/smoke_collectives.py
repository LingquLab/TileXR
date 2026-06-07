#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import torch_npu  # noqa: F401

from tilexr_collectives import all_gather, all_to_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TileXR vllm-ascend collectives smoke test")
    parser.add_argument("--rank-size", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--first-npu", type=int, default=0)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--dtype", choices=["int32", "fp16"], default="int32")
    parser.add_argument("--op", choices=["allgather", "alltoall"], default="allgather")
    parser.add_argument("--install-prefix", required=True)
    return parser.parse_args()


def dtype_from_name(name: str):
    if name == "int32":
        return torch.int32
    if name == "fp16":
        return torch.float16
    raise ValueError(f"unsupported dtype name: {name}")


def make_values(rank_size: int, rank: int, count: int, dtype):
    base = rank * 1000
    values = torch.arange(base, base + count, dtype=torch.int32, device=f"npu:{torch.npu.current_device()}")
    if dtype is torch.float16:
        values = values.to(torch.float16)
    return values


def check_all_gather(result, rank_size: int, count: int, dtype) -> None:
    cpu = result.cpu()
    for src in range(rank_size):
        expected = torch.arange(src * 1000, src * 1000 + count, dtype=torch.int32)
        if dtype is torch.float16:
            expected = expected.to(torch.float16)
        actual = cpu[src].reshape(-1)
        if not torch.equal(actual, expected):
            raise AssertionError(f"AllGather mismatch for src={src}: actual={actual} expected={expected}")


def make_all_to_all_values(rank_size: int, rank: int, count: int, dtype):
    host = []
    for dst in range(rank_size):
        for idx in range(count):
            host.append(rank * 100000 + dst * 1000 + idx)
    values = torch.tensor(host, dtype=torch.int32, device=f"npu:{torch.npu.current_device()}")
    if dtype is torch.float16:
        values = values.to(torch.float16)
    return values


def check_all_to_all(result, rank_size: int, rank: int, count: int, dtype) -> None:
    cpu = result.cpu().reshape(-1)
    expected_values = []
    for src in range(rank_size):
        for idx in range(count):
            expected_values.append(src * 100000 + rank * 1000 + idx)
    expected = torch.tensor(expected_values, dtype=torch.int32)
    if dtype is torch.float16:
        expected = expected.to(torch.float16)
    if not torch.equal(cpu, expected):
        raise AssertionError(f"AllToAll mismatch on rank={rank}: actual={cpu} expected={expected}")


def main() -> None:
    args = parse_args()
    if args.rank_size <= 0:
        raise ValueError("--rank-size must be positive")
    if args.rank < 0 or args.rank >= args.rank_size:
        raise ValueError("--rank must be in [0, rank_size)")
    if args.count <= 0:
        raise ValueError("--count must be positive")

    device_id = args.first_npu + args.rank
    torch.npu.set_device(device_id)
    dtype = dtype_from_name(args.dtype)

    if args.op == "allgather":
        send = make_values(args.rank_size, args.rank, args.count, dtype).contiguous()
        result = all_gather(send, rank=args.rank, world_size=args.rank_size, install_prefix=args.install_prefix)
        torch.npu.synchronize()
        check_all_gather(result, args.rank_size, args.count, dtype)
    else:
        send = make_all_to_all_values(args.rank_size, args.rank, args.count, dtype).contiguous()
        result = all_to_all(send, rank=args.rank, world_size=args.rank_size, install_prefix=args.install_prefix)
        torch.npu.synchronize()
        check_all_to_all(result, args.rank_size, args.rank, args.count, dtype)

    print(f"PASS rank={args.rank} op={args.op} dtype={args.dtype} count={args.count}")


if __name__ == "__main__":
    main()
