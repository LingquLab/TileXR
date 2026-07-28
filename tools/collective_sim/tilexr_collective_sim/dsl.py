from typing import Any, Dict, List


def allgather_direct_algorithm(rank_count: int, message_bytes: int) -> Dict[str, Any]:
    buffers: List[Dict[str, Any]] = []
    ops: List[Dict[str, Any]] = []

    for rank in range(rank_count):
        buffers.append({"id": f"r{rank}_in", "rank": rank, "role": "user_input", "chunks": [f"rank{rank}.chunk0"]})
        buffers.append({"id": f"r{rank}_out", "rank": rank, "role": "user_output", "chunks": []})
        for src in range(rank_count):
            buffers.append({"id": f"r{rank}_comm_from_{src}", "rank": rank, "role": "comm_buffer", "chunks": []})

    for rank in range(rank_count):
        ops.append({
            "id": f"copy_in_r{rank}",
            "type": "copy",
            "rank": rank,
            "bytes": 0,
            "src_buffer": f"r{rank}_in",
            "dst_buffer": f"r{rank}_comm_from_{rank}",
            "mode": "sdma",
        })

    for src in range(rank_count):
        for dst in range(rank_count):
            if src == dst:
                continue
            ops.append({
                "id": f"send_r{src}_to_r{dst}",
                "type": "send",
                "rank": src,
                "bytes": 0,
                "src_rank": src,
                "dst_rank": dst,
                "src_buffer": f"r{src}_comm_from_{src}",
                "dst_buffer": f"r{dst}_comm_from_{src}",
                "deps": [f"copy_in_r{src}"],
                "mode": "datacopy",
            })

    for rank in range(rank_count):
        for src in range(rank_count):
            dep = f"copy_in_r{rank}" if src == rank else f"send_r{src}_to_r{rank}"
            ops.append({
                "id": f"copy_out_r{rank}_from_{src}",
                "type": "copy",
                "rank": rank,
                "bytes": 0,
                "src_buffer": f"r{rank}_comm_from_{src}",
                "dst_buffer": f"r{rank}_out",
                "deps": [dep],
                "mode": "sdma",
            })

    return {
        "name": "direct_allgather",
        "collective": "allgather",
        "rank_count": rank_count,
        "buffers": buffers,
        "ops": ops,
        "metadata": {
            "chunk_bytes": message_bytes,
            "description": "direct all-to-all AllGather over communication buffers",
        },
    }
