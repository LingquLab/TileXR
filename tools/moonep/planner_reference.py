from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReferencePlan:
    dispatched_capacity: int
    dst: tuple[int, ...]
    cu_seqlens: tuple[int, ...]
    zero_fill_ranges: tuple[int, ...]
    experts_to_copy: tuple[int, ...]
    remote_stats: tuple[int, ...]


def deterministic_all_topk(
    rank_size: int, tokens_per_rank: int, topk: int, expert_count: int
) -> tuple[int, ...]:
    route_count = tokens_per_rank * topk
    return tuple(
        (route + source * topk) % expert_count
        for source in range(rank_size)
        for route in range(route_count)
    )


def _first_max(values: Sequence[int]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _first_min(values: Sequence[int]) -> int:
    return min(range(len(values)), key=lambda index: (values[index], index))


def build_reference_plan(
    *,
    rank: int,
    rank_size: int,
    tokens_per_rank: int,
    topk: int,
    expert_count: int,
    all_topk: Sequence[int],
    prefetch_slots: int | None = None,
    token_padding: int = 1,
) -> ReferencePlan:
    if (
        rank_size <= 0
        or rank < 0
        or rank >= rank_size
        or tokens_per_rank <= 0
        or topk <= 0
        or expert_count <= 0
        or expert_count % rank_size != 0
    ):
        raise ValueError("invalid Planner reference dimensions")
    route_count = tokens_per_rank * topk
    experts_per_rank = expert_count // rank_size
    slots = experts_per_rank if prefetch_slots is None else int(prefetch_slots)
    padding = int(token_padding)
    if slots <= 0 or slots > experts_per_rank:
        raise ValueError("prefetch_slots must be in [1, E/R]")
    if padding <= 0:
        raise ValueError("token_padding must be positive")
    dispatched_capacity = route_count + (padding - 1) * 2 * experts_per_rank
    if len(all_topk) != rank_size * route_count:
        raise ValueError("all_topk size does not match rank_size * S * K")
    if rank_size * dispatched_capacity > 2**31:
        raise OverflowError("encoded destination range exceeds signed int32")

    tpe = [[0 for _ in range(expert_count)] for _ in range(rank_size)]
    for source in range(rank_size):
        for route in range(route_count):
            expert = int(all_topk[source * route_count + route])
            if expert < 0 or expert >= expert_count:
                raise ValueError(f"expert id {expert} is outside [0, {expert_count})")
            tpe[source][expert] += 1

    tpe_prefix = [row[:] for row in tpe]
    global_counts = [0 for _ in range(expert_count)]
    for expert in range(expert_count):
        cumulative = 0
        for source in range(rank_size):
            cumulative += tpe[source][expert]
            tpe_prefix[source][expert] = cumulative
        global_counts[expert] = cumulative

    balance = []
    for home in range(rank_size):
        begin = home * experts_per_rank
        balance.append(
            sum(global_counts[begin : begin + experts_per_rank]) - route_count
        )
    moves = [[0 for _ in range(rank_size)] for _ in range(rank_size)]
    while True:
        surplus = _first_max(balance)
        deficit = _first_min(balance)
        if balance[surplus] <= 0 or balance[deficit] >= 0:
            break
        count = -balance[deficit]
        moves[surplus][deficit] = count
        balance[surplus] -= count
        balance[deficit] = 0

    allocation = [[0 for _ in range(rank_size)] for _ in range(expert_count)]
    for expert, count in enumerate(global_counts):
        allocation[expert][expert // experts_per_rank] = count
    for home in range(rank_size):
        begin = home * experts_per_rank
        remaining = global_counts[begin : begin + experts_per_rank]
        quotas = moves[home][:]
        while True:
            destination = _first_max(quotas)
            if quotas[destination] <= 0:
                break
            local_expert = _first_max(remaining)
            if remaining[local_expert] <= 0:
                raise RuntimeError("allocation exhausted before migration quota")
            count = min(remaining[local_expert], quotas[destination])
            expert = begin + local_expert
            allocation[expert][destination] += count
            allocation[expert][home] -= count
            remaining[local_expert] -= count
            quotas[destination] -= count

    expert_offsets = [
        [0 for _ in range(expert_count)] for _ in range(rank_size)
    ]
    all_cu = [
        [0 for _ in range(expert_count + slots)]
        for _ in range(rank_size)
    ]
    all_zero_fill = [
        [[0, 0] for _ in range(expert_count + slots)]
        for _ in range(rank_size)
    ]
    experts_to_copy = [
        [-1 for _ in range(slots)] for _ in range(rank_size)
    ]
    all_remote_stats = [[0, 0] for _ in range(rank_size)]
    for destination in range(rank_size):
        local_begin = destination * experts_per_rank
        local_end = local_begin + experts_per_rank
        remote = [
            expert
            for expert in range(expert_count)
            if not local_begin <= expert < local_end
            and allocation[expert][destination] > 0
        ]
        remote.sort(
            key=lambda expert: (allocation[expert][destination], expert),
            reverse=True,
        )
        all_remote_stats[destination][0] = len(remote)
        selected = [False for _ in range(expert_count)]
        for slot, expert in enumerate(remote[:slots]):
            experts_to_copy[destination][slot] = expert
            selected[expert] = True
            all_remote_stats[expert // experts_per_rank][1] += 1

        end = 0
        for group in range(expert_count + slots):
            count = 0
            expert = -1
            if group < expert_count:
                if not selected[group]:
                    expert = group
                    count = allocation[expert][destination]
            else:
                expert = experts_to_copy[destination][group - expert_count]
                if expert >= 0:
                    count = allocation[expert][destination]
            if count > 0:
                expert_offsets[destination][expert] = end
            padded = ((count + padding - 1) // padding) * padding if count > 0 else 0
            if padded > count:
                all_zero_fill[destination][group] = [end + count, padded - count]
            end += padded
            all_cu[destination][group] = end
        if end > dispatched_capacity:
            raise RuntimeError("padded layout exceeds NvS")

    allocation_prefix = [row[:] for row in allocation]
    for expert in range(expert_count):
        cumulative = 0
        for destination in range(rank_size):
            cumulative += allocation[expert][destination]
            allocation_prefix[expert][destination] = cumulative

    local_counts = [0 for _ in range(expert_count)]
    dst = []
    for token in range(tokens_per_rank):
        seen_destinations: set[int] = set()
        for topk_index in range(topk):
            route = token * topk + topk_index
            expert = int(all_topk[rank * route_count + route])
            occurrence = local_counts[expert]
            local_counts[expert] += 1
            source_prefix = 0 if rank == 0 else tpe_prefix[rank - 1][expert]
            global_occurrence = source_prefix + occurrence
            destination = 0
            while (
                destination < rank_size
                and allocation_prefix[expert][destination] <= global_occurrence
            ):
                destination += 1
            if destination >= rank_size:
                raise RuntimeError("destination not found")
            previous = (
                0
                if destination == 0
                else allocation_prefix[expert][destination - 1]
            )
            raw = (
                destination * dispatched_capacity
                + expert_offsets[destination][expert]
                + global_occurrence
                - previous
            )
            duplicate = destination in seen_destinations
            seen_destinations.add(destination)
            dst.append(-raw - 1 if duplicate else raw)

    return ReferencePlan(
        dispatched_capacity=dispatched_capacity,
        dst=tuple(dst),
        cu_seqlens=tuple(all_cu[rank]),
        zero_fill_ranges=tuple(
            value for group in all_zero_fill[rank] for value in group
        ),
        experts_to_copy=tuple(
            expert for row in experts_to_copy for expert in row
        ),
        remote_stats=tuple(all_remote_stats[rank]),
    )
