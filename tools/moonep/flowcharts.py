from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any, Iterable


FLOWCHART_STAGES = (
    ("1_planning", "planning", "Planning"),
    ("2_dispatch", "dispatch", "Dispatch"),
    ("3_prefetch-weight", "prefetch_weight", "PrefetchWeight"),
    ("4_expert-forward", "expert_forward", "ExpertForward"),
    ("5_combine", "combine", "Combine"),
    ("6_reduce-grad", "reduce_grad", "ReduceGrad"),
)

MERMAID_INIT = (
    '%%{init: {"theme":"base",'
    '"fontFamily":"Microsoft YaHei, Segoe UI, sans-serif",'
    '"themeVariables":{"fontFamily":"Microsoft YaHei, Segoe UI, sans-serif",'
    '"fontSize":"18px","primaryTextColor":"#17212b",'
    '"lineColor":"#66717d"},'
    '"flowchart":{"htmlLabels":true,"curve":"basis","nodeSpacing":60,'
    '"rankSpacing":75,"wrappingWidth":800,"diagramPadding":40}}}%%'
)

_DIMENSION_LINES = (
    ("world_size", "R", "逻辑 Rank 数"),
    ("tokens_per_rank", "S", "每个 Rank 的原始 token 数"),
    ("topk", "K", "每个 token 选择的 expert 数"),
    ("expert_count", "E", "全局 expert 数"),
    ("prefetch_slots", "B", "每个 Rank 的远端权重预取槽数"),
    ("token_padding", "P", "每个 expert 的 token 对齐粒度"),
    ("hidden_size", "H", "hidden 向量宽度"),
    ("intermediate_size", "I", "MLP 中间维度"),
)

_VISIBLE_FIELDS = {
    ("planning", "input"): ("topk_experts", "tokens_per_expert"),
    ("planning", "output"): (
        "plan.dst",
        "plan.cu_seqlens",
        "plan.experts_to_copy",
        "plan.zero_fill_ranges",
        "plan.remote_stats",
    ),
    ("dispatch", "input"): (
        "hidden",
        "route_weights",
        "plan.dst",
        "plan.cu_seqlens",
    ),
    ("dispatch", "output"): (
        "hidden",
        "route_weights",
        "plan.dst",
        "plan.cu_seqlens",
        "plan.dedup",
    ),
    ("prefetch_weight", "input"): (
        "plan.experts_to_copy",
        "projections",
    ),
    ("prefetch_weight", "output"): ("projections",),
    ("expert_forward", "input"): (
        "hidden",
        "cu_seqlens",
        "route_weights",
        "projections",
    ),
    ("expert_forward", "output"): ("hidden",),
    ("combine", "input"): (
        "expert_output",
        "route_weights",
        "plan.dst",
        "plan.dedup",
    ),
    ("combine", "output"): ("hidden", "route_weights"),
    ("reduce_grad", "input"): (
        "plan.experts_to_copy",
        "full_grads",
        "reduce_buffers",
    ),
    ("reduce_grad", "output"): ("full_grads", "reduce_buffers"),
}

_STAGE_SEMANTICS = {
    "planning": (
        "汇总各 Rank 的 tokens_per_expert，按 expert owner 负载规划迁移，"
        "生成目标 Rank/offset、group 边界与远端权重预取映射。",
        "检查 dst 编码、cu_seqlens、experts_to_copy、zero_fill_ranges 和 remote_stats。",
    ),
    "dispatch": (
        "按 plan.dst 解码目标 Rank 与 offset，跨 Rank 交换 hidden 和 route weight，"
        "并为同一 source token 的重复目标建立 dedup 元数据。",
        "检查目标 offset 的 hidden/weight 来源、有效前缀以及 dedup groups/loffs/counts。",
    ),
    "prefetch_weight": (
        "按 experts_to_copy 把远端 expert 的 gate/up/down 权重复制到本 Rank 的预取槽。",
        "检查原 expert 行保持不变、有效预取槽等于 owner 权重、未使用槽保持输入值。",
    ),
    "expert_forward": (
        "按 cu_seqlens 将 dispatch buffer 分组，执行 Gate GMM、Up GMM、SwiGLU、"
        "Down GMM，并应用逐 route 权重。",
        "检查输出 shape/dtype，输入 hidden、权重、group 边界和 projection 均未被改写。",
    ),
    "combine": (
        "按 plan.dst 将 expert 输出送回 source token，恢复重复 hidden 位置并聚合 top-k route。",
        "检查每个 source token 的聚合结果、恢复后的 route weight 以及输入保持不变。",
    ),
    "reduce_grad": (
        "按 experts_to_copy 将预取槽产生的 gate/up/down 梯度回收到原 expert owner，"
        "并清零当前 Rank 已消费的本地 live slot。",
        "检查梯度只加入一次且只加入 owner；非 owner 行、非本地行和未使用槽保持不变。",
    ),
}

_MAX_TENSOR_ELEMENTS = 32
_VALUES_PER_LINE = 8


def _load_snapshot(torch_module, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"missing reference tensor snapshot: {path}")
    return torch_module.load(path, map_location="cpu", weights_only=True)


def _walk_tensors(torch_module, value: Any, path: str = "") -> Iterable[tuple[str, Any]]:
    if torch_module.is_tensor(value):
        yield path or "value", value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "__type__":
                continue
            child_path = str(key) if not path else f"{path}.{key}"
            yield from _walk_tensors(torch_module, child, child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk_tensors(torch_module, child, f"{path}[{index}]")


def _get_mapping(value: Any, *keys: str) -> dict[str, Any]:
    current = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"snapshot is missing {'.'.join(keys)}")
        current = current[key]
    if not isinstance(current, dict):
        raise ValueError(f"snapshot field {'.'.join(keys)} is not a mapping")
    return current


def _matches(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in prefixes)


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _tensor_lines(path: str, tensor: Any) -> list[str]:
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    numel = int(flat.numel())
    shown = flat[:_MAX_TENSOR_ELEMENTS].tolist()
    lines = [
        f"{path} shape={list(tensor.shape)} dtype={tensor.dtype}",
    ]
    for start in range(0, len(shown), _VALUES_PER_LINE):
        chunk = ", ".join(
            _format_value(value) for value in shown[start : start + _VALUES_PER_LINE]
        )
        lines.append(f"values[{start}:{start + len(shown[start:start + _VALUES_PER_LINE])}]=[{chunk}]")
    if numel > len(shown):
        lines.append(f"仅展示前 {len(shown)} 个元素；完整值见同目录 .pt/.txt（共 {numel} 个）")
    return lines


def _snapshot_lines(torch_module, stage: str, direction: str, snapshot: Any) -> list[str]:
    prefixes = _VISIBLE_FIELDS[(stage, direction)]
    tensors = [
        (path, tensor)
        for path, tensor in _walk_tensors(torch_module, snapshot)
        if _matches(path, prefixes)
    ]
    if not tensors:
        raise ValueError(f"{stage} {direction} snapshot has no expected tensors")
    lines: list[str] = []
    for index, (path, tensor) in enumerate(tensors):
        if index:
            lines.append("")
        lines.extend(_tensor_lines(path, tensor))
    return lines


def _label(lines: Iterable[str]) -> str:
    return "<br/>".join(html.escape(line, quote=True) for line in lines)


def _dimension_label(dimensions: dict[str, Any], stage_title: str) -> str:
    lines = [f"0. {stage_title} 阶段参数"]
    for key, symbol, meaning in _DIMENSION_LINES:
        if key not in dimensions:
            raise ValueError(f"planning snapshot dimensions are missing {key}")
        lines.append(f"{symbol}={int(dimensions[key])}：{meaning}")
    route_count = int(dimensions["tokens_per_rank"]) * int(dimensions["topk"])
    experts_per_rank = int(dimensions["expert_count"]) // int(dimensions["world_size"])
    nvsh = route_count + (
        (int(dimensions["token_padding"]) - 1) * 2 * experts_per_rank
    )
    lines.extend(
        (
            f"S×K={route_count}：每个 Rank 的 route 数",
            f"NvS={nvsh}：每个 Rank 的 dispatch buffer 容量",
            "图中数值来自本次 reference 快照；candidate 仅参与正确性比较，不作为图源",
        )
    )
    return _label(lines)


def _render_source(
    torch_module,
    *,
    stage: str,
    stage_title: str,
    world_size: int,
    snapshots: dict[tuple[int, str], Any],
    dimensions: dict[str, Any],
) -> str:
    transform, check = _STAGE_SEMANTICS[stage]
    lines = [MERMAID_INIT, "flowchart LR"]
    lines.append(f'    PARAM["{_dimension_label(dimensions, stage_title)}"]')
    for rank in range(world_size):
        input_label = _label(
            [f"1.{rank + 1} Rank {rank} 输入"]
            + _snapshot_lines(torch_module, stage, "input", snapshots[(rank, "input")])
        )
        lines.append(f'    R{rank}I["{input_label}"]')
        lines.append(f"    PARAM --> R{rank}I")
    transform_label = _label(
        (
            f"2. {stage_title} 语义转换",
            transform,
            "多 Rank reference 使用 HCCL 完成所需交换；TileXR candidate 应产生相同语义结果。",
        )
    )
    lines.append(f'    TRANSFORM["{transform_label}"]')
    for rank in range(world_size):
        lines.append(f"    R{rank}I --> TRANSFORM")
    for rank in range(world_size):
        output_label = _label(
            [f"3.{rank + 1} Rank {rank} 输出"]
            + _snapshot_lines(torch_module, stage, "output", snapshots[(rank, "output")])
        )
        lines.append(f'    R{rank}O["{output_label}"]')
        lines.append(f"    TRANSFORM --> R{rank}O")
    lines.append(f'    CHECK["{_label(("4. 正确性检查点", check))}"]')
    for rank in range(world_size):
        lines.append(f"    R{rank}O --> CHECK")
    lines.extend(
        (
            "",
            "    classDef param fill:#fff8df,stroke:#a27617,stroke-width:2px,color:#17212b",
            "    classDef rank0 fill:#edf8f1,stroke:#31845a,stroke-width:2px,color:#17212b",
            "    classDef rank1 fill:#eef5ff,stroke:#3478b8,stroke-width:2px,color:#17212b",
            "    classDef shared fill:#f4f5f7,stroke:#5c6670,stroke-width:2px,color:#17212b",
            "    classDef output fill:#f5f0ff,stroke:#7654a8,stroke-width:2px,color:#17212b",
            "    class PARAM param",
            "    class TRANSFORM shared",
            "    class CHECK output",
        )
    )
    for rank in range(world_size):
        style = "rank0" if rank % 2 == 0 else "rank1"
        lines.append(f"    class R{rank}I,R{rank}O {style}")
    return "\n".join(lines) + "\n"


def generate_mermaid_sources(
    torch_module,
    *,
    case_dir: str | Path,
    world_size: int,
    output_dir: str | Path,
) -> list[Path]:
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    case_path = Path(case_dir).resolve()
    if not case_path.is_dir():
        raise FileNotFoundError(f"MoonEP case directory not found: {case_path}")
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    loaded: dict[tuple[str, int, str], Any] = {}
    for _, stage, _ in FLOWCHART_STAGES:
        for rank in range(world_size):
            for direction in ("input", "output"):
                snapshot_path = (
                    case_path
                    / f"rank_{rank}"
                    / "tensor_dumps"
                    / stage
                    / "reference"
                    / f"{direction}.pt"
                )
                loaded[(stage, rank, direction)] = _load_snapshot(
                    torch_module, snapshot_path
                )

    dimensions = _get_mapping(loaded[("planning", 0, "output")], "plan", "dimensions")
    paths = []
    for prefix, stage, stage_title in FLOWCHART_STAGES:
        source = _render_source(
            torch_module,
            stage=stage,
            stage_title=stage_title,
            world_size=world_size,
            snapshots={
                (rank, direction): loaded[(stage, rank, direction)]
                for rank in range(world_size)
                for direction in ("input", "output")
            },
            dimensions=dimensions,
        )
        path = output_path / f"{prefix}-{world_size}rank-detailed-lr.mmd"
        path.write_text(source, encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MoonEP six-stage Mermaid sources from reference snapshots"
    )
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    paths = generate_mermaid_sources(
        torch,
        case_dir=args.case_dir,
        world_size=args.world_size,
        output_dir=args.output_dir,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
