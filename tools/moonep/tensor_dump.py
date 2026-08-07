from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Callable

from .report import write_json


PreviewSink = Callable[[str], None]


class StageTensorDumper:
    """Persist untimed correctness-stage values and compact tensor previews."""

    def __init__(
        self,
        torch_module,
        output_dir: str | Path,
        *,
        preview_elements: int = 8,
        preview_sink: PreviewSink | None = None,
    ) -> None:
        if preview_elements <= 0:
            raise ValueError("preview_elements must be positive")
        self.torch = torch_module
        self.output_dir = Path(output_dir)
        self.preview_elements = int(preview_elements)
        self.preview_sink = preview_sink
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_log = self.output_dir / "preview.log"
        self.preview_log.write_text("", encoding="utf-8")

    def dump(
        self,
        stage: str,
        backend_role: str,
        backend_name: str,
        direction: str,
        value: Any,
    ) -> None:
        if direction not in ("input", "output"):
            raise ValueError("direction must be input or output")
        target_dir = self.output_dir / stage / backend_role
        target_dir.mkdir(parents=True, exist_ok=True)

        tensors: list[dict[str, Any]] = []
        values: dict[str, Any] = {}
        readable_tensors: list[tuple[dict[str, Any], Any]] = []
        snapshot = self._snapshot(value, "", tensors, values, readable_tensors)
        snapshot_path = target_dir / f"{direction}.pt"
        readable_path = target_dir / f"{direction}.txt"
        self.torch.save(snapshot, snapshot_path)
        self._write_human_readable(
            readable_path,
            stage=stage,
            backend_role=backend_role,
            backend_name=backend_name,
            direction=direction,
            tensors=readable_tensors,
            values=values,
        )
        write_json(
            target_dir / f"{direction}.json",
            {
                "schema_version": 1,
                "stage": stage,
                "backend_role": backend_role,
                "backend_name": backend_name,
                "direction": direction,
                "snapshot": snapshot_path.name,
                "human_readable": readable_path.name,
                "preview_elements": self.preview_elements,
                "tensors": tensors,
                "values": values,
            },
        )
        self._emit_previews(stage, backend_role, direction, tensors)

    def _snapshot(self, value, path, tensors, values, readable_tensors):
        if self.torch.is_tensor(value):
            cpu = value.detach().cpu().contiguous().clone()
            flat = cpu.reshape(-1)
            preview = flat[: self.preview_elements].tolist()
            record = {
                "path": path or "value",
                "shape": [int(item) for item in value.shape],
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": int(value.numel()),
                "preview": preview,
                "preview_truncated": int(value.numel()) > self.preview_elements,
            }
            tensors.append(record)
            readable_tensors.append((record, cpu))
            return cpu
        if is_dataclass(value) and not isinstance(value, type):
            result = {"__type__": type(value).__name__}
            for field in fields(value):
                child_path = self._child_path(path, field.name)
                result[field.name] = self._snapshot(
                    getattr(value, field.name),
                    child_path,
                    tensors,
                    values,
                    readable_tensors,
                )
            return result
        if isinstance(value, dict):
            return {
                str(key): self._snapshot(
                    item,
                    self._child_path(path, str(key)),
                    tensors,
                    values,
                    readable_tensors,
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._snapshot(
                    item,
                    f"{path}[{index}]",
                    tensors,
                    values,
                    readable_tensors,
                )
                for index, item in enumerate(value)
            ]
        if value is None or isinstance(value, (str, int, float, bool)):
            values[path or "value"] = value
            return value
        raise TypeError(f"cannot dump value of type {type(value).__name__} at {path}")

    @staticmethod
    def _child_path(parent: str, child: str) -> str:
        return child if not parent else f"{parent}.{child}"

    @staticmethod
    def _write_human_readable(
        path: Path,
        *,
        stage: str,
        backend_role: str,
        backend_name: str,
        direction: str,
        tensors,
        values,
    ) -> None:
        lines = [
            "# MoonEP complete tensor snapshot",
            f"stage: {stage}",
            f"backend_role: {backend_role}",
            f"backend_name: {backend_name}",
            f"direction: {direction}",
        ]
        for record, tensor in tensors:
            lines.extend(
                (
                    "",
                    f"[tensor] {record['path']}",
                    f"shape: {record['shape']}",
                    f"dtype: {record['dtype']}",
                    f"device: {record['device']}",
                    f"numel: {record['numel']}",
                    "values:",
                    pformat(tensor.tolist(), width=120, sort_dicts=False),
                )
            )
        for value_path, scalar in values.items():
            lines.extend(("", f"[value] {value_path} = {scalar!r}"))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _emit_previews(self, stage, backend_role, direction, tensors) -> None:
        lines = [
            f"[tensor-dump] {stage}/{backend_role}/{direction} {tensor['path']} "
            f"shape={tensor['shape']} dtype={tensor['dtype']} "
            f"preview={tensor['preview']}"
            for tensor in tensors
        ]
        if not lines:
            return
        with self.preview_log.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
        if self.preview_sink is not None:
            for line in lines:
                self.preview_sink(line)
