from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ComparisonRow:
    run_name: str
    run_dir: Path
    model_name: str
    epoch: int
    metric_value: float
    device: str


@dataclass(frozen=True)
class ComparisonReport:
    metric: str
    manifest_identity: str
    rows: tuple[ComparisonRow, ...]
    config_differences: dict[str, tuple[object, ...]]


@dataclass(frozen=True)
class _LoadedRun:
    row: ComparisonRow
    manifest_identity: str
    flattened_config: dict[str, object]


def compare_runs(run_dirs: Sequence[Path], *, metric: str) -> ComparisonReport:
    if not run_dirs:
        raise ValueError("comparison requires at least one run directory")
    if not metric.strip():
        raise ValueError("comparison metric must not be empty")
    lower_is_better = "loss" in metric.lower()
    loaded = tuple(_load_run(Path(path), metric, lower_is_better=lower_is_better) for path in run_dirs)
    identities = {run.manifest_identity for run in loaded}
    if len(identities) != 1:
        raise ValueError("run manifest identities differ; compare runs prepared from the same data")

    ranked = sorted(
        loaded,
        key=lambda run: (
            run.row.metric_value if lower_is_better else -run.row.metric_value,
            run.row.run_name,
        ),
    )
    rows = tuple(run.row for run in ranked)
    differences = _config_differences(tuple(run.flattened_config for run in ranked))
    return ComparisonReport(metric, ranked[0].manifest_identity, rows, differences)


def format_comparison(report: ComparisonReport) -> str:
    lines = [
        f"metric: {report.metric}",
        f"manifest_identity: {report.manifest_identity}",
        "run\tmodel\tepoch\tvalue\tdevice",
    ]
    lines.extend(
        f"{row.run_name}\t{row.model_name}\t{row.epoch}\t{row.metric_value:.6g}\t{row.device}" for row in report.rows
    )
    lines.append("config differences:")
    if not report.config_differences:
        lines.append("  none")
    else:
        for key, values in report.config_differences.items():
            rendered = " | ".join(
                f"{row.run_name}={json.dumps(value, sort_keys=True)}"
                for row, value in zip(report.rows, values, strict=True)
            )
            lines.append(f"  {key}: {rendered}")
    return "\n".join(lines) + "\n"


def write_comparison_csv(report: ComparisonReport, output: Path) -> Path:
    if output.exists():
        raise FileExistsError(f"comparison output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_name",
        "run_dir",
        "model_name",
        "epoch",
        "metric",
        "metric_value",
        "device",
        "manifest_identity",
    ]
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for row in report.rows:
                writer.writerow(
                    {
                        "run_name": row.run_name,
                        "run_dir": row.run_dir.as_posix(),
                        "model_name": row.model_name,
                        "epoch": row.epoch,
                        "metric": report.metric,
                        "metric_value": row.metric_value,
                        "device": row.device,
                        "manifest_identity": report.manifest_identity,
                    }
                )
        try:
            os.link(temporary, output)
        except FileExistsError:
            raise FileExistsError(f"comparison output already exists: {output}") from None
        return output
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)


def _load_run(run_dir: Path, metric: str, *, lower_is_better: bool) -> _LoadedRun:
    config = _load_yaml_mapping(run_dir / "config.yaml")
    metadata = _load_yaml_mapping(run_dir / "run.yaml")
    identity = metadata.get("manifest_identity")
    if not isinstance(identity, str) or not identity:
        raise ValueError(f"{run_dir / 'run.yaml'}: manifest_identity must be a nonempty string")
    metric_rows = _read_metric_rows(run_dir / "metrics.csv", metric)
    best_epoch, best_value = (
        min(metric_rows, key=lambda item: item[1]) if lower_is_better else max(metric_rows, key=lambda item: item[1])
    )
    model = config.get("model")
    model_name = model.get("name") if isinstance(model, Mapping) else None
    if not isinstance(model_name, str) or not model_name:
        raise ValueError(f"{run_dir / 'config.yaml'}: model.name must be a nonempty string")
    configured_name = config.get("run_name")
    run_name = configured_name if isinstance(configured_name, str) and configured_name else run_dir.name
    device = metadata.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError(f"{run_dir / 'run.yaml'}: device must be a nonempty string")
    return _LoadedRun(
        ComparisonRow(run_name, run_dir, model_name, best_epoch, best_value, device),
        identity,
        _flatten_config(config),
    )


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read run artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    return {str(key): item for key, item in value.items()}


def _read_metric_rows(path: Path, metric: str) -> tuple[tuple[int, float], ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or metric not in reader.fieldnames:
                raise ValueError(f"{path}: missing metric column {metric!r}")
            if "epoch" not in reader.fieldnames:
                raise ValueError(f"{path}: missing epoch column")
            rows: list[tuple[int, float]] = []
            for line_number, raw in enumerate(reader, start=2):
                try:
                    epoch = int(raw["epoch"])
                    value = float(raw[metric])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: invalid epoch or {metric}") from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{line_number}: metric {metric!r} must be finite")
                rows.append((epoch, value))
    except OSError as exc:
        raise ValueError(f"cannot read run artifact {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: metrics file is empty")
    return tuple(rows)


def _flatten_config(config: Mapping[str, object]) -> dict[str, object]:
    ignored = {"run_name", "output_dir", "device", "data.num_workers"}
    flattened: dict[str, object] = {}

    def visit(prefix: str, value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                visit(path, item)
        elif prefix not in ignored:
            flattened[prefix] = value

    visit("", config)
    return flattened


def _config_differences(configs: tuple[dict[str, object], ...]) -> dict[str, tuple[object, ...]]:
    keys = sorted({key for config in configs for key in config})
    differences: dict[str, tuple[object, ...]] = {}
    for key in keys:
        values = tuple(config.get(key) for config in configs)
        if any(value != values[0] for value in values[1:]):
            differences[key] = values
    return differences
