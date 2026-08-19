from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when configuration input is invalid."""


@dataclass(frozen=True)
class DataConfig:
    name: str = "voc2007"
    data_dir: Path = Path("data/raw")
    manifest_dir: Path = Path("data/manifests")
    num_workers: int = 0
    horizontal_flip: float = 0.5
    max_train_samples: int | None = None
    max_valid_samples: int | None = None
    max_test_samples: int | None = None


@dataclass(frozen=True)
class ModelConfig:
    name: str = "fasterrcnn_mobilenet_v3_large_320_fpn"
    weights: str = "none"
    expected_num_classes: int = 21
    params: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 2
    batch_size: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    optimizer: str = "sgd"
    scheduler: str = "none"
    seed: int = 42
    amp: bool = False
    grad_clip: float = 0.0
    best_metric: str = "map_50_95"


@dataclass(frozen=True)
class EvaluationConfig:
    score_threshold: float = 0.05
    error_score_threshold: float = 0.5
    error_iou_threshold: float = 0.5
    max_detections: int = 100


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    device: str = "auto"
    output_dir: Path = Path("artifacts")
    run_name: str | None = None


def load_config(path: Path | None = None, overrides: Sequence[tuple[str, str]] = ()) -> AppConfig:
    values = asdict(AppConfig())
    if path is not None:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            raise ConfigError("configuration root must be a mapping")
        _merge_known(values, loaded)

    for key, raw_value in overrides:
        if not key:
            raise ConfigError("override key must not be empty")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid override {key}: {exc}") from exc
        _set_known(values, key, value)

    config = _construct_config(values)
    _validate_config(config)
    return config


def load_config_with_sources(
    path: Path | None = None,
    overrides: Sequence[tuple[str, str]] = (),
) -> tuple[AppConfig, dict[str, str]]:
    config = load_config(path, overrides)
    sources = {key: "default" for key in _leaf_paths(asdict(AppConfig()))}
    if path is not None:
        try:
            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
        if isinstance(loaded, Mapping):
            for key in _leaf_paths(loaded):
                sources[key] = "yaml"
    for key, _ in overrides:
        sources[key] = "cli"
    return config, dict(sorted(sources.items()))


def config_from_dict(values: Mapping[str, object]) -> AppConfig:
    merged = asdict(AppConfig())
    _merge_known(merged, values)
    config = _construct_config(merged)
    _validate_config(config)
    return config


def config_to_dict(config: AppConfig) -> dict[str, object]:
    return _serialize(asdict(config))


def _leaf_paths(values: Mapping[Any, object], prefix: str = "") -> tuple[str, ...]:
    paths: list[str] = []
    for raw_key, value in values.items():
        key = str(raw_key)
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping) and value:
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return tuple(paths)


def _merge_known(target: dict[str, Any], incoming: Mapping[Any, object], prefix: str = "") -> None:
    for raw_key, value in incoming.items():
        if not isinstance(raw_key, str):
            raise ConfigError(f"{prefix or 'configuration'} keys must be strings")
        path = f"{prefix}.{raw_key}" if prefix else raw_key
        if raw_key not in target:
            raise ConfigError(f"unknown configuration field: {path}")
        current = target[raw_key]
        if isinstance(current, dict) and raw_key != "params":
            if not isinstance(value, Mapping):
                raise ConfigError(f"{path} must be a mapping")
            _merge_known(current, value, path)
        else:
            target[raw_key] = value


def _set_known(values: dict[str, Any], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    current: dict[str, Any] = values
    for index, part in enumerate(parts):
        path = ".".join(parts[: index + 1])
        if part not in current:
            parent_path = ".".join(parts[:index])
            if parent_path == "model.params" and index == len(parts) - 1 and part:
                current[part] = value
                return
            raise ConfigError(f"unknown configuration field: {path}")
        if index == len(parts) - 1:
            current[part] = value
            return
        child = current[part]
        if not isinstance(child, dict):
            raise ConfigError(f"{path} is not a configuration section")
        current = child


def _construct_config(values: dict[str, Any]) -> AppConfig:
    try:
        data_values = dict(values["data"])
        data_values["data_dir"] = Path(data_values["data_dir"])
        data_values["manifest_dir"] = Path(data_values["manifest_dir"])
        return AppConfig(
            data=DataConfig(**data_values),
            model=ModelConfig(**dict(values["model"])),
            train=TrainConfig(**dict(values["train"])),
            evaluation=EvaluationConfig(**dict(values["evaluation"])),
            device=values["device"],
            output_dir=Path(values["output_dir"]),
            run_name=values["run_name"],
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid configuration value: {exc}") from exc


def _validate_config(config: AppConfig) -> None:
    _require_choice("data.name", config.data.name, {"voc2007"})
    _require_integer("data.num_workers", config.data.num_workers, minimum=0)
    _require_probability("data.horizontal_flip", config.data.horizontal_flip)
    for field_name, value in (
        ("max_train_samples", config.data.max_train_samples),
        ("max_valid_samples", config.data.max_valid_samples),
        ("max_test_samples", config.data.max_test_samples),
    ):
        if value is not None:
            _require_integer(f"data.{field_name}", value, minimum=1)

    _require_nonempty_string("model.name", config.model.name)
    if config.model.weights not in {"none", "imagenet1k_v1"}:
        raise ConfigError("model.weights must be 'none' or 'imagenet1k_v1'")
    _require_integer("model.expected_num_classes", config.model.expected_num_classes, minimum=2)
    if not isinstance(config.model.params, dict):
        raise ConfigError("model.params must be a mapping")

    _require_integer("train.epochs", config.train.epochs, minimum=1)
    _require_integer("train.batch_size", config.train.batch_size, minimum=1)
    _require_number("train.lr", config.train.lr, minimum=0.0, exclusive=True)
    _require_number("train.momentum", config.train.momentum, minimum=0.0)
    _require_number("train.weight_decay", config.train.weight_decay, minimum=0.0)
    _require_number("train.grad_clip", config.train.grad_clip, minimum=0.0)
    _require_integer("train.seed", config.train.seed, minimum=0, maximum=2**32 - 1)
    _require_type("train.amp", config.train.amp, bool)
    _require_choice("train.optimizer", config.train.optimizer, {"adamw", "sgd"})
    _require_choice("train.scheduler", config.train.scheduler, {"none", "step"})
    if config.train.best_metric != "map_50_95":
        raise ConfigError("train.best_metric currently supports only 'map_50_95'")

    _require_probability("evaluation.score_threshold", config.evaluation.score_threshold)
    _require_probability("evaluation.error_score_threshold", config.evaluation.error_score_threshold)
    _require_probability("evaluation.error_iou_threshold", config.evaluation.error_iou_threshold)
    _require_integer("evaluation.max_detections", config.evaluation.max_detections, minimum=1)
    if config.evaluation.max_detections != 100:
        raise ConfigError("evaluation.max_detections currently supports only 100")
    _require_nonempty_string("device", config.device)
    if config.run_name is not None:
        _require_nonempty_string("run_name", config.run_name)


def _require_type(path: str, value: object, expected: type[object]) -> None:
    if not isinstance(value, expected):
        raise ConfigError(f"{path} must be {expected.__name__}")


def _require_integer(
    path: str,
    value: object,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path} must be an integer")
    if minimum is not None and maximum is not None and not minimum <= value <= maximum:
        raise ConfigError(f"{path} must be between {minimum} and {maximum}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{path} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{path} must be at most {maximum}")


def _require_number(path: str, value: object, minimum: float, *, exclusive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path} must be a number")
    if not math.isfinite(value):
        raise ConfigError(f"{path} must be finite")
    if (exclusive and value <= minimum) or (not exclusive and value < minimum):
        comparison = "greater than" if exclusive else "at least"
        raise ConfigError(f"{path} must be {comparison} {minimum}")


def _require_probability(path: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{path} must be a number")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ConfigError(f"{path} must be between 0.0 and 1.0")


def _require_nonempty_string(path: str, value: object) -> None:
    if not isinstance(value, str):
        raise ConfigError(f"{path} must be str")
    if not value.strip():
        raise ConfigError(f"{path} must not be empty")


def _require_choice(path: str, value: object, choices: set[str]) -> None:
    _require_nonempty_string(path, value)
    if value not in choices:
        expected = " or ".join(repr(choice) for choice in sorted(choices))
        raise ConfigError(f"{path} must be {expected}")


def _serialize(value: object) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    return value
