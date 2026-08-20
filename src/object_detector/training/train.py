from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from object_detector.config import AppConfig, config_to_dict
from object_detector.data.dataset import VocDetectionDataset, build_detection_transforms, detection_collate
from object_detector.data.manifest import DatasetMetadata, load_dataset_metadata, verify_prepared_data
from object_detector.evaluation.metrics import DetectionMetric
from object_detector.models.registry import build_model
from object_detector.preflight import resolve_device, validate_training_request
from object_detector.training.checkpoint import (
    CheckpointCompatibilityError,
    ResumeIdentity,
    build_run_metadata,
    load_checkpoint,
    save_checkpoint,
    validate_resume_identity,
)
from object_detector.training.trainer import DryRunResult, dry_run, move_batch, train_one_epoch

ModelFactory = Callable[[str, int, str, Mapping[str, object]], nn.Module]
MetricFactory = Callable[[Sequence[str]], DetectionMetric]


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    completed_epochs: int
    dry_run_result: DryRunResult | None = None


def run_training(
    config: AppConfig,
    *,
    resume: Path | None = None,
    dry_run_mode: bool = False,
    model_factory: ModelFactory = build_model,
    metric_factory: MetricFactory = DetectionMetric,
) -> RunResult:
    if dry_run_mode and resume is not None:
        raise ValueError("--dry-run cannot be combined with --resume")
    run_name = config.run_name or "run"
    run_dir = config.output_dir / run_name
    in_place_resume_from_best = False
    if resume is not None and run_dir.resolve() == resume.parent.resolve():
        last_checkpoint = run_dir / "last.pt"
        if last_checkpoint.exists() and resume.resolve() != last_checkpoint.resolve():
            raise ValueError(
                f"run directory contains an existing last.pt; resume from {last_checkpoint} "
                "or use a new empty run directory"
            )
        if not last_checkpoint.exists():
            best_checkpoint = run_dir / "best.pt"
            if resume.resolve() != best_checkpoint.resolve():
                raise ValueError(
                    f"when last.pt is missing, only {best_checkpoint} can be resumed in place; "
                    "use a new empty run directory for any other checkpoint"
                )
            in_place_resume_from_best = True
    if (
        resume is not None
        and run_dir.exists()
        and any(run_dir.iterdir())
        and run_dir.resolve() != resume.parent.resolve()
    ):
        raise FileExistsError(f"resume run directory is not the checkpoint directory: {run_dir}")

    metadata = load_dataset_metadata(config.data.manifest_dir)
    verify_prepared_data(config.data.data_dir, metadata, config.data.manifest_dir)
    preflight = validate_training_request(config, metadata)
    preflight.raise_for_issues()
    for notice in preflight.notices:
        print(f"notice: {notice}")

    device = resolve_device(config.device)
    _seed_everything(config.train.seed)
    class_names = ("background", *metadata.class_names)
    model = _build_configured_model(config, model_factory, len(class_names)).to(device)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)
    train_dataset = VocDetectionDataset.from_manifests(
        config.data.manifest_dir,
        "train",
        data_dir=config.data.data_dir,
        training=True,
        transforms=build_detection_transforms(config.data.horizontal_flip, training=True),
        limit=config.data.max_train_samples,
    )
    valid_dataset = VocDetectionDataset.from_manifests(
        config.data.manifest_dir,
        "valid",
        data_dir=config.data.data_dir,
        training=False,
        limit=config.data.max_valid_samples,
    )
    generator = torch.Generator().manual_seed(config.train.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=detection_collate,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=detection_collate,
    )

    if dry_run_mode:
        result = dry_run(
            model,
            train_loader,
            optimizer,
            device,
            amp=config.train.amp,
            grad_clip=config.train.grad_clip,
        )
        return RunResult(run_dir=run_dir, completed_epochs=0, dry_run_result=result)

    start_epoch = 1
    best_metric = float("-inf")
    history: list[dict[str, float | int]] = []
    historical_best_checkpoint: dict[str, object] | None = None
    lineage_id = uuid.uuid4().hex
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        lineage_id = _validate_lineage_id(checkpoint, "resume checkpoint")
        _validate_resume_config(checkpoint, config)
        resume_identity = ResumeIdentity(
            model_name=config.model.name,
            class_names=class_names,
            manifest_identity=metadata.identity,
            preprocessing=_preprocessing_metadata(),
        )
        validate_resume_identity(checkpoint, resume_identity)
        scheduler_state_reference = scheduler.state_dict() if scheduler is not None else None
        saved_epoch, saved_history, best_metric = _validate_resumable_checkpoint(
            checkpoint,
            "resume checkpoint",
            scheduler_state_reference=scheduler_state_reference,
        )
        if in_place_resume_from_best:
            _validate_best_metric_history(checkpoint, saved_history, best_metric)
        model.load_state_dict(_mapping_value(checkpoint, "model_state"))
        optimizer.load_state_dict(dict(_mapping_value(checkpoint, "optimizer_state")))
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(dict(_mapping_value(checkpoint, "scheduler_state")))
        if config.train.epochs <= saved_epoch:
            raise ValueError(
                f"train.epochs ({config.train.epochs}) must be greater than checkpoint epoch {saved_epoch}"
            )
        start_epoch = saved_epoch + 1
        if run_dir.resolve() != resume.parent.resolve():
            historical_best_checkpoint = _load_historical_best_checkpoint(
                resume,
                checkpoint,
                config,
                resume_identity,
                saved_epoch=saved_epoch,
                best_metric=best_metric,
                scheduler_state_reference=scheduler_state_reference,
            )
        history = cast(list[dict[str, float | int]], saved_history)
        _restore_rng_state(checkpoint, generator, device)
    elif run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_yaml_atomic(run_dir / "config.yaml", config_to_dict(config))
    run_metadata = {
        **build_run_metadata(device=device, seed=config.train.seed),
        "manifest_identity": metadata.identity,
        "split_hashes": metadata.split_hashes,
        "class_names": list(class_names),
    }
    _write_yaml_atomic(run_dir / "run.yaml", run_metadata)
    if historical_best_checkpoint is not None:
        save_checkpoint(run_dir / "best.pt", historical_best_checkpoint)

    for epoch in range(start_epoch, config.train.epochs + 1):
        training_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            amp=config.train.amp,
            grad_clip=config.train.grad_clip,
        )
        validation_metrics = _evaluate_validation(model, valid_loader, device, class_names, metric_factory)
        row: dict[str, float | int] = {"epoch": epoch, **training_metrics}
        row.update(
            {
                f"valid_{key}": float(value)
                for key, value in validation_metrics.items()
                if isinstance(value, int | float)
            }
        )
        history.append(row)
        if scheduler is not None:
            scheduler.step()
        current = float(cast(int | float, validation_metrics[config.train.best_metric]))
        improved = current > best_metric
        if improved:
            best_metric = current
        payload = _checkpoint_payload(
            config,
            metadata,
            class_names,
            model,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            history,
            run_metadata,
            generator,
            lineage_id,
        )
        save_checkpoint(run_dir / "last.pt", payload)
        if improved:
            save_checkpoint(run_dir / "best.pt", payload)
        _write_metrics_csv(run_dir / "metrics.csv", history)
    return RunResult(run_dir=run_dir, completed_epochs=max(config.train.epochs, start_epoch - 1))


def _build_configured_model(config: AppConfig, model_factory: ModelFactory, num_classes: int) -> nn.Module:
    if config.model.factory is not None:
        if model_factory is not build_model:
            raise ValueError("external model factories require the default model builder")
        return build_model(
            config.model.name,
            num_classes,
            config.model.weights,
            config.model.params,
            factory=config.model.factory,
        )
    return model_factory(config.model.name, num_classes, config.model.weights, config.model.params)


def _evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: Sequence[str],
    metric_factory: MetricFactory,
) -> dict[str, object]:
    model.eval()
    metric = metric_factory(class_names)
    with torch.inference_mode():
        for images, targets in loader:
            device_images, _ = move_batch(images, targets, device)
            predictions = model(device_images)
            cpu_predictions = [
                {key: value.detach().cpu() for key, value in prediction.items()} for prediction in predictions
            ]
            metric.update(cpu_predictions, targets)
    return metric.compute()


def _build_optimizer(config: AppConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.train.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.train.lr,
            momentum=config.train.momentum,
            weight_decay=config.train.weight_decay,
        )
    if config.train.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=config.train.weight_decay)
    raise ValueError(f"unsupported optimizer {config.train.optimizer!r}")


def _build_scheduler(
    config: AppConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.train.scheduler == "none":
        return None
    if config.train.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    raise ValueError(f"unsupported scheduler {config.train.scheduler!r}")


def _checkpoint_payload(
    config: AppConfig,
    metadata: DatasetMetadata,
    class_names: Sequence[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    best_metric: float,
    history: list[dict[str, float | int]],
    run_metadata: Mapping[str, object],
    generator: torch.Generator,
    lineage_id: str,
) -> dict[str, object]:
    cuda_device_count = cast(int, run_metadata["cuda_device_count"])
    return {
        "schema_version": 1,
        "lineage_id": lineage_id,
        "config": config_to_dict(config),
        "model": {
            "name": config.model.name,
            "factory": config.model.factory,
            "params": dict(config.model.params),
        },
        "weight_policy": config.model.weights,
        "class_names": list(class_names),
        "preprocessing": _preprocessing_metadata(),
        "manifest_identity": metadata.identity,
        "split_hashes": dict(metadata.split_hashes),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "metric_history": [dict(row) for row in history],
        "run_metadata": dict(run_metadata),
        "rng_state": _capture_rng_state(generator, cuda_device_count=cuda_device_count),
    }


def _validate_resume_config(checkpoint: Mapping[str, object], config: AppConfig) -> None:
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("checkpoint is missing resolved config")
    expected = config_to_dict(config)
    saved_copy = json.loads(json.dumps(saved))
    expected_copy = json.loads(json.dumps(expected))
    for values in (saved_copy, expected_copy):
        values["train"].pop("epochs", None)
        values["data"].pop("num_workers", None)
        values.pop("device", None)
        values.pop("output_dir", None)
        values.pop("run_name", None)
    if saved_copy != expected_copy:
        raise ValueError("resume configuration changes training semantics")


def _load_historical_best_checkpoint(
    resume: Path,
    resume_checkpoint: Mapping[str, object],
    config: AppConfig,
    resume_identity: ResumeIdentity,
    *,
    saved_epoch: int,
    best_metric: float,
    scheduler_state_reference: Mapping[str, object] | None,
) -> dict[str, object]:
    best_path = resume if resume.name == "best.pt" else resume.parent / "best.pt"
    try:
        checkpoint = dict(resume_checkpoint) if best_path == resume else load_checkpoint(best_path)
    except CheckpointCompatibilityError as exc:
        raise CheckpointCompatibilityError(f"historical best checkpoint is unavailable: {best_path}: {exc}") from exc

    try:
        _validate_resume_config(checkpoint, config)
        validate_resume_identity(checkpoint, resume_identity)
        best_epoch, best_history, checkpoint_best_metric = _validate_resumable_checkpoint(
            checkpoint,
            "historical best checkpoint",
            scheduler_state_reference=scheduler_state_reference,
        )
        resume_epoch, resume_history, resume_best_metric = _validate_resumable_checkpoint(
            resume_checkpoint,
            "resume checkpoint",
            scheduler_state_reference=scheduler_state_reference,
        )
        if resume_epoch != saved_epoch:
            raise ValueError("resume checkpoint epoch changed during validation")
        if not 1 <= best_epoch <= saved_epoch:
            raise ValueError(f"epoch must be between 1 and resume epoch {saved_epoch}")
        if checkpoint_best_metric != best_metric or resume_best_metric != best_metric:
            raise ValueError("best_metric does not match the resume checkpoint")
        if _validate_lineage_id(checkpoint, "historical best checkpoint") != _validate_lineage_id(
            resume_checkpoint,
            "resume checkpoint",
        ):
            raise ValueError("lineage_id does not match the resume checkpoint")
        for field in (
            "model",
            "weight_policy",
            "class_names",
            "preprocessing",
            "manifest_identity",
            "split_hashes",
        ):
            if checkpoint.get(field) != resume_checkpoint.get(field):
                raise ValueError(f"source field {field} does not match the resume checkpoint")
        _validate_checkpoint_state_compatibility(checkpoint, resume_checkpoint)
        if best_history != resume_history[: len(best_history)]:
            raise ValueError("metric_history must be an exact prefix of the resume checkpoint")
        _validate_best_metric_history(checkpoint, best_history, checkpoint_best_metric)
        _validate_resume_metric_history_suffix(
            checkpoint,
            resume_history[len(best_history) :],
            checkpoint_best_metric,
        )
    except (CheckpointCompatibilityError, ValueError) as exc:
        raise CheckpointCompatibilityError(f"historical best checkpoint is incompatible: {best_path}: {exc}") from exc
    return checkpoint


def _validate_resumable_checkpoint(
    checkpoint: Mapping[str, object],
    label: str,
    *,
    scheduler_state_reference: Mapping[str, object] | None,
) -> tuple[int, list[dict[str, object]], float]:
    config = _mapping_value(checkpoint, "config")
    _mapping_value(checkpoint, "model")
    _mapping_value(checkpoint, "model_state")
    _mapping_value(checkpoint, "optimizer_state")
    _mapping_value(checkpoint, "preprocessing")
    _mapping_value(checkpoint, "split_hashes")
    _mapping_value(checkpoint, "run_metadata")
    _validate_lineage_id(checkpoint, label)
    _validate_rng_state_structure(checkpoint)

    train_config = config.get("train")
    if not isinstance(train_config, Mapping):
        raise ValueError(f"{label} config.train must be a mapping")
    scheduler = train_config.get("scheduler")
    scheduler_state = checkpoint.get("scheduler_state")
    if scheduler == "none":
        if scheduler_state is not None:
            raise ValueError(f"{label} scheduler_state must be null for scheduler 'none'")
        if scheduler_state_reference is not None:
            raise ValueError(f"{label} scheduler configuration does not match the fresh scheduler")
    elif scheduler == "step":
        if not isinstance(scheduler_state, Mapping):
            raise ValueError(f"{label} scheduler_state must be a mapping for scheduler 'step'")
        if not isinstance(scheduler_state_reference, Mapping) or not _state_value_is_compatible(
            scheduler_state,
            scheduler_state_reference,
        ):
            raise ValueError(f"{label} scheduler_state is incompatible with the fresh scheduler")
    else:
        raise ValueError(f"{label} config.train.scheduler is invalid")

    epoch = checkpoint.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
        raise ValueError(f"{label} epoch must be a positive integer")
    history = _validate_metric_history(checkpoint, epoch, label)
    metric = checkpoint.get("best_metric")
    if isinstance(metric, bool) or not isinstance(metric, int | float) or not math.isfinite(metric):
        raise ValueError(f"{label} best_metric must be a finite number")
    best_metric = float(metric)
    _validate_checkpoint_metric_history(checkpoint, history, best_metric, label)
    return epoch, history, best_metric


def _validate_metric_history(
    checkpoint: Mapping[str, object],
    epoch: int,
    label: str,
) -> list[dict[str, object]]:
    value = checkpoint.get("metric_history")
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} metric_history must be a sequence of mappings")
    history: list[dict[str, object]] = []
    for expected_epoch, row in enumerate(value, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} metric_history must be a sequence of mappings")
        row_epoch = row.get("epoch")
        if isinstance(row_epoch, bool) or not isinstance(row_epoch, int) or row_epoch != expected_epoch:
            raise ValueError(f"{label} metric_history epochs must be consecutive from 1")
        history.append(dict(row))
    if not history or history[-1]["epoch"] != epoch:
        raise ValueError(f"{label} metric_history must end at checkpoint epoch {epoch}")
    return history


def _validate_checkpoint_metric_history(
    checkpoint: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    best_metric: float,
    label: str,
) -> None:
    config = _mapping_value(checkpoint, "config")
    train_config = config.get("train")
    if not isinstance(train_config, Mapping) or not isinstance(train_config.get("best_metric"), str):
        raise ValueError(f"{label} config.train.best_metric must be a string")
    metric_field = f"valid_{train_config['best_metric']}"
    values: list[float] = []
    for row in history:
        metric_value = row.get(metric_field)
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, int | float)
            or not math.isfinite(metric_value)
        ):
            raise ValueError(f"{label} {metric_field} history must contain finite numbers")
        values.append(float(metric_value))
    if best_metric != max(values):
        raise ValueError(f"{label} best_metric must equal the maximum {metric_field} in metric_history")


def _validate_best_metric_history(
    checkpoint: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    best_metric: float,
) -> None:
    config = _mapping_value(checkpoint, "config")
    train_config = config.get("train")
    if not isinstance(train_config, Mapping) or not isinstance(train_config.get("best_metric"), str):
        raise ValueError("historical best checkpoint config.train.best_metric must be a string")
    metric_field = f"valid_{train_config['best_metric']}"
    metric_value = history[-1].get(metric_field)
    if (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, int | float)
        or not math.isfinite(metric_value)
        or float(metric_value) != best_metric
    ):
        raise ValueError("best_metric does not match metric_history")
    for row in history[:-1]:
        earlier_value = row.get(metric_field)
        if (
            isinstance(earlier_value, bool)
            or not isinstance(earlier_value, int | float)
            or not math.isfinite(earlier_value)
        ):
            raise ValueError(f"historical best checkpoint {metric_field} history must contain finite numbers")
        if best_metric <= float(earlier_value):
            raise ValueError("historical best checkpoint metric must be strictly greater than all earlier values")


def _validate_resume_metric_history_suffix(
    checkpoint: Mapping[str, object],
    history: Sequence[Mapping[str, object]],
    best_metric: float,
) -> None:
    config = _mapping_value(checkpoint, "config")
    train_config = config.get("train")
    if not isinstance(train_config, Mapping) or not isinstance(train_config.get("best_metric"), str):
        raise ValueError("historical best checkpoint config.train.best_metric must be a string")
    metric_field = f"valid_{train_config['best_metric']}"
    for row in history:
        metric_value = row.get(metric_field)
        if (
            isinstance(metric_value, bool)
            or not isinstance(metric_value, int | float)
            or not math.isfinite(metric_value)
            or float(metric_value) > best_metric
        ):
            raise ValueError(
                f"resume checkpoint {metric_field} values after historical best must be finite and at most best_metric"
            )


def _validate_checkpoint_state_compatibility(
    checkpoint: Mapping[str, object],
    resume_checkpoint: Mapping[str, object],
) -> None:
    model_state = _mapping_value(checkpoint, "model_state")
    resume_model_state = _mapping_value(resume_checkpoint, "model_state")
    if model_state.keys() != resume_model_state.keys():
        raise ValueError("model_state keys do not match the resume checkpoint")
    for key, value in model_state.items():
        reference = resume_model_state[key]
        if not isinstance(value, torch.Tensor) or not isinstance(reference, torch.Tensor):
            raise ValueError(f"model_state field {key!r} must be a tensor")
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError(f"model_state field {key!r} is incompatible with the resume checkpoint")

    optimizer_state = _mapping_value(checkpoint, "optimizer_state")
    resume_optimizer_state = _mapping_value(resume_checkpoint, "optimizer_state")
    _validate_optimizer_state_compatibility(optimizer_state, resume_optimizer_state)

    scheduler_state = checkpoint.get("scheduler_state")
    resume_scheduler_state = resume_checkpoint.get("scheduler_state")
    if isinstance(scheduler_state, Mapping) and isinstance(resume_scheduler_state, Mapping):
        if scheduler_state.keys() != resume_scheduler_state.keys():
            raise ValueError("scheduler_state keys do not match the resume checkpoint")
        for key, value in scheduler_state.items():
            if not _state_value_is_compatible(value, resume_scheduler_state[key]):
                raise ValueError(f"scheduler_state field {key!r} is incompatible with the resume checkpoint")


def _validate_optimizer_state_compatibility(
    state: Mapping[str, object],
    reference: Mapping[str, object],
) -> None:
    values = state.get("state")
    groups = state.get("param_groups")
    reference_values = reference.get("state")
    reference_groups = reference.get("param_groups")
    if not isinstance(values, Mapping) or not isinstance(reference_values, Mapping):
        raise ValueError("optimizer_state.state must be a mapping")
    if not isinstance(groups, list | tuple) or not isinstance(reference_groups, list | tuple):
        raise ValueError("optimizer_state.param_groups must be a sequence")
    if not groups or len(groups) != len(reference_groups):
        raise ValueError("optimizer_state.param_groups do not match the resume checkpoint")

    for index, (group, reference_group) in enumerate(zip(groups, reference_groups, strict=True)):
        if not isinstance(group, Mapping) or not isinstance(reference_group, Mapping):
            raise ValueError("optimizer_state.param_groups must contain mappings")
        if group.keys() != reference_group.keys():
            raise ValueError(f"optimizer_state param group {index} keys do not match the resume checkpoint")
        parameters = group.get("params")
        reference_parameters = reference_group.get("params")
        if parameters != reference_parameters or not isinstance(parameters, list | tuple):
            raise ValueError(f"optimizer_state param group {index} parameters do not match the resume checkpoint")
        for key, value in group.items():
            if key != "params" and not _state_value_is_compatible(value, reference_group[key]):
                raise ValueError(f"optimizer_state param group {index} field {key!r} is incompatible")

    if not set(values).issubset(reference_values):
        raise ValueError("optimizer_state contains parameters absent from the resume checkpoint")
    for key, value in values.items():
        if not _state_value_is_compatible(value, reference_values[key]):
            raise ValueError(f"optimizer_state parameter {key!r} is incompatible with the resume checkpoint")


def _state_value_is_compatible(value: object, reference: object) -> bool:
    if isinstance(value, torch.Tensor) or isinstance(reference, torch.Tensor):
        return (
            isinstance(value, torch.Tensor)
            and isinstance(reference, torch.Tensor)
            and value.shape == reference.shape
            and value.dtype == reference.dtype
        )
    if isinstance(value, Mapping) or isinstance(reference, Mapping):
        if not isinstance(value, Mapping) or not isinstance(reference, Mapping) or value.keys() != reference.keys():
            return False
        return all(_state_value_is_compatible(item, reference[key]) for key, item in value.items())
    if isinstance(value, list | tuple) or isinstance(reference, list | tuple):
        if (
            not isinstance(value, list | tuple)
            or not isinstance(reference, list | tuple)
            or len(value) != len(reference)
        ):
            return False
        return all(_state_value_is_compatible(item, expected) for item, expected in zip(value, reference, strict=True))
    return type(value) is type(reference)


def _validate_rng_state_structure(checkpoint: Mapping[str, object]) -> int | None:
    run_metadata = _mapping_value(checkpoint, "run_metadata")
    saved_device_value = run_metadata.get("device")
    if not isinstance(saved_device_value, str):
        raise ValueError("checkpoint run_metadata.device must be a string")
    try:
        saved_device = torch.device(saved_device_value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"checkpoint run_metadata.device is invalid: {exc}") from exc
    if saved_device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError("checkpoint run_metadata.device must identify a CPU, CUDA, or MPS device")
    cuda_device_count = run_metadata.get("cuda_device_count")
    if isinstance(cuda_device_count, bool) or not isinstance(cuda_device_count, int) or cuda_device_count < 0:
        raise ValueError("checkpoint run_metadata.cuda_device_count must be a nonnegative integer")
    rng_state = _mapping_value(checkpoint, "rng_state")
    python_state = rng_state.get("python")
    numpy_state = rng_state.get("numpy")
    torch_cpu = rng_state.get("torch_cpu")
    torch_cuda = rng_state.get("torch_cuda")
    loader_generator = rng_state.get("loader_generator")
    if not isinstance(python_state, list | tuple):
        raise ValueError("checkpoint rng_state.python must be a sequence")
    if not isinstance(numpy_state, Mapping):
        raise ValueError("checkpoint rng_state.numpy must be a mapping")
    if not isinstance(numpy_state.get("bit_generator"), str):
        raise ValueError("checkpoint rng_state.numpy.bit_generator must be a string")
    numpy_values = numpy_state.get("state")
    if not isinstance(numpy_values, list | tuple) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in numpy_values
    ):
        raise ValueError("checkpoint rng_state.numpy.state must be an integer sequence")
    for key in ("position", "has_gauss"):
        value = numpy_state.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"checkpoint rng_state.numpy.{key} must be an integer")
    cached_gaussian = numpy_state.get("cached_gaussian")
    if isinstance(cached_gaussian, bool) or not isinstance(cached_gaussian, int | float):
        raise ValueError("checkpoint rng_state.numpy.cached_gaussian must be a number")
    if not isinstance(torch_cpu, torch.Tensor) or not isinstance(loader_generator, torch.Tensor):
        raise ValueError("checkpoint rng_state torch_cpu and loader_generator must be tensors")
    if not isinstance(torch_cuda, list | tuple) or any(not isinstance(value, torch.Tensor) for value in torch_cuda):
        raise ValueError("checkpoint rng_state.torch_cuda must be a tensor sequence")
    if len(torch_cuda) != cuda_device_count:
        raise ValueError("checkpoint rng_state.torch_cuda length must match run_metadata.cuda_device_count")
    if saved_device.type == "cuda":
        if saved_device.index is None:
            raise ValueError("checkpoint run_metadata.device must include an explicit CUDA device index")
        if cuda_device_count == 0:
            raise ValueError("checkpoint saved on a CUDA device must have a positive cuda_device_count")
        saved_cuda_index = saved_device.index
        if saved_cuda_index >= cuda_device_count:
            raise ValueError("checkpoint source CUDA device index must be less than cuda_device_count")
    else:
        if cuda_device_count != 0 or torch_cuda:
            raise ValueError(
                "checkpoint saved on a non-CUDA device must have cuda_device_count 0 and no CUDA RNG states"
            )
        saved_cuda_index = None
    try:
        random.Random().setstate(cast(tuple[Any, ...], python_state))
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint rng_state.python is not restorable: {exc}") from exc
    try:
        numpy_generator = np.random.RandomState()
        numpy_generator.set_state(
            (
                cast(str, numpy_state["bit_generator"]),
                np.asarray(cast(Sequence[int], numpy_values), dtype=np.uint32),
                cast(int, numpy_state["position"]),
                cast(int, numpy_state["has_gauss"]),
                cast(float, cached_gaussian),
            )
        )
        torch.Generator().set_state(torch_cpu.cpu())
        torch.Generator().set_state(loader_generator.cpu())
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint rng_state is not restorable: {exc}") from exc
    if any(value.dtype != torch.uint8 or value.ndim != 1 or value.numel() == 0 for value in torch_cuda):
        raise ValueError("checkpoint rng_state.torch_cuda tensors must be nonempty one-dimensional byte tensors")
    return saved_cuda_index


def _preprocessing_metadata() -> dict[str, object]:
    return {"resize_owner": "torchvision_model", "input_range": [0.0, 1.0], "color_space": "RGB"}


def _mapping_value(checkpoint: Mapping[str, object], key: str) -> Mapping[str, Any]:
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint field {key} must be a mapping")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _capture_rng_state(generator: torch.Generator, *, cuda_device_count: int) -> dict[str, object]:
    numpy_state = cast(tuple[str, np.ndarray, int, int, float], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [torch.cuda.get_rng_state(index) for index in range(cuda_device_count)],
        "loader_generator": generator.get_state(),
    }


def _restore_rng_state(
    checkpoint: Mapping[str, object],
    generator: torch.Generator,
    device: torch.device,
) -> None:
    saved_cuda_index = _validate_rng_state_structure(checkpoint)
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, Mapping):
        raise ValueError("checkpoint is missing reproducibility RNG state")
    numpy_state = rng_state.get("numpy")
    if not isinstance(numpy_state, Mapping):
        raise ValueError("checkpoint numpy RNG state must be a mapping")
    try:
        random.setstate(cast(tuple[Any, ...], rng_state["python"]))
        np.random.set_state(
            (
                cast(str, numpy_state["bit_generator"]),
                np.asarray(cast(Sequence[int], numpy_state["state"]), dtype=np.uint32),
                cast(int, numpy_state["position"]),
                cast(int, numpy_state["has_gauss"]),
                cast(float, numpy_state["cached_gaussian"]),
            )
        )
        torch.set_rng_state(cast(torch.Tensor, rng_state["torch_cpu"]))
        generator.set_state(cast(torch.Tensor, rng_state["loader_generator"]))
        cuda_states = cast(Sequence[torch.Tensor], rng_state.get("torch_cuda", []))
        if device.type != "cuda" or not torch.cuda.is_available() or not cuda_states:
            return
        current_cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        if current_cuda_index < 0 or current_cuda_index >= torch.cuda.device_count():
            raise ValueError("current CUDA device index does not exist")
        if saved_cuda_index is None:
            raise ValueError("checkpoint with CUDA RNG states must identify a source CUDA device")
        current_device = torch.device("cuda", current_cuda_index)
        torch.cuda.set_rng_state(cuda_states[saved_cuda_index], device=current_device)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint reproducibility RNG state is invalid: {exc}") from exc


def _validate_lineage_id(checkpoint: Mapping[str, object], label: str) -> str:
    lineage_id = checkpoint.get("lineage_id")
    if not isinstance(lineage_id, str) or not lineage_id:
        raise ValueError(f"{label} lineage_id must be a nonempty string")
    return lineage_id


def _write_yaml_atomic(path: Path, data: Mapping[str, object]) -> None:
    _write_text_atomic(path, yaml.safe_dump(dict(data), sort_keys=False))


def _write_metrics_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
