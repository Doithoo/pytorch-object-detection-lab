from __future__ import annotations

import csv
import json
import os
import random
import tempfile
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
from object_detector.data.manifest import DatasetMetadata, load_dataset_metadata
from object_detector.evaluation.metrics import DetectionMetric
from object_detector.models.registry import build_model
from object_detector.preflight import resolve_device, validate_training_request
from object_detector.training.checkpoint import (
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
    metadata = load_dataset_metadata(config.data.manifest_dir)
    preflight = validate_training_request(config, metadata)
    preflight.raise_for_issues()
    for notice in preflight.notices:
        print(f"notice: {notice}")

    device = resolve_device(config.device)
    _seed_everything(config.train.seed)
    class_names = ("background", *metadata.class_names)
    model = model_factory(
        config.model.name,
        len(class_names),
        config.model.weights,
        config.model.params,
    ).to(device)
    optimizer = _build_optimizer(config, model)
    scheduler = _build_scheduler(config, optimizer)
    run_name = config.run_name or "run"
    run_dir = config.output_dir / run_name

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
    if resume is not None:
        checkpoint = load_checkpoint(resume)
        _validate_resume_config(checkpoint, config)
        validate_resume_identity(
            checkpoint,
            ResumeIdentity(
                model_name=config.model.name,
                class_names=class_names,
                manifest_identity=metadata.identity,
                preprocessing=_preprocessing_metadata(),
            ),
        )
        model.load_state_dict(_mapping_value(checkpoint, "model_state"))
        optimizer.load_state_dict(dict(_mapping_value(checkpoint, "optimizer_state")))
        if scheduler is not None and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(dict(_mapping_value(checkpoint, "scheduler_state")))
        start_epoch = cast(int, checkpoint["epoch"]) + 1
        best_metric = cast(float, checkpoint["best_metric"])
        saved_history = cast(Sequence[Mapping[str, float | int]], checkpoint.get("metric_history", []))
        history = [dict(item) for item in saved_history]
        _restore_rng_state(checkpoint, generator)
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
        current = float(cast(int | float, validation_metrics["map_50_95"]))
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
        )
        save_checkpoint(run_dir / "last.pt", payload)
        if improved:
            save_checkpoint(run_dir / "best.pt", payload)
        _write_metrics_csv(run_dir / "metrics.csv", history)
    return RunResult(run_dir=run_dir, completed_epochs=max(config.train.epochs, start_epoch - 1))


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
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "config": config_to_dict(config),
        "model": {"name": config.model.name, "params": dict(config.model.params)},
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
        "rng_state": _capture_rng_state(generator),
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


def _capture_rng_state(generator: torch.Generator) -> dict[str, object]:
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
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_generator": generator.get_state(),
    }


def _restore_rng_state(checkpoint: Mapping[str, object], generator: torch.Generator) -> None:
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
        if cuda_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(list(cuda_states))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint reproducibility RNG state is invalid: {exc}") from exc


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
