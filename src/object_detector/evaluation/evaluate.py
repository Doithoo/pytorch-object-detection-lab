from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from object_detector.config import AppConfig, ConfigError, config_from_dict
from object_detector.data.dataset import VocDetectionDataset, detection_collate
from object_detector.data.manifest import load_dataset_metadata, verify_prepared_data
from object_detector.evaluation.errors import DetectionError, analyze_image_errors
from object_detector.evaluation.metrics import DetectionMetric, metric_backend_versions
from object_detector.evaluation.visualization import render_detection_evidence
from object_detector.models.registry import build_model
from object_detector.preflight import resolve_device
from object_detector.training.checkpoint import (
    CheckpointCompatibilityError,
    load_checkpoint,
    validate_preprocessing_contract,
)

ModelFactory = Callable[[str, int, str, Mapping[str, object]], nn.Module]


@dataclass(frozen=True)
class EvaluationResult:
    output_dir: Path
    metrics: dict[str, object]
    errors: tuple[DetectionError, ...]


@dataclass(frozen=True)
class _EvidenceReference:
    image_id: str
    dataset_index: int
    predictions: tuple[dict[str, object], ...]


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    split: str,
    output_dir: Path,
    device: str,
    score_threshold: float,
    overwrite: bool,
    model_factory: ModelFactory = build_model,
) -> EvaluationResult:
    checkpoint = load_checkpoint(checkpoint_path)
    validate_preprocessing_contract(checkpoint)
    config = _checkpoint_config(checkpoint)
    metadata = load_dataset_metadata(config.data.manifest_dir)
    verify_prepared_data(config.data.data_dir, metadata, config.data.manifest_dir)
    if checkpoint.get("manifest_identity") != metadata.identity:
        raise CheckpointCompatibilityError("checkpoint manifest_identity does not match prepared dataset")

    class_values = checkpoint.get("class_names")
    if not isinstance(class_values, list | tuple) or not all(isinstance(value, str) for value in class_values):
        raise CheckpointCompatibilityError("checkpoint class_names must be a sequence of strings")
    class_names = tuple(class_values)
    model_data = _checkpoint_mapping(checkpoint, "model")
    model_name = model_data.get("name")
    params = model_data.get("params", {})
    if not isinstance(model_name, str) or not isinstance(params, Mapping):
        raise CheckpointCompatibilityError("checkpoint model specification is invalid")
    model = model_factory(model_name, len(class_names), "none", dict(params))
    model.load_state_dict(dict(_checkpoint_mapping(checkpoint, "model_state")))
    resolved_device = resolve_device(device)
    model.to(resolved_device)

    limit_by_split = {
        "train": config.data.max_train_samples,
        "valid": config.data.max_valid_samples,
        "test": config.data.max_test_samples,
    }
    if split not in limit_by_split:
        raise ValueError(f"unknown split {split!r}")
    dataset = VocDetectionDataset.from_manifests(
        config.data.manifest_dir,
        split,
        data_dir=config.data.data_dir,
        training=False,
        limit=limit_by_split[split],
    )
    return evaluate_model(
        model,
        dataset,
        class_names,
        resolved_device,
        output_dir,
        score_threshold=score_threshold,
        error_score_threshold=config.evaluation.error_score_threshold,
        error_iou_threshold=config.evaluation.error_iou_threshold,
        overwrite=overwrite,
        manifest_identity=metadata.identity,
        checkpoint_sha256=_sha256_file(checkpoint_path),
        split=split,
        max_detections=config.evaluation.max_detections,
        batch_size=config.train.batch_size,
        num_workers=config.data.num_workers,
    )


def evaluate_model(
    model: nn.Module,
    dataset: VocDetectionDataset,
    class_names: Sequence[str],
    device: torch.device,
    output_dir: Path,
    *,
    score_threshold: float,
    error_score_threshold: float,
    error_iou_threshold: float,
    overwrite: bool = False,
    manifest_identity: str | None = None,
    checkpoint_sha256: str | None = None,
    split: str | None = None,
    max_detections: int = 100,
    batch_size: int = 1,
    num_workers: int = 0,
) -> EvaluationResult:
    if batch_size < 1:
        raise ValueError("evaluation batch_size must be positive")
    if num_workers < 0:
        raise ValueError("evaluation num_workers must not be negative")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"evaluation output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    visualization_dir = stage / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.eval()
        metric = DetectionMetric(class_names)
        evidence_references: list[_EvidenceReference] = []
        error_records: list[DetectionError] = []
        prediction_records: list[dict[str, object]] = []
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=detection_collate,
        )
        dataset_index = 0
        with torch.inference_mode():
            for images, targets in loader:
                device_images = [image.to(device) for image in images]
                outputs = cast(Sequence[Mapping[str, torch.Tensor]], model(device_images))
                cpu_predictions = [_cpu_prediction(output) for output in outputs]
                metric.update(cpu_predictions, targets)
                for target, prediction in zip(targets, cpu_predictions, strict=True):
                    image_id = dataset.source_image_id(dataset_index)
                    image_errors = analyze_image_errors(
                        image_id, prediction, target, class_names, error_score_threshold, error_iou_threshold
                    )
                    error_records.extend(image_errors)
                    serialized = _serializable_predictions(prediction, class_names, score_threshold)
                    evidence_references.append(_EvidenceReference(image_id, dataset_index, tuple(serialized)))
                    prediction_records.append(
                        {
                            "image_id": image_id,
                            "predictions": serialized,
                        }
                    )
                    dataset_index += 1
                del images, targets, device_images, outputs, cpu_predictions
        if not evidence_references:
            raise ValueError("evaluation dataset is empty")
        metrics = metric.compute()
        evaluation_payload: dict[str, object] = {
            "metrics": _rounded(metrics),
            "backend_versions": metric_backend_versions(),
            "score_threshold": score_threshold,
            "error_score_threshold": error_score_threshold,
            "error_iou_threshold": error_iou_threshold,
            "max_detections": max_detections,
        }
        if manifest_identity is not None:
            evaluation_payload["manifest_identity"] = manifest_identity
        if checkpoint_sha256 is not None:
            evaluation_payload["checkpoint_sha256"] = checkpoint_sha256
        if split is not None:
            evaluation_payload["split"] = split
        _write_json_atomic(stage / "evaluation.json", evaluation_payload)
        _write_json_atomic(stage / "predictions.json", prediction_records)
        _write_per_class_csv(stage / "per_class.csv", cast(Sequence[Mapping[str, object]], metrics["per_class"]))
        _write_errors_csv(stage / "errors.csv", error_records)
        _render_ranked_evidence(
            dataset,
            evidence_references,
            error_records,
            class_names,
            visualization_dir,
            score_threshold,
        )
        _publish_directory(stage, output_dir, overwrite=overwrite)
        return EvaluationResult(output_dir, metrics, tuple(error_records))
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _publish_directory(stage: Path, destination: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite and any(destination.iterdir()):
            raise FileExistsError(f"evaluation output directory already exists: {destination}")
        backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except OSError:
        if backup is not None and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _cpu_prediction(prediction: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in prediction.items()}


def _serializable_predictions(
    prediction: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
    score_threshold: float,
) -> list[dict[str, object]]:
    records = []
    for box, label, score in zip(
        prediction["boxes"].tolist(),
        prediction["labels"].tolist(),
        prediction["scores"].tolist(),
        strict=True,
    ):
        if score < score_threshold:
            continue
        class_id = int(label)
        class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class-{class_id}"
        records.append(
            {
                "box": [round(float(value), 6) for value in box],
                "class_id": class_id,
                "class_name": class_name,
                "score": round(float(score), 6),
            }
        )
    return records


def _write_per_class_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = ["class_id", "class_name", "map_50_95", "mar_100"]
    _write_csv_atomic(path, fieldnames, [{key: _rounded(row[key]) for key in fieldnames} for row in rows])


def _write_errors_csv(path: Path, errors: Sequence[DetectionError]) -> None:
    fieldnames = ["image_id", "kind", "class_name", "score", "iou", "box"]
    rows = []
    for error in errors:
        row = asdict(error)
        row["score"] = "" if error.score is None else round(error.score, 6)
        row["iou"] = round(error.iou, 6)
        row["box"] = json.dumps([round(value, 6) for value in error.box])
        rows.append(row)
    _write_csv_atomic(path, fieldnames, rows)


def _write_csv_atomic(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, data: object) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _render_ranked_evidence(
    dataset: VocDetectionDataset,
    references: Sequence[_EvidenceReference],
    errors: Sequence[DetectionError],
    class_names: Sequence[str],
    output_dir: Path,
    score_threshold: float,
) -> None:
    by_id = {item.image_id: item for item in references}
    _render_evidence_reference(dataset, references[0], class_names, output_dir / "summary.png", score_threshold)
    for kind in ("missed", "false_positive"):
        counts = Counter(error.image_id for error in errors if error.kind == kind)
        ranked_ids = sorted(counts, key=lambda image_id: (-counts[image_id], image_id))[:5]
        for rank, image_id in enumerate(ranked_ids, start=1):
            _render_evidence_reference(
                dataset,
                by_id[image_id],
                class_names,
                output_dir / f"{kind}-{rank:02d}-{_safe_name(image_id)}.png",
                score_threshold,
            )


def _render_evidence_reference(
    dataset: VocDetectionDataset,
    reference: _EvidenceReference,
    class_names: Sequence[str],
    output: Path,
    score_threshold: float,
) -> None:
    image, target = dataset[reference.dataset_index]
    predictions = reference.predictions
    prediction = {
        "boxes": torch.tensor([item["box"] for item in predictions], dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor([item["class_id"] for item in predictions], dtype=torch.int64),
        "scores": torch.tensor([item["score"] for item in predictions], dtype=torch.float32),
    }
    render_detection_evidence(
        image,
        prediction,
        target,
        class_names,
        output,
        score_threshold=score_threshold,
    )


def _safe_name(image_id: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in image_id)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_config(checkpoint: Mapping[str, object]) -> AppConfig:
    raw = _checkpoint_mapping(checkpoint, "config")
    try:
        return config_from_dict(raw)
    except ConfigError as exc:
        raise CheckpointCompatibilityError(f"checkpoint resolved config is invalid: {exc}") from exc


def _checkpoint_mapping(checkpoint: Mapping[str, object], key: str) -> Mapping[str, Any]:
    value = checkpoint.get(key)
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"checkpoint field {key} must be a mapping")
    return value


def _rounded(value: object) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, Mapping):
        return {str(key): _rounded(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_rounded(item) for item in value]
    return value
