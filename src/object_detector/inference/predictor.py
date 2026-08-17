from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from torch import nn
from torchvision.transforms.functional import pil_to_tensor

from object_detector.evaluation.visualization import render_detection_evidence
from object_detector.models.registry import build_model
from object_detector.preflight import resolve_device
from object_detector.training.checkpoint import CheckpointCompatibilityError, load_checkpoint

ModelFactory = Callable[[str, int, str, Mapping[str, object]], nn.Module]
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class PredictedObject:
    class_id: int
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class Prediction:
    image: str
    width: int
    height: int
    detections: tuple[PredictedObject, ...]


@dataclass(frozen=True)
class PredictionError:
    image: str
    error: str


@dataclass(frozen=True)
class BatchPredictionResult:
    predictions: tuple[Prediction, ...]
    errors: tuple[PredictionError, ...]


class Predictor:
    def __init__(self, model: nn.Module, class_names: Sequence[str], device: torch.device) -> None:
        self.model = model
        self.class_names = tuple(class_names)
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        device: str = "auto",
        model_factory: ModelFactory = build_model,
    ) -> Predictor:
        checkpoint = load_checkpoint(path)
        model_data = require_mapping(checkpoint, "model")
        class_names = tuple(require_string_sequence(checkpoint, "class_names"))
        params = require_mapping(model_data, "params", prefix="model")
        model_name = model_data.get("name")
        if not isinstance(model_name, str):
            raise CheckpointCompatibilityError("checkpoint field model.name must be a string")
        model = model_factory(model_name, len(class_names), "none", params)
        model.load_state_dict(dict(require_mapping(checkpoint, "model_state")))
        resolved_device = resolve_device(device)
        model.to(resolved_device).eval()
        return cls(model, class_names, resolved_device)

    def predict_single(
        self,
        image_path: Path,
        output_dir: Path,
        *,
        score_threshold: float,
        display_limit: int,
        overwrite: bool = False,
    ) -> Prediction:
        json_path = output_dir / f"{image_path.stem}.json"
        image_output = output_dir / f"{image_path.stem}.png"
        if not overwrite and (json_path.exists() or image_output.exists()):
            raise FileExistsError(f"prediction output already exists for {image_path.name}")
        prediction, image, tensors = self._predict_path(image_path, score_threshold)
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(json_path, asdict(prediction))
        render_detection_evidence(
            image,
            _display_prediction(tensors, display_limit),
            _empty_target(),
            self.class_names,
            image_output,
            score_threshold=score_threshold,
        )
        return prediction

    def predict_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        score_threshold: float,
        display_limit: int,
        overwrite: bool = False,
    ) -> BatchPredictionResult:
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            raise FileExistsError(f"prediction output directory already exists: {output_dir}")
        paths = sorted(
            (
                path
                for path in input_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            ),
            key=lambda path: path.relative_to(input_dir).as_posix(),
        )
        predictions: list[Prediction] = []
        errors: list[PredictionError] = []
        serialized_predictions: list[dict[str, object]] = []
        serialized_errors: list[dict[str, str]] = []
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in paths:
            relative = path.relative_to(input_dir)
            try:
                prediction, image, tensors = self._predict_path(path, score_threshold)
            except (OSError, ValueError) as exc:
                error = PredictionError(str(path), str(exc))
                errors.append(error)
                serialized_errors.append({"image": relative.as_posix(), "error": str(exc)})
                continue
            predictions.append(prediction)
            serialized = asdict(prediction)
            serialized["image"] = relative.as_posix()
            serialized_predictions.append(serialized)
            visualization_path = output_dir / "visualizations" / relative.parent / f"{relative.name}.png"
            render_detection_evidence(
                image,
                _display_prediction(tensors, display_limit),
                _empty_target(),
                self.class_names,
                visualization_path,
                score_threshold=score_threshold,
            )
        result = BatchPredictionResult(tuple(predictions), tuple(errors))
        _write_json_atomic(
            output_dir / "predictions.json",
            {"predictions": serialized_predictions, "errors": serialized_errors},
        )
        return result

    def _predict_path(
        self,
        image_path: Path,
        score_threshold: float,
    ) -> tuple[Prediction, torch.Tensor, dict[str, torch.Tensor]]:
        with Image.open(image_path) as source:
            image = pil_to_tensor(source.convert("RGB")).float().div(255.0)
            width, height = source.size
        with torch.inference_mode():
            raw_output = self.model([image.to(self.device)])
        output = cast(Sequence[Mapping[str, torch.Tensor]], raw_output)[0]
        cpu_output = {key: value.detach().cpu() for key, value in output.items()}
        mask = cpu_output["scores"] >= score_threshold
        filtered = {key: value[mask] for key, value in cpu_output.items()}
        detections = tuple(
            _predicted_object(box, label, score, self.class_names)
            for box, label, score in zip(filtered["boxes"], filtered["labels"], filtered["scores"], strict=True)
        )
        return Prediction(str(image_path), width, height, detections), image, filtered


def require_mapping(
    source: Mapping[str, object],
    key: str,
    *,
    prefix: str = "",
) -> Mapping[str, Any]:
    value = source.get(key)
    path = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(f"checkpoint field {path} must be a mapping")
    return value


def require_string_sequence(source: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = source.get(key)
    if not isinstance(value, list | tuple) or not value or not all(isinstance(item, str) for item in value):
        raise CheckpointCompatibilityError(f"checkpoint field {key} must be a nonempty sequence of strings")
    return tuple(value)


def _predicted_object(
    box: torch.Tensor,
    label: torch.Tensor,
    score: torch.Tensor,
    class_names: Sequence[str],
) -> PredictedObject:
    class_id = int(label)
    if class_id < 0 or class_id >= len(class_names):
        raise CheckpointCompatibilityError(f"prediction contains unknown class ID {class_id}")
    values = box.tolist()
    return PredictedObject(
        class_id,
        class_names[class_id],
        float(score),
        (float(values[0]), float(values[1]), float(values[2]), float(values[3])),
    )


def _display_prediction(prediction: Mapping[str, torch.Tensor], display_limit: int) -> dict[str, torch.Tensor]:
    if display_limit < 0:
        raise ValueError("display_limit must not be negative")
    return {key: value[:display_limit] for key, value in prediction.items()}


def _empty_target() -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "iscrowd": torch.empty((0,), dtype=torch.int64),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
