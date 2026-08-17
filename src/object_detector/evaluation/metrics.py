from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


@dataclass(frozen=True)
class ClassMetrics:
    class_id: int
    class_name: str
    map_50_95: float
    mar_100: float


@dataclass(frozen=True)
class DetectionMetrics:
    map_50_95: float
    map_50: float
    map_75: float
    mar_1: float
    mar_10: float
    mar_100: float
    per_class: tuple[ClassMetrics, ...]
    image_count: int
    target_count: int
    prediction_count: int


class DetectionMetric:
    def __init__(self, class_names: Sequence[str]) -> None:
        if len(class_names) < 2 or class_names[0] != "background":
            raise ValueError("class_names must start with background and include an object class")
        self.class_names = tuple(class_names)
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        self.image_count = 0
        self.target_count = 0
        self.prediction_count = 0

    def update(
        self,
        predictions: Sequence[Mapping[str, torch.Tensor]],
        targets: Sequence[Mapping[str, torch.Tensor]],
    ) -> None:
        if len(predictions) != len(targets):
            raise ValueError("predictions and targets must have the same length")
        metric_predictions = [
            {key: prediction[key].detach().cpu() for key in ("boxes", "scores", "labels")} for prediction in predictions
        ]
        metric_targets = [
            {key: target[key].detach().cpu() for key in ("boxes", "labels", "iscrowd", "area") if key in target}
            for target in targets
        ]
        self.metric.update(metric_predictions, metric_targets)
        self.image_count += len(predictions)
        self.prediction_count += sum(int(prediction["scores"].numel()) for prediction in predictions)
        self.target_count += sum(_ordinary_target_count(target) for target in targets)

    def compute(self) -> dict[str, object]:
        if self.image_count == 0:
            raise ValueError("cannot compute detection metrics without images")
        raw = self.metric.compute()
        per_class = _per_class_metrics(raw, self.class_names)
        result = DetectionMetrics(
            map_50_95=_normalized_float(raw["map"]),
            map_50=_normalized_float(raw["map_50"]),
            map_75=_normalized_float(raw["map_75"]),
            mar_1=_normalized_float(raw["mar_1"]),
            mar_10=_normalized_float(raw["mar_10"]),
            mar_100=_normalized_float(raw["mar_100"]),
            per_class=per_class,
            image_count=self.image_count,
            target_count=self.target_count,
            prediction_count=self.prediction_count,
        )
        return asdict(result)

    def reset(self) -> None:
        self.metric.reset()
        self.image_count = 0
        self.target_count = 0
        self.prediction_count = 0


def metric_backend_versions() -> dict[str, str]:
    return {
        "torchmetrics": version("torchmetrics"),
        "pycocotools": version("pycocotools"),
    }


def _normalized_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu())
    return max(result, 0.0)


def _ordinary_target_count(target: Mapping[str, torch.Tensor]) -> int:
    labels = target["labels"]
    iscrowd = target.get("iscrowd")
    if iscrowd is None:
        return int(labels.numel())
    return int((iscrowd == 0).sum().item())


def _per_class_metrics(raw: Mapping[str, torch.Tensor], class_names: Sequence[str]) -> tuple[ClassMetrics, ...]:
    class_ids = raw["classes"].detach().cpu().reshape(-1).tolist()
    average_precisions = raw["map_per_class"].detach().cpu().reshape(-1).tolist()
    average_recalls = raw["mar_100_per_class"].detach().cpu().reshape(-1).tolist()
    result = []
    for class_id, average_precision, average_recall in zip(class_ids, average_precisions, average_recalls, strict=True):
        class_id = int(class_id)
        if class_id == 0:
            continue
        if class_id >= len(class_names):
            raise ValueError(f"metric returned unknown class ID {class_id}")
        result.append(
            ClassMetrics(
                class_id=class_id,
                class_name=class_names[class_id],
                map_50_95=max(float(average_precision), 0.0),
                mar_100=max(float(average_recall), 0.0),
            )
        )
    return tuple(result)
