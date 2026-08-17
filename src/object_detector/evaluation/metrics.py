from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionMetric:
    def __init__(self, class_names: Sequence[str]) -> None:
        if len(class_names) < 2 or class_names[0] != "background":
            raise ValueError("class_names must start with background and include an object class")
        self.class_names = tuple(class_names)
        self.metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
        self.image_count = 0

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

    def compute(self) -> dict[str, object]:
        if self.image_count == 0:
            raise ValueError("cannot compute detection metrics without images")
        raw = self.metric.compute()
        return {
            "map_50_95": _normalized_float(raw["map"]),
            "image_count": self.image_count,
        }

    def reset(self) -> None:
        self.metric.reset()
        self.image_count = 0


def _normalized_float(value: torch.Tensor) -> float:
    result = float(value.detach().cpu())
    return max(result, 0.0)
