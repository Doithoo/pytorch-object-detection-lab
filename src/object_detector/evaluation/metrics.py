from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version

import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


@dataclass(frozen=True)
class ClassMetrics:
    class_id: int
    class_name: str
    map_50_95: float
    mar_100: float
    voc_ap_50_11: float


@dataclass(frozen=True)
class DetectionMetrics:
    map_50_95: float
    map_50: float
    map_75: float
    voc_map_50_11: float
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
        self._voc_predictions: list[dict[str, torch.Tensor]] = []
        self._voc_targets: list[dict[str, torch.Tensor]] = []

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
        self._voc_predictions.extend(metric_predictions)
        self._voc_targets.extend(metric_targets)

    def compute(self) -> dict[str, object]:
        if self.image_count == 0:
            raise ValueError("cannot compute detection metrics without images")
        raw = self.metric.compute()
        voc_ap = _voc_ap_50_11(self._voc_predictions, self._voc_targets, self.class_names)
        per_class = _per_class_metrics(raw, self.class_names, voc_ap)
        result = DetectionMetrics(
            map_50_95=_normalized_float(raw["map"]),
            map_50=_normalized_float(raw["map_50"]),
            map_75=_normalized_float(raw["map_75"]),
            voc_map_50_11=sum(voc_ap.values()) / len(voc_ap),
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
        self._voc_predictions.clear()
        self._voc_targets.clear()


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


def _per_class_metrics(
    raw: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
    voc_ap: Mapping[int, float],
) -> tuple[ClassMetrics, ...]:
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
                voc_ap_50_11=voc_ap.get(class_id, 0.0),
            )
        )
    return tuple(result)


def _voc_ap_50_11(
    predictions: Sequence[Mapping[str, torch.Tensor]],
    targets: Sequence[Mapping[str, torch.Tensor]],
    class_names: Sequence[str],
) -> dict[int, float]:
    result: dict[int, float] = {}
    for class_id in range(1, len(class_names)):
        records: list[tuple[float, int, torch.Tensor]] = []
        positive_count = 0
        ordinary_by_image: list[torch.Tensor] = []
        difficult_by_image: list[torch.Tensor] = []
        for image_index, target in enumerate(targets):
            labels = target["labels"]
            boxes = target["boxes"]
            iscrowd = target.get("iscrowd", torch.zeros_like(labels))
            ordinary = boxes[(labels == class_id) & (iscrowd == 0)]
            difficult = boxes[(labels == class_id) & (iscrowd != 0)]
            ordinary_by_image.append(ordinary)
            difficult_by_image.append(difficult)
            positive_count += len(ordinary)
            prediction = predictions[image_index]
            keep = prediction["labels"] == class_id
            for box, score in zip(prediction["boxes"][keep], prediction["scores"][keep], strict=True):
                records.append((float(score), image_index, box))
        if positive_count == 0:
            result[class_id] = 0.0
            continue
        records.sort(key=lambda item: -item[0])
        matched = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in ordinary_by_image]
        true_positive: list[float] = []
        false_positive: list[float] = []
        for _score, image_index, box in records:
            ordinary = ordinary_by_image[image_index]
            best_index, best_iou = _best_iou(box, ordinary)
            if best_index is not None and best_iou >= 0.5:
                if not matched[image_index][best_index]:
                    matched[image_index][best_index] = True
                    true_positive.append(1.0)
                    false_positive.append(0.0)
                else:
                    true_positive.append(0.0)
                    false_positive.append(1.0)
                continue
            _, difficult_iou = _best_iou(box, difficult_by_image[image_index])
            if difficult_iou >= 0.5:
                continue
            true_positive.append(0.0)
            false_positive.append(1.0)
        if not true_positive:
            result[class_id] = 0.0
            continue
        true_positive_tensor = torch.tensor(true_positive).cumsum(0)
        false_positive_tensor = torch.tensor(false_positive).cumsum(0)
        recalls = true_positive_tensor / positive_count
        precisions = true_positive_tensor / (true_positive_tensor + false_positive_tensor).clamp_min(1e-12)
        result[class_id] = (
            sum(
                float(precisions[recalls >= threshold].max()) if bool((recalls >= threshold).any()) else 0.0
                for threshold in torch.arange(0.0, 1.01, 0.1)
            )
            / 11.0
        )
    return result


def _best_iou(box: torch.Tensor, boxes: torch.Tensor) -> tuple[int | None, float]:
    if not len(boxes):
        return None, 0.0
    ious = box_iou(box.reshape(1, 4), boxes).reshape(-1)
    index = int(torch.argmax(ious))
    return index, float(ious[index])
