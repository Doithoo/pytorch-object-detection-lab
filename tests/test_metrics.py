import pytest
import torch

import object_detector.evaluation.metrics as metrics
from object_detector.evaluation.metrics import DetectionMetric

EXPECTED_KEYS = {
    "map_50_95",
    "map_50",
    "map_75",
    "mar_1",
    "mar_10",
    "mar_100",
    "per_class",
    "image_count",
    "target_count",
    "prediction_count",
}


def test_metric_backend_versions_are_recordable() -> None:
    versions = metrics.metric_backend_versions()

    assert set(versions) == {"torchmetrics", "pycocotools"}
    assert all(isinstance(value, str) and value for value in versions.values())


def _target(*, difficult: bool = False):
    return {
        "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
        "labels": torch.tensor([1]),
        "iscrowd": torch.tensor([int(difficult)]),
        "area": torch.tensor([64.0]),
    }


def _prediction(target, *, empty: bool = False):
    if empty:
        return {
            "boxes": torch.empty((0, 4)),
            "labels": torch.empty((0,), dtype=torch.int64),
            "scores": torch.empty((0,)),
        }
    return {
        "boxes": target["boxes"].clone(),
        "labels": target["labels"].clone(),
        "scores": torch.tensor([0.99]),
    }


def test_perfect_prediction_returns_map() -> None:
    metric = DetectionMetric(("background", "dog"))
    targets = [_target(), _target()]
    predictions = [_prediction(target) for target in targets]

    metric.update(predictions, targets)
    result = metric.compute()

    assert set(result) == EXPECTED_KEYS
    assert result["map_50_95"] == pytest.approx(1.0)
    assert result["map_50"] == pytest.approx(1.0)
    assert result["map_75"] == pytest.approx(1.0)
    assert result["mar_100"] == pytest.approx(1.0)
    assert result["per_class"] == (
        {"class_id": 1, "class_name": "dog", "map_50_95": pytest.approx(1.0), "mar_100": pytest.approx(1.0)},
    )
    assert result["image_count"] == 2
    assert result["target_count"] == 2
    assert result["prediction_count"] == 2


def test_empty_predictions_return_numeric_zeros() -> None:
    metric = DetectionMetric(("background", "dog"))
    targets = [_target(), _target()]

    metric.update([_prediction(target, empty=True) for target in targets], targets)
    result = metric.compute()

    assert set(result) == EXPECTED_KEYS
    for key in ("map_50_95", "map_50", "map_75", "mar_1", "mar_10", "mar_100"):
        assert result[key] == 0.0
    assert result["image_count"] == 2
    assert result["target_count"] == 2
    assert result["prediction_count"] == 0


def test_difficult_only_match_is_ignored() -> None:
    metric = DetectionMetric(("background", "dog"))
    difficult = _target(difficult=True)
    empty_target = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "iscrowd": torch.empty((0,), dtype=torch.int64),
        "area": torch.empty((0,)),
    }

    metric.update([_prediction(difficult), _prediction(empty_target, empty=True)], [difficult, empty_target])
    result = metric.compute()

    assert result["target_count"] == 0
    assert result["prediction_count"] == 1
    assert result["map_50_95"] == 0.0
    assert result["mar_100"] == 0.0
