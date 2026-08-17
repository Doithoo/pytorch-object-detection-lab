import pytest
import torch

from object_detector.evaluation.metrics import DetectionMetric


def test_perfect_prediction_returns_map() -> None:
    metric = DetectionMetric(("background", "dog"))
    target = {
        "boxes": torch.tensor([[1.0, 1.0, 9.0, 9.0]]),
        "labels": torch.tensor([1]),
        "iscrowd": torch.tensor([0]),
        "area": torch.tensor([64.0]),
    }
    prediction = {
        "boxes": target["boxes"].clone(),
        "labels": target["labels"].clone(),
        "scores": torch.tensor([0.99]),
    }

    metric.update([prediction], [target])
    result = metric.compute()

    assert result["map_50_95"] == pytest.approx(1.0)
