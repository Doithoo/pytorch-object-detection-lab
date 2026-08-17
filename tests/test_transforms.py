import torch

from object_detector.data.transforms import RandomHorizontalFlip, filter_degenerate_boxes


def _target(boxes: list[list[float]]) -> dict[str, torch.Tensor]:
    count = len(boxes)
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
        "labels": torch.ones(count, dtype=torch.int64),
        "image_id": torch.tensor([1], dtype=torch.int64),
        "area": torch.ones(count, dtype=torch.float32),
        "iscrowd": torch.zeros(count, dtype=torch.int64),
        "difficult": torch.zeros(count, dtype=torch.bool),
    }


def test_horizontal_flip_updates_xyxy() -> None:
    image = torch.zeros((3, 10, 20))

    flipped, result = RandomHorizontalFlip(1.0)(image, _target([[2.0, 1.0, 8.0, 6.0]]))

    assert flipped.shape == image.shape
    assert result["boxes"].tolist() == [[12.0, 1.0, 18.0, 6.0]]


def test_degenerate_filter_preserves_field_alignment() -> None:
    target = _target([[2.0, 1.0, 8.0, 6.0], [4.0, 3.0, 4.0, 8.0]])
    target["labels"] = torch.tensor([3, 9])

    filtered, removed = filter_degenerate_boxes(target)

    assert removed == 1
    assert filtered["boxes"].tolist() == [[2.0, 1.0, 8.0, 6.0]]
    assert filtered["labels"].tolist() == [3]
    assert filtered["image_id"].tolist() == [1]
