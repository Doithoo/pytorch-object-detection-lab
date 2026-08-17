from __future__ import annotations

import pytest
import torch

from object_detector.data.transforms import DetectionTarget
from object_detector.training.trainer import NonFiniteLossError, dry_run, train_one_epoch
from tests.fixtures.models import FakeDetector


def _batch() -> tuple[list[torch.Tensor], list[DetectionTarget]]:
    return (
        [torch.zeros((3, 8, 8))],
        [
            {
                "boxes": torch.tensor([[1.0, 1.0, 7.0, 7.0]]),
                "labels": torch.tensor([1], dtype=torch.int64),
                "image_id": torch.tensor([11], dtype=torch.int64),
            }
        ],
    )


def test_train_one_epoch_aggregates_named_losses() -> None:
    model = FakeDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    stats = train_one_epoch(model, [_batch(), _batch()], optimizer, torch.device("cpu"))

    assert set(stats) == {"loss_total", "loss_classifier", "loss_box_reg"}
    assert stats["loss_total"] > stats["loss_classifier"]
    assert model.scale.item() < 1.0


def test_nonfinite_loss_names_component_and_image() -> None:
    model = FakeDetector(nan_loss=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(NonFiniteLossError, match="loss_classifier.*11"):
        train_one_epoch(model, [_batch()], optimizer, torch.device("cpu"))


def test_dry_run_performs_one_update() -> None:
    model = FakeDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.scale.item()

    result = dry_run(model, [_batch()], optimizer, torch.device("cpu"))

    assert result.batch_size == 1
    assert result.image_shapes == ((3, 8, 8),)
    assert result.target_counts == (1,)
    assert result.losses["loss_total"] > 0
    assert model.scale.item() != before


def test_training_supports_amp_and_gradient_clipping() -> None:
    model = FakeDetector()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    stats = train_one_epoch(
        model,
        [_batch()],
        optimizer,
        torch.device("cpu"),
        amp=True,
        grad_clip=0.1,
    )

    assert torch.isfinite(torch.tensor(stats["loss_total"]))
