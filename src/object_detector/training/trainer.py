from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import torch
from torch import nn

from object_detector.data.transforms import DetectionTarget


class NonFiniteLossError(RuntimeError):
    """Raised when a detector returns a NaN or infinite loss."""


@dataclass(frozen=True)
class DryRunResult:
    batch_size: int
    image_shapes: tuple[tuple[int, ...], ...]
    target_counts: tuple[int, ...]
    losses: dict[str, float]


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[tuple[list[torch.Tensor], list[DetectionTarget]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals: dict[str, float] = {}
    sample_count = 0
    for images, targets in loader:
        images, targets = move_batch(images, targets, device)
        optimizer.zero_grad(set_to_none=True)
        losses = model(images, targets)
        _validate_losses(losses, targets)
        total = sum_losses(losses)
        total.backward()
        optimizer.step()
        batch_size = len(images)
        sample_count += batch_size
        totals["loss_total"] = totals.get("loss_total", 0.0) + float(total.detach()) * batch_size
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach()) * batch_size
    if sample_count == 0:
        raise ValueError("training loader yielded no batches")
    return {name: value / sample_count for name, value in totals.items()}


def dry_run(
    model: nn.Module,
    loader: Iterable[tuple[list[torch.Tensor], list[DetectionTarget]]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> DryRunResult:
    model.train()
    iterator = iter(loader)
    try:
        images, targets = next(iterator)
    except StopIteration as exc:
        raise ValueError("dry run loader yielded no batches") from exc
    image_shapes = tuple(tuple(image.shape) for image in images)
    target_counts = tuple(int(target["boxes"].shape[0]) for target in targets)
    images, targets = move_batch(images, targets, device)
    optimizer.zero_grad(set_to_none=True)
    losses = model(images, targets)
    _validate_losses(losses, targets)
    total = sum_losses(losses)
    total.backward()
    optimizer.step()
    return DryRunResult(
        batch_size=len(images),
        image_shapes=image_shapes,
        target_counts=target_counts,
        losses={"loss_total": float(total.detach()), **{name: float(value.detach()) for name, value in losses.items()}},
    )


def move_batch(
    images: list[torch.Tensor],
    targets: list[DetectionTarget],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[DetectionTarget]]:
    return (
        [image.to(device) for image in images],
        [{key: value.to(device) for key, value in target.items()} for target in targets],
    )


def sum_losses(losses: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if not losses:
        raise ValueError("detector returned no training losses")
    return torch.stack(tuple(losses.values())).sum()


def _validate_losses(losses: Mapping[str, torch.Tensor], targets: list[DetectionTarget]) -> None:
    for name, value in losses.items():
        if value.numel() != 1 or not torch.isfinite(value).item():
            image_ids = [int(item["image_id"].flatten()[0]) for item in targets]
            raise NonFiniteLossError(f"non-finite {name} for image IDs {image_ids}")
