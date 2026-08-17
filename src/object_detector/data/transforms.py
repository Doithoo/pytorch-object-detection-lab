from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import torch
from torchvision.transforms import ColorJitter as TorchColorJitter
from torchvision.transforms import functional as F

DetectionTarget: TypeAlias = dict[str, torch.Tensor]
DetectionTransform: TypeAlias = Callable[
    [torch.Tensor, DetectionTarget],
    tuple[torch.Tensor, DetectionTarget],
]

_OBJECT_FIELDS = ("boxes", "labels", "area", "iscrowd", "difficult")


class Compose:
    def __init__(self, transforms: Sequence[DetectionTransform]) -> None:
        self.transforms = tuple(transforms)

    def __call__(self, image: torch.Tensor, target: DetectionTarget) -> tuple[torch.Tensor, DetectionTarget]:
        for transform in self.transforms:
            image, target = transform(image, target)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, probability: float) -> None:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("horizontal flip probability must be between 0 and 1")
        self.probability = probability

    def __call__(self, image: torch.Tensor, target: DetectionTarget) -> tuple[torch.Tensor, DetectionTarget]:
        if torch.rand(()) >= self.probability:
            return image, target
        result = _clone_target(target)
        width = image.shape[-1]
        boxes = result["boxes"]
        if boxes.numel():
            xmin = width - boxes[:, 2]
            xmax = width - boxes[:, 0]
            boxes[:, 0] = xmin
            boxes[:, 2] = xmax
        return F.hflip(image), result


class ColorJitter:
    def __init__(self, brightness: float = 0.1, contrast: float = 0.1, saturation: float = 0.1) -> None:
        self.transform = TorchColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)

    def __call__(self, image: torch.Tensor, target: DetectionTarget) -> tuple[torch.Tensor, DetectionTarget]:
        return self.transform(image), target


def filter_degenerate_boxes(target: DetectionTarget) -> tuple[DetectionTarget, int]:
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return _clone_target(target), 0
    keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    result = _clone_target(target)
    for field in _OBJECT_FIELDS:
        result[field] = result[field][keep]
    return result, int((~keep).sum().item())


def select_objects(target: DetectionTarget, keep: torch.Tensor) -> DetectionTarget:
    result = _clone_target(target)
    for field in _OBJECT_FIELDS:
        result[field] = result[field][keep]
    return result


def _clone_target(target: DetectionTarget) -> DetectionTarget:
    return {key: value.clone() for key, value in target.items()}
