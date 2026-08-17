from __future__ import annotations

from dataclasses import dataclass

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


@dataclass(frozen=True)
class VocObject:
    class_name: str
    box: tuple[float, float, float, float]
    difficult: bool


@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...]

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height
