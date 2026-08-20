from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

from object_detector.data.manifest import DatasetMetadata, ManifestRow, load_dataset_metadata, read_manifest
from object_detector.data.transforms import (
    Compose,
    DetectionTarget,
    DetectionTransform,
    RandomHorizontalFlip,
    filter_degenerate_boxes,
    select_objects,
)
from object_detector.data.voc import parse_voc_annotation


class DatasetError(RuntimeError):
    """Raised when a prepared dataset no longer matches its annotations."""


class VocDetectionDataset(Dataset[tuple[torch.Tensor, DetectionTarget]]):
    def __init__(
        self,
        rows: Sequence[ManifestRow],
        dataset_root: Path,
        metadata: DatasetMetadata,
        *,
        training: bool,
        transforms: DetectionTransform | None = None,
    ) -> None:
        self.rows = tuple(rows)
        self.dataset_root = dataset_root
        self.metadata = metadata
        self.training = training
        self.transforms = transforms

    @classmethod
    def from_manifests(
        cls,
        manifest_dir: Path,
        split: str,
        *,
        data_dir: Path | None = None,
        training: bool,
        transforms: DetectionTransform | None = None,
        limit: int | None = None,
    ) -> VocDetectionDataset:
        if split not in {"train", "valid", "test"}:
            raise DatasetError(f"unknown split {split!r}")
        metadata = load_dataset_metadata(manifest_dir)
        rows = read_manifest(manifest_dir / f"{split}.csv")
        if limit is not None:
            if limit <= 0:
                raise DatasetError("sample limit must be positive")
            rows = rows[:limit]
        resolved_data_dir = data_dir if data_dir is not None else manifest_dir.parent / "raw"
        return cls(
            rows,
            resolved_data_dir / metadata.dataset_root,
            metadata,
            training=training,
            transforms=transforms,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, DetectionTarget]:
        row = self.rows[index]
        image_path = self.dataset_root / row.image_path
        annotation = parse_voc_annotation(
            self.dataset_root / row.annotation_path,
            allowed_classes=self.metadata.class_names,
        )
        try:
            with Image.open(image_path) as source:
                image = pil_to_tensor(source.convert("RGB")).float().div(255.0)
        except OSError as exc:
            raise DatasetError(f"cannot read image {image_path}: {exc}") from exc
        if (image.shape[-1], image.shape[-2]) != annotation.size:
            raise DatasetError(
                f"{image_path}: image size {(image.shape[-1], image.shape[-2])} "
                f"does not match annotation size {annotation.size}"
            )

        boxes = torch.tensor([item.box for item in annotation.objects], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(
            [self.metadata.label_by_name[item.class_name] for item in annotation.objects],
            dtype=torch.int64,
        )
        difficult = torch.tensor([item.difficult for item in annotation.objects], dtype=torch.bool)
        target: DetectionTarget = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([_numeric_image_id(row.image_id)], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": difficult.to(dtype=torch.int64),
            "difficult": difficult,
        }
        if self.training:
            target = select_objects(target, ~difficult)
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        target, _ = filter_degenerate_boxes(target)
        return image, target

    def source_image_id(self, index: int) -> str:
        return self.rows[index].image_id


def build_detection_transforms(horizontal_flip: float, *, training: bool) -> DetectionTransform:
    transforms: list[DetectionTransform] = []
    if training and horizontal_flip > 0.0:
        transforms.append(RandomHorizontalFlip(horizontal_flip))
    return Compose(transforms)


def detection_collate(
    batch: Sequence[tuple[torch.Tensor, DetectionTarget]],
) -> tuple[list[torch.Tensor], list[DetectionTarget]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def _numeric_image_id(image_id: str) -> int:
    digest = hashlib.sha256(image_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
