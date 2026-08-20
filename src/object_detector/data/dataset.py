from __future__ import annotations

import hashlib
import json
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
        self._coco_annotations: dict[str, list[dict[str, object]]] = {}
        self._coco_categories: dict[int, str] = {}
        if metadata.annotation_format == "coco":
            if not self.rows:
                raise DatasetError("COCO split is empty and cannot identify its annotation file")
            self._load_coco_annotations(self.dataset_root / self.rows[0].annotation_path)

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
        if self.metadata.annotation_format == "coco":
            return self._get_coco_item(row, image_path)
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

    def _load_coco_annotations(self, path: Path) -> None:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot read COCO annotations {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise DatasetError(f"COCO annotations {path} must contain a mapping")
        categories = raw.get("categories")
        annotations = raw.get("annotations")
        if not isinstance(categories, list) or not isinstance(annotations, list):
            raise DatasetError(f"COCO annotations {path} are missing categories or annotations")
        for category in categories:
            if (
                isinstance(category, dict)
                and isinstance(category.get("id"), int)
                and isinstance(category.get("name"), str)
            ):
                self._coco_categories[category["id"]] = category["name"]
        for annotation in annotations:
            if isinstance(annotation, dict) and isinstance(annotation.get("image_id"), int):
                self._coco_annotations.setdefault(str(annotation["image_id"]), []).append(annotation)

    def _get_coco_item(self, row: ManifestRow, image_path: Path) -> tuple[torch.Tensor, DetectionTarget]:
        try:
            with Image.open(image_path) as source:
                image = pil_to_tensor(source.convert("RGB")).float().div(255.0)
        except OSError as exc:
            raise DatasetError(f"cannot read image {image_path}: {exc}") from exc
        width, height = image.shape[-1], image.shape[-2]
        boxes: list[list[float]] = []
        labels: list[int] = []
        difficult: list[bool] = []
        for annotation in self._coco_annotations.get(row.image_id, []):
            category_id = annotation.get("category_id")
            category_name = self._coco_categories.get(category_id) if isinstance(category_id, int) else None
            bbox = annotation.get("bbox")
            if category_name is None or not isinstance(bbox, list) or len(bbox) != 4:
                raise DatasetError(f"invalid COCO annotation for image {row.image_id}")
            x, y, box_width, box_height = (float(value) for value in bbox)
            boxes.append([max(x, 0.0), max(y, 0.0), min(x + box_width, width), min(y + box_height, height)])
            labels.append(self.metadata.label_by_name[category_name])
            difficult.append(bool(annotation.get("iscrowd", 0)))
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        difficult_tensor = torch.tensor(difficult, dtype=torch.bool)
        target: DetectionTarget = {
            "boxes": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([_numeric_image_id(row.image_id)], dtype=torch.int64),
            "area": (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1]),
            "iscrowd": difficult_tensor.to(dtype=torch.int64),
            "difficult": difficult_tensor,
        }
        if self.training:
            target = select_objects(target, ~difficult_tensor)
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
