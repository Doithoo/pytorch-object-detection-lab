from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from object_detector.data.transforms import DetectionTarget


def inspect_prepared_data(
    manifest_dir: Path,
    *,
    split: str,
    data_dir: Path | None = None,
    limit: int = 16,
) -> dict[str, object]:
    if limit <= 0:
        raise ValueError("inspection limit must be positive")

    from object_detector.data.dataset import VocDetectionDataset
    from object_detector.data.manifest import load_dataset_metadata, read_manifest

    metadata = load_dataset_metadata(manifest_dir)
    rows = read_manifest(manifest_dir / f"{split}.csv")
    if not rows:
        raise ValueError(f"inspection split {split!r} is empty")
    dataset = VocDetectionDataset.from_manifests(
        manifest_dir,
        split,
        data_dir=data_dir,
        training=False,
        limit=min(limit, len(rows)),
    )

    ordinary_counts: Counter[str] = Counter()
    difficult_counts: Counter[str] = Counter()
    image_heights: list[int] = []
    image_widths: list[int] = []
    box_widths: list[float] = []
    box_heights: list[float] = []
    box_areas: list[float] = []
    ordinary_objects = 0
    difficult_objects = 0
    empty_images = 0
    images_with_difficult = 0
    class_names = ("background", *metadata.class_names)

    for index in range(len(dataset)):
        image, target = dataset[index]
        image_heights.append(int(image.shape[-2]))
        image_widths.append(int(image.shape[-1]))
        labels = target["labels"].tolist()
        difficult = target["difficult"].tolist()
        if not labels:
            empty_images += 1
        if any(difficult):
            images_with_difficult += 1
        for label, is_difficult in zip(labels, difficult, strict=True):
            class_name = class_names[label]
            if is_difficult:
                difficult_counts[class_name] += 1
                difficult_objects += 1
            else:
                ordinary_counts[class_name] += 1
                ordinary_objects += 1
        boxes = target["boxes"]
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        box_widths.extend(float(value) for value in widths)
        box_heights.extend(float(value) for value in heights)
        box_areas.extend(float(value) for value in target["area"])

    return {
        "dataset": metadata.name,
        "identity": metadata.identity,
        "split": split,
        "total_images": len(rows),
        "inspected_images": len(dataset),
        "ordinary_objects": ordinary_objects,
        "difficult_objects": difficult_objects,
        "empty_images": empty_images,
        "images_with_difficult": images_with_difficult,
        "class_counts": {
            "ordinary": dict(sorted(ordinary_counts.items())),
            "difficult": dict(sorted(difficult_counts.items())),
        },
        "image_size": {
            "min_height": min(image_heights),
            "max_height": max(image_heights),
            "min_width": min(image_widths),
            "max_width": max(image_widths),
        },
        "boxes": {
            "count": len(box_widths),
            "min_width": min(box_widths, default=None),
            "max_width": max(box_widths, default=None),
            "min_height": min(box_heights, default=None),
            "max_height": max(box_heights, default=None),
            "min_area": min(box_areas, default=None),
            "max_area": max(box_areas, default=None),
        },
    }


def render_detection_preview(
    samples: Sequence[tuple[torch.Tensor, DetectionTarget]],
    class_names: Sequence[str],
    output: Path,
    *,
    columns: int = 2,
) -> None:
    if not samples:
        raise ValueError("preview requires at least one sample")
    if columns <= 0:
        raise ValueError("preview columns must be positive")
    images = [_tensor_to_image(image) for image, _ in samples]
    cell_width = max(image.width for image in images)
    cell_height = max(image.height for image in images)
    rows = math.ceil(len(samples) / columns)
    canvas = Image.new("RGB", (cell_width * columns, cell_height * rows), "white")
    for index, ((_, target), image) in enumerate(zip(samples, images, strict=True)):
        annotated = image.copy()
        _draw_target(annotated, target, class_names)
        left = (index % columns) * cell_width
        top = (index // columns) * cell_height
        canvas.paste(annotated, (left, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def draw_detections(image: Image.Image, target: DetectionTarget, class_names: Sequence[str]) -> Image.Image:
    result = image.convert("RGB").copy()
    _draw_target(result, target, class_names)
    return result


def _draw_target(image: Image.Image, target: DetectionTarget, class_names: Sequence[str]) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, label, difficult in zip(
        target["boxes"].tolist(),
        target["labels"].tolist(),
        target["difficult"].tolist(),
        strict=True,
    ):
        color = "#d97706" if difficult else "#16a34a"
        if difficult:
            _dashed_rectangle(draw, tuple(box), fill=color, width=2)
        else:
            draw.rectangle(tuple(box), outline=color, width=2)
        class_name = class_names[label] if 0 <= label < len(class_names) else f"class-{label}"
        text = f"{class_name}{' difficult' if difficult else ''}"
        text_box = draw.textbbox((box[0], box[1]), text, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((box[0], box[1]), text, fill="white", font=font)


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    fill: str,
    width: int,
    dash: int = 4,
) -> None:
    left, top, right, bottom = box
    for start in range(int(left), int(right), dash * 2):
        draw.line((start, top, min(start + dash, right), top), fill=fill, width=width)
        draw.line((start, bottom, min(start + dash, right), bottom), fill=fill, width=width)
    for start in range(int(top), int(bottom), dash * 2):
        draw.line((left, start, left, min(start + dash, bottom)), fill=fill, width=width)
        draw.line((right, start, right, min(start + dash, bottom)), fill=fill, width=width)


def _tensor_to_image(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image [3,H,W], got {tuple(image.shape)}")
    array = image.detach().cpu().clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(np.asarray(array))
