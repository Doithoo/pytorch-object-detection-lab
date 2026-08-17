from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from object_detector.data.transforms import DetectionTarget


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
