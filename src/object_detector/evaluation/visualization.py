from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

TARGET_COLOR = "#16a34a"
DIFFICULT_COLOR = "#d97706"
PREDICTION_COLOR = "#2563eb"
LEGEND_HEIGHT = 16


def render_detection_evidence(
    image: torch.Tensor,
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
    output: Path,
    *,
    score_threshold: float,
) -> None:
    source = _tensor_to_image(image)
    canvas = Image.new("RGB", (source.width, source.height + LEGEND_HEIGHT), "white")
    canvas.paste(source, (0, LEGEND_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((2, 2), "target", fill=TARGET_COLOR, font=font)
    draw.text((43, 2), "difficult", fill=DIFFICULT_COLOR, font=font)
    draw.text((101, 2), "prediction", fill=PREDICTION_COLOR, font=font)

    target_labels = target["labels"].detach().cpu()
    iscrowd = target.get("iscrowd", torch.zeros_like(target_labels)).detach().cpu()
    for box, label, difficult in zip(
        target["boxes"].detach().cpu().tolist(),
        target_labels.tolist(),
        iscrowd.tolist(),
        strict=True,
    ):
        shifted = _shift_box(box)
        color = DIFFICULT_COLOR if difficult else TARGET_COLOR
        if difficult:
            _dashed_rectangle(draw, shifted, fill=color)
        else:
            draw.rectangle(shifted, outline=color, width=2)
        _draw_label(draw, shifted, _class_name(int(label), class_names), color, font)

    for box, label, score in zip(
        prediction["boxes"].detach().cpu().tolist(),
        prediction["labels"].detach().cpu().tolist(),
        prediction["scores"].detach().cpu().tolist(),
        strict=True,
    ):
        if score < score_threshold:
            continue
        shifted = _shift_box(box)
        draw.rectangle(shifted, outline=PREDICTION_COLOR, width=2)
        _draw_label(
            draw,
            shifted,
            f"{_class_name(int(label), class_names)} {float(score):.2f}",
            PREDICTION_COLOR,
            font,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _tensor_to_image(image: torch.Tensor) -> Image.Image:
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected image [3,H,W], got {tuple(image.shape)}")
    array = image.detach().cpu().clamp(0.0, 1.0).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(np.asarray(array))


def _shift_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    return float(box[0]), float(box[1]) + LEGEND_HEIGHT, float(box[2]), float(box[3]) + LEGEND_HEIGHT


def _draw_label(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    text: str,
    color: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    text_box = draw.textbbox((box[0], box[1]), text, font=font)
    draw.rectangle(text_box, fill=color)
    draw.text((box[0], box[1]), text, fill="white", font=font)


def _dashed_rectangle(
    draw: ImageDraw.ImageDraw,
    box: tuple[float, float, float, float],
    *,
    fill: str,
    width: int = 2,
    dash: int = 4,
) -> None:
    left, top, right, bottom = box
    for start in range(int(left), int(right), dash * 2):
        draw.line((start, top, min(start + dash, right), top), fill=fill, width=width)
        draw.line((start, bottom, min(start + dash, right), bottom), fill=fill, width=width)
    for start in range(int(top), int(bottom), dash * 2):
        draw.line((left, start, left, min(start + dash, bottom)), fill=fill, width=width)
        draw.line((right, start, right, min(start + dash, bottom)), fill=fill, width=width)


def _class_name(class_id: int, class_names: Sequence[str]) -> str:
    if class_id < 0 or class_id >= len(class_names):
        return f"class-{class_id}"
    return class_names[class_id]
