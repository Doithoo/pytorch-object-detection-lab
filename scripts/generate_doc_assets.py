from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from object_detector.data.inspection import draw_detections
from object_detector.data.transforms import DetectionTarget
from object_detector.evaluation.visualization import render_detection_evidence

WIDTH = 640
HEIGHT = 360
HEADER_HEIGHT = 40
FOOTER_HEIGHT = 32
CLASS_NAMES = ("background", "person", "dog")


def generate_assets(output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = _synthetic_image()
    target = _target()

    target_render = draw_detections(to_pil_image(image), target, CLASS_NAMES)
    target_output = output_dir / "detection-target-anatomy.png"
    _frame(
        target_render,
        title="Detection target anatomy",
        footer="Synthetic teaching diagram: green = ordinary target, orange dashed = difficult target",
    ).save(target_output)

    prediction = {
        "boxes": torch.tensor(
            [[88.0, 76.0, 292.0, 306.0], [388.0, 82.0, 584.0, 302.0], [300.0, 190.0, 430.0, 330.0]],
            dtype=torch.float32,
        ),
        "labels": torch.tensor([1, 1, 2], dtype=torch.int64),
        "scores": torch.tensor([0.94, 0.81, 0.72], dtype=torch.float32),
    }
    with tempfile.TemporaryDirectory(prefix="object-detector-doc-assets-") as temporary_dir:
        evidence_path = Path(temporary_dir) / "evidence.png"
        render_detection_evidence(
            image,
            prediction,
            target,
            CLASS_NAMES,
            evidence_path,
            score_threshold=0.5,
        )
        with Image.open(evidence_path) as evidence:
            framed_evidence = _frame(
                evidence.convert("RGB"),
                title="Reading detection errors",
                footer="Synthetic teaching diagram: compare target, difficult, and prediction boxes",
            )
    evidence_output = output_dir / "detection-error-analysis.png"
    framed_evidence.save(evidence_output)
    return target_output, evidence_output


def _synthetic_image() -> torch.Tensor:
    vertical = torch.linspace(0.0, 1.0, HEIGHT).view(HEIGHT, 1).expand(HEIGHT, WIDTH)
    horizontal = torch.linspace(0.0, 1.0, WIDTH).view(1, WIDTH).expand(HEIGHT, WIDTH)
    red = 0.12 + 0.52 * horizontal
    green = 0.18 + 0.46 * vertical
    blue = 0.58 + 0.18 * (1.0 - horizontal) + 0.08 * vertical
    return torch.stack((red, green, blue)).clamp(0.0, 1.0)


def _target() -> DetectionTarget:
    boxes = torch.tensor([[72.0, 64.0, 286.0, 310.0], [372.0, 72.0, 590.0, 310.0]], dtype=torch.float32)
    difficult = torch.tensor([False, True])
    return {
        "boxes": boxes,
        "labels": torch.tensor([1, 2], dtype=torch.int64),
        "image_id": torch.tensor([1], dtype=torch.int64),
        "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
        "iscrowd": difficult.to(dtype=torch.int64),
        "difficult": difficult,
    }


def _frame(image: Image.Image, *, title: str, footer: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + HEADER_HEIGHT + FOOTER_HEIGHT), "white")
    canvas.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((16, 14), title, fill="#111827", font=font)
    draw.text((16, image.height + HEADER_HEIGHT + 10), footer, fill="#374151", font=font)
    return canvas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic documentation images")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for output in generate_assets(args.output_dir):
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
