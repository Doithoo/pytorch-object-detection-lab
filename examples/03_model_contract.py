from __future__ import annotations

import argparse

import torch

from object_detector.models.registry import build_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the real torchvision detector mode-dependent contract")
    parser.add_argument("--image-size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.image_size < 32:
        raise ValueError("image size must be at least 32")

    torch.manual_seed(0)
    model = build_model("fasterrcnn_mobilenet_v3_large_320_fpn", num_classes=2, weights="none")
    image = torch.rand((3, args.image_size, args.image_size))
    target = {
        "boxes": torch.tensor([[4.0, 4.0, float(args.image_size - 4), float(args.image_size - 4)]]),
        "labels": torch.tensor([1], dtype=torch.int64),
    }

    model.train()
    losses = model([image], [target])
    print(f"training_losses={sorted(losses)}")

    model.eval()
    with torch.inference_mode():
        prediction = model([image])[0]
    print(f"evaluation_outputs={sorted(prediction)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
