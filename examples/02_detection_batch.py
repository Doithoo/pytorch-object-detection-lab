from __future__ import annotations

import argparse

import torch

from object_detector.data.dataset import detection_collate


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Collate variable-size images and target dictionaries")


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    sample_a = (torch.zeros((3, 16, 20)), _target(1))
    sample_b = (torch.zeros((3, 12, 24)), _target(2))
    images, targets = detection_collate([sample_a, sample_b])
    print(f"image_shapes={[tuple(image.shape) for image in images]}")
    print(f"target_counts={[int(target['boxes'].shape[0]) for target in targets]}")
    return 0


def _target(count: int) -> dict[str, torch.Tensor]:
    boxes = torch.tensor([[1.0, 1.0, 8.0, 8.0]] * count)
    return {
        "boxes": boxes,
        "labels": torch.ones(count, dtype=torch.int64),
        "image_id": torch.tensor([count], dtype=torch.int64),
        "area": torch.full((count,), 49.0),
        "iscrowd": torch.zeros(count, dtype=torch.int64),
        "difficult": torch.zeros(count, dtype=torch.bool),
    }


if __name__ == "__main__":
    raise SystemExit(main())
