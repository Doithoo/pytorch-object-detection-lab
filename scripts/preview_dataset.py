from __future__ import annotations

import argparse
from pathlib import Path

from object_detector.data.dataset import VocDetectionDataset
from object_detector.data.inspection import render_detection_preview


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render Pascal VOC images and boxes")
    parser.add_argument("manifest_dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset_preview.png"))
    parser.add_argument("--limit", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = VocDetectionDataset.from_manifests(
        args.manifest_dir,
        args.split,
        data_dir=args.data_dir,
        training=False,
        limit=args.limit,
    )
    samples = [dataset[index] for index in range(len(dataset))]
    render_detection_preview(samples, ("background", *dataset.metadata.class_names), args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
