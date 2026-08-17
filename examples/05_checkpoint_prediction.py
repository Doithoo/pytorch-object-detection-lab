from __future__ import annotations

import argparse
from pathlib import Path

from object_detector.inference.predictor import Predictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one local image from a self-contained checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/example_prediction"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    result = predictor.predict_single(
        args.image,
        args.output_dir,
        score_threshold=args.score_threshold,
        display_limit=20,
    )
    print(f"detections={len(result.detections)} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
