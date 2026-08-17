from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import yaml

from object_detector import __version__
from object_detector.config import config_to_dict, load_config
from object_detector.data.manifest import VOC2007_SPLIT_COUNTS, prepare_voc2007
from object_detector.evaluation.evaluate import evaluate_checkpoint
from object_detector.inference.predictor import Predictor
from object_detector.training.train import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detect")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    show_config = subparsers.add_parser("show-config", help="show the resolved configuration")
    show_config.add_argument("--config", type=Path)
    show_config.add_argument("--set", dest="overrides", action="append", nargs=2, default=[], metavar=("KEY", "VALUE"))
    show_config.set_defaults(handler=_show_config)

    prepare_data = subparsers.add_parser("prepare-data", help="validate VOC 2007 and write fixed manifests")
    prepare_data.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    prepare_data.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    prepare_data.add_argument("--allow-nonstandard-counts", action="store_true")
    prepare_data.set_defaults(handler=_prepare_data)

    train = subparsers.add_parser("train", help="train an object detector")
    train.add_argument("--config", type=Path)
    train.add_argument("--set", dest="overrides", action="append", nargs=2, default=[], metavar=("KEY", "VALUE"))
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--resume", type=Path)
    train.add_argument("--device")
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("train", "valid", "test"), default="test")
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--score-threshold", type=float, default=0.05)
    evaluate.add_argument("--overwrite", action="store_true")
    evaluate.set_defaults(handler=_evaluate)

    predict = subparsers.add_parser("predict", help="predict from a checkpoint")
    predict.add_argument("--checkpoint", type=Path, required=True)
    input_mode = predict.add_mutually_exclusive_group(required=True)
    input_mode.add_argument("--image", type=Path)
    input_mode.add_argument("--input-dir", type=Path)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--score-threshold", type=float, default=0.5)
    predict.add_argument("--display-limit", type=int, default=20)
    predict.add_argument("--overwrite", action="store_true")
    predict.set_defaults(handler=_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 0
    return int(handler(args))


def _show_config(args: argparse.Namespace) -> int:
    config = load_config(args.config, [tuple(override) for override in args.overrides])
    print(yaml.safe_dump(config_to_dict(config), sort_keys=False), end="")
    return 0


def _prepare_data(args: argparse.Namespace) -> int:
    expected_counts = None if args.allow_nonstandard_counts else VOC2007_SPLIT_COUNTS
    metadata = prepare_voc2007(args.data_dir, args.manifest_dir, expected_split_counts=expected_counts)
    counts = metadata.split_counts
    print(f"identity={metadata.identity}")
    print(f"train={counts['train']} valid={counts['valid']} test={counts['test']}")
    return 0


def _train(args: argparse.Namespace) -> int:
    config = load_config(args.config, [tuple(override) for override in args.overrides])
    if args.device is not None:
        config = replace(config, device=args.device)
    result = run_training(config, resume=args.resume, dry_run_mode=args.dry_run)
    if result.dry_run_result is not None:
        diagnostics = result.dry_run_result
        print(f"image_shapes={diagnostics.image_shapes}")
        print(f"target_counts={diagnostics.target_counts}")
        for name, value in diagnostics.losses.items():
            print(f"{name}={value}")
        print("dry-run OK")
    else:
        print(result.run_dir)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    result = evaluate_checkpoint(
        args.checkpoint,
        split=args.split,
        output_dir=args.output_dir,
        device=args.device,
        score_threshold=args.score_threshold,
        overwrite=args.overwrite,
    )
    print(result.output_dir)
    return 0


def _predict(args: argparse.Namespace) -> int:
    predictor = Predictor.from_checkpoint(args.checkpoint, device=args.device)
    options = {
        "score_threshold": args.score_threshold,
        "display_limit": args.display_limit,
        "overwrite": args.overwrite,
    }
    if args.image is not None:
        predictor.predict_single(args.image, args.output_dir, **options)
    else:
        predictor.predict_directory(args.input_dir, args.output_dir, **options)
    print(args.output_dir)
    return 0
