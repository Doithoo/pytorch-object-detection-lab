from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from object_detector import __version__
from object_detector.config import config_to_dict, load_config
from object_detector.data.manifest import VOC2007_SPLIT_COUNTS, prepare_voc2007


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
