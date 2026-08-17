from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from object_detector import __version__
from object_detector.config import config_to_dict, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="detect")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command")

    show_config = subparsers.add_parser("show-config", help="show the resolved configuration")
    show_config.add_argument("--config", type=Path)
    show_config.add_argument("--set", dest="overrides", action="append", nargs=2, default=[], metavar=("KEY", "VALUE"))
    show_config.set_defaults(handler=_show_config)
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
