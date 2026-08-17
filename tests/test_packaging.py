from object_detector import __version__
from object_detector.cli import build_parser


def test_version_and_console_name() -> None:
    assert __version__ == "0.1.0"
    parser = build_parser()
    assert parser.prog == "detect"
