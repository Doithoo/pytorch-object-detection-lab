from pathlib import Path

import tomllib

from object_detector import __version__
from object_detector.cli import build_parser


def test_version_and_console_name() -> None:
    assert __version__ == "0.1.0"
    parser = build_parser()
    assert parser.prog == "detect"


def test_publication_metadata_and_configs_are_declared() -> None:
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["urls"]["Source"] == "https://github.com/Yashowhoo/pytorch-object-detection-lab"
    packaged_configs = metadata["tool"]["setuptools"]["data-files"]["share/object-detector/configs"]
    assert {Path(path).name for path in packaged_configs} == {
        "fasterrcnn_resnet50_fpn.yaml",
        "learning_minimal.yaml",
        "reference_fasterrcnn.yaml",
        "ssdlite320_mobilenet_v3.yaml",
    }
