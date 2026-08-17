from pathlib import Path

import pytest

from object_detector.config import ConfigError, config_to_dict, load_config


def test_yaml_then_cli_override_precedence(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("train:\n  epochs: 3\n", encoding="utf-8")

    config = load_config(path, [("train.epochs", "5"), ("data.num_workers", "0")])

    assert config.train.epochs == 5
    assert config.data.num_workers == 0
    assert config.model.name == "fasterrcnn_mobilenet_v3_large_320_fpn"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  epochz: 3\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="train.epochz"):
        load_config(path)


def test_paths_are_serialized_for_yaml() -> None:
    serialized = config_to_dict(load_config())

    assert serialized["data"]["data_dir"] == "data/raw"
    assert serialized["output_dir"] == "artifacts"
