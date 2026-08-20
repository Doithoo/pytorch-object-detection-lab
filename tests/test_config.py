from pathlib import Path

import pytest

from object_detector.config import ConfigError, config_to_dict, load_config, load_config_with_sources


def test_yaml_then_cli_override_precedence(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text("train:\n  epochs: 3\n", encoding="utf-8")

    config = load_config(path, [("train.epochs", "5"), ("data.num_workers", "0")])

    assert config.train.epochs == 5
    assert config.data.num_workers == 0
    assert config.model.name == "fasterrcnn_mobilenet_v3_large_320_fpn"


def test_cli_can_set_a_model_parameter_and_report_its_source() -> None:
    config, sources = load_config_with_sources(overrides=[("model.params.min_size", "320")])

    assert config.model.params == {"min_size": 320}
    assert sources["model.params.min_size"] == "cli"


def test_config_accepts_external_model_factory() -> None:
    config = load_config(overrides=[("model.factory", "tests.fixtures.models:build_external_detector")])

    assert config.model.factory == "tests.fixtures.models:build_external_detector"
    assert config_to_dict(config)["model"]["factory"] == "tests.fixtures.models:build_external_detector"


def test_unknown_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  epochz: 3\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="train.epochz"):
        load_config(path)


def test_unsupported_metric_controls_are_rejected(tmp_path: Path) -> None:
    assert load_config(overrides=[("train.best_metric", "voc_map_50_11")]).train.best_metric == "voc_map_50_11"

    best_metric = tmp_path / "best.yaml"
    best_metric.write_text("train:\n  best_metric: map_50\n", encoding="utf-8")
    max_detections = tmp_path / "max.yaml"
    max_detections.write_text("evaluation:\n  max_detections: 10\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="train.best_metric"):
        load_config(best_metric)
    with pytest.raises(ConfigError, match="evaluation.max_detections"):
        load_config(max_detections)


def test_paths_are_serialized_for_yaml() -> None:
    serialized = config_to_dict(load_config())

    assert serialized["data"]["data_dir"] == "data/raw"
    assert serialized["output_dir"] == "artifacts"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("train.lr", ".nan", "finite"),
        ("train.weight_decay", ".inf", "finite"),
        ("train.optimizer", "rmsprop", "adamw.*sgd"),
        ("train.scheduler", "cosine", "none.*step"),
        ("data.name", "custom", "voc2007"),
        ("model.name", '""', "must not be empty"),
        ("device", '""', "must not be empty"),
        ("run_name", '""', "must not be empty"),
    ],
)
def test_invalid_runtime_contract_values_are_rejected(key: str, value: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=[(key, value)])


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_train_seed_outside_numpy_range_is_rejected(seed: int) -> None:
    with pytest.raises(ConfigError, match=r"train\.seed must be between 0 and 4294967295"):
        load_config(overrides=[("train.seed", str(seed))])


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_train_seed_accepts_numpy_range_boundaries(seed: int) -> None:
    assert load_config(overrides=[("train.seed", str(seed))]).train.seed == seed
