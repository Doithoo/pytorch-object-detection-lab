from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

from object_detector.config import AppConfig, load_config
from object_detector.training.train import run_training
from tests.conftest import PreparedVoc
from tests.fixtures.models import FakeDetector


def _config(prepared_voc: PreparedVoc, tmp_path: Path, epochs: int) -> AppConfig:
    config = load_config()
    return replace(
        config,
        data=replace(
            config.data,
            data_dir=prepared_voc.voc_root.parent.parent,
            manifest_dir=prepared_voc.manifests,
            horizontal_flip=0.0,
            max_train_samples=2,
            max_valid_samples=1,
            max_test_samples=1,
        ),
        model=replace(config.model, name="fake", weights="none"),
        train=replace(config.train, epochs=epochs, batch_size=1, lr=0.1),
        device="cpu",
        output_dir=tmp_path / "artifacts",
        run_name="test-run",
    )


def _fake_model_factory(*args, **kwargs):
    return FakeDetector()


def test_training_writes_and_resumes_complete_artifacts(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    first = run_training(_config(prepared_voc, tmp_path, 2), model_factory=_fake_model_factory)

    expected = {"config.yaml", "run.yaml", "metrics.csv", "last.pt", "best.pt"}
    assert expected <= {path.name for path in first.run_dir.iterdir()}
    with (first.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2

    resumed = run_training(
        _config(prepared_voc, tmp_path, 3),
        resume=first.run_dir / "last.pt",
        model_factory=_fake_model_factory,
    )

    with (resumed.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
