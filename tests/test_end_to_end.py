import csv
import json
import socket
from dataclasses import replace
from pathlib import Path

import torch
import yaml

from object_detector.config import load_config
from object_detector.data.manifest import prepare_voc2007
from object_detector.evaluation.evaluate import evaluate_checkpoint
from object_detector.inference.predictor import Predictor
from object_detector.training.checkpoint import load_checkpoint
from object_detector.training.train import run_training
from tests.fixtures.models import FakeDetector
from tests.fixtures.voc import build_voc_tree


def _fake_model_factory(*args, **kwargs):
    return FakeDetector()


def test_offline_workflow(tmp_path: Path, monkeypatch) -> None:
    def reject_network(self, address):
        raise AssertionError(f"unexpected network access: {address}")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    data_dir = tmp_path / "raw"
    voc_root = build_voc_tree(data_dir)
    manifest_dir = tmp_path / "manifests"
    metadata = prepare_voc2007(data_dir, manifest_dir, expected_split_counts=None)
    config = load_config(
        Path("configs/learning_minimal.yaml"),
        [("train.epochs", "1"), ("train.batch_size", "1"), ("data.horizontal_flip", "0.0")],
    )
    config = replace(
        config,
        data=replace(
            config.data,
            data_dir=data_dir,
            manifest_dir=manifest_dir,
            max_train_samples=2,
            max_valid_samples=1,
            max_test_samples=1,
        ),
        model=replace(config.model, name="fake", expected_num_classes=len(metadata.class_names) + 1),
        device="cpu",
        output_dir=tmp_path / "artifacts",
        run_name="acceptance",
    )

    dry_run = run_training(config, dry_run_mode=True, model_factory=_fake_model_factory)
    assert dry_run.dry_run_result is not None
    print("dry-run OK")
    first = run_training(config, model_factory=_fake_model_factory)
    resumed = run_training(
        replace(config, train=replace(config.train, epochs=2)),
        resume=first.run_dir / "last.pt",
        model_factory=_fake_model_factory,
    )

    assert {path.name for path in resumed.run_dir.iterdir()} == {
        "best.pt",
        "config.yaml",
        "last.pt",
        "metrics.csv",
        "run.yaml",
    }
    with (resumed.run_dir / "metrics.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["epoch"]) for row in rows] == [1, 2]
    run_metadata = yaml.safe_load((resumed.run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert run_metadata["manifest_identity"] == metadata.identity
    for checkpoint_name in ("best.pt", "last.pt"):
        checkpoint = load_checkpoint(resumed.run_dir / checkpoint_name)
        assert checkpoint["manifest_identity"] == metadata.identity

    evaluation = evaluate_checkpoint(
        resumed.run_dir / "best.pt",
        split="test",
        output_dir=tmp_path / "evaluation",
        device="cpu",
        score_threshold=0.5,
        overwrite=False,
        model_factory=_fake_model_factory,
    )
    assert {path.name for path in evaluation.output_dir.iterdir()} == {
        "errors.csv",
        "evaluation.json",
        "per_class.csv",
        "predictions.json",
        "visualizations",
    }
    evaluation_payload = json.loads((evaluation.output_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation_payload["manifest_identity"] == metadata.identity

    predictor = Predictor.from_checkpoint(
        resumed.run_dir / "best.pt",
        device="cpu",
        model_factory=_fake_model_factory,
    )
    image = voc_root / "JPEGImages" / "test-1.jpg"
    predictor.predict_single(image, tmp_path / "prediction", score_threshold=0.5, display_limit=10)
    assert {path.name for path in (tmp_path / "prediction").iterdir()} == {"test-1.json", "test-1.png"}
    prediction_payload = json.loads((tmp_path / "prediction" / "test-1.json").read_text(encoding="utf-8"))
    assert prediction_payload["manifest_identity"] == metadata.identity
    assert prediction_payload["detections"]
    assert torch.equal(load_checkpoint(resumed.run_dir / "best.pt")["model_state"]["scale"], predictor.model.scale)
