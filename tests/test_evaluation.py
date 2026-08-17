import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import object_detector.evaluation.evaluate as evaluation
from object_detector.config import AppConfig, DataConfig, ModelConfig, config_to_dict
from object_detector.data.dataset import VocDetectionDataset
from object_detector.training.checkpoint import CheckpointCompatibilityError, save_checkpoint
from tests.conftest import PreparedVoc


class EmptyDetector(nn.Module):
    def forward(self, images):
        return [
            {
                "boxes": torch.empty((0, 4)),
                "labels": torch.empty((0,), dtype=torch.int64),
                "scores": torch.empty((0,)),
            }
            for _ in images
        ]


def test_empty_evaluation_writes_complete_artifact_set(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    dataset = VocDetectionDataset.from_manifests(prepared_voc.manifests, "valid", training=False)
    output = tmp_path / "evaluation"

    result = evaluation.evaluate_model(
        EmptyDetector(),
        dataset,
        ("background", *prepared_voc.metadata.class_names),
        torch.device("cpu"),
        output,
        score_threshold=0.5,
        error_score_threshold=0.5,
        error_iou_threshold=0.5,
    )

    assert result.output_dir == output
    assert result.metrics["map_50_95"] == 0.0
    assert result.metrics["image_count"] == 1
    assert {path.name for path in output.iterdir()} == {
        "evaluation.json",
        "per_class.csv",
        "predictions.json",
        "errors.csv",
        "visualizations",
    }
    payload = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["metrics"]["prediction_count"] == 0
    assert set(payload["backend_versions"]) == {"torchmetrics", "pycocotools"}
    assert json.loads((output / "predictions.json").read_text(encoding="utf-8"))[0]["predictions"] == []
    with (output / "per_class.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ["class_id", "class_name", "map_50_95", "mar_100"]
    with (output / "errors.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ["image_id", "kind", "class_name", "score", "iou", "box"]
    assert (output / "visualizations" / "summary.png").is_file()


def _checkpoint_payload(prepared_voc: PreparedVoc, *, manifest_identity: str | None = None):
    class_names = ("background", *prepared_voc.metadata.class_names)
    config = AppConfig(
        data=DataConfig(
            data_dir=prepared_voc.voc_root.parent.parent,
            manifest_dir=prepared_voc.manifests,
            max_valid_samples=1,
        ),
        model=ModelConfig(name="fake", expected_num_classes=len(class_names)),
    )
    return {
        "schema_version": 1,
        "config": config_to_dict(config),
        "model": {"name": "fake", "params": {}},
        "class_names": list(class_names),
        "preprocessing": {"input_range": [0.0, 1.0]},
        "manifest_identity": manifest_identity or prepared_voc.metadata.identity,
        "model_state": {},
    }


def test_checkpoint_evaluation_reconstructs_model_and_dataset(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, _checkpoint_payload(prepared_voc))

    result = evaluation.evaluate_checkpoint(
        checkpoint,
        split="valid",
        output_dir=tmp_path / "from-checkpoint",
        device="cpu",
        score_threshold=0.5,
        overwrite=False,
        model_factory=lambda name, num_classes, weights, params: EmptyDetector(),
    )

    assert result.metrics["image_count"] == 1
    assert (result.output_dir / "evaluation.json").is_file()


def test_checkpoint_evaluation_rejects_manifest_mismatch_before_model_build(
    prepared_voc: PreparedVoc, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, _checkpoint_payload(prepared_voc, manifest_identity="wrong"))
    model_built = False

    def model_factory(name, num_classes, weights, params):
        nonlocal model_built
        model_built = True
        return EmptyDetector()

    with pytest.raises(CheckpointCompatibilityError, match="manifest_identity"):
        evaluation.evaluate_checkpoint(
            checkpoint,
            split="valid",
            output_dir=tmp_path / "mismatch",
            device="cpu",
            score_threshold=0.5,
            overwrite=False,
            model_factory=model_factory,
        )

    assert model_built is False
