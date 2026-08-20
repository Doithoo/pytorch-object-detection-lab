import csv
import json
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

import object_detector.evaluation.evaluate as evaluation
from object_detector.config import AppConfig, DataConfig, ModelConfig, config_to_dict
from object_detector.data.dataset import VocDetectionDataset
from object_detector.training.checkpoint import EXPECTED_PREPROCESSING, CheckpointCompatibilityError, save_checkpoint
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


class CountingDetector(EmptyDetector):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, images):
        self.batch_sizes.append(len(images))
        return super().forward(images)


class StreamingDataset:
    def __init__(self, length: int = 4) -> None:
        self.length = length
        self.first_pass_calls = 0
        self.image_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        if self.first_pass_calls < self.length:
            assert sum(reference() is not None for reference in self.image_refs) <= 1, (
                "evaluation retained decoded images from earlier samples"
            )
            self.first_pass_calls += 1
        image = torch.zeros((3, 12, 12))
        self.image_refs.append(weakref.ref(image))
        target = {
            "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([index]),
            "area": torch.tensor([49.0]),
            "iscrowd": torch.tensor([0]),
            "difficult": torch.tensor([False]),
        }
        return image, target

    def source_image_id(self, index: int) -> str:
        return f"image-{index}"


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
        assert next(csv.reader(handle)) == ["class_id", "class_name", "map_50_95", "mar_100", "voc_ap_50_11"]
    with (output / "errors.csv").open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ["image_id", "kind", "class_name", "score", "iou", "box"]
    assert (output / "visualizations" / "summary.png").is_file()


def test_evaluation_does_not_retain_all_decoded_images(tmp_path: Path) -> None:
    dataset = StreamingDataset()

    result = evaluation.evaluate_model(
        EmptyDetector(),
        dataset,  # type: ignore[arg-type]
        ("background", "object"),
        torch.device("cpu"),
        tmp_path / "streaming-evaluation",
        score_threshold=0.5,
        error_score_threshold=0.5,
        error_iou_threshold=0.5,
    )

    assert result.metrics["image_count"] == len(dataset)


def test_evaluation_batches_model_inference(tmp_path: Path) -> None:
    dataset = StreamingDataset()
    model = CountingDetector()

    evaluation.evaluate_model(
        model,
        dataset,  # type: ignore[arg-type]
        ("background", "object"),
        torch.device("cpu"),
        tmp_path / "batched-evaluation",
        score_threshold=0.5,
        error_score_threshold=0.5,
        error_iou_threshold=0.5,
        batch_size=2,
    )

    assert model.batch_sizes == [2, 2]


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
        "preprocessing": dict(EXPECTED_PREPROCESSING),
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
    payload = json.loads((result.output_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert payload["manifest_identity"] == prepared_voc.metadata.identity


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
