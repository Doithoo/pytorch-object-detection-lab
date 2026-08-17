import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import object_detector.inference.predictor as predictor_module
from object_detector.training.checkpoint import save_checkpoint


class FakeDetector(nn.Module):
    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[1.0, 2.0, 9.5, 8.5]], device=image.device),
                "labels": torch.tensor([2], device=image.device),
                "scores": torch.tensor([0.875], device=image.device),
            }
            for image in images
        ]


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        {
            "schema_version": 1,
            "model": {"name": "fake-detector", "params": {"size": 320}},
            "class_names": ["background", "cat", "dog"],
            "model_state": {},
        },
    )
    return path


def _build_predictor(checkpoint: Path):
    captured = {}

    def factory(name, num_classes, weights, params):
        captured.update(name=name, num_classes=num_classes, weights=weights, params=params)
        return FakeDetector()

    predictor = predictor_module.Predictor.from_checkpoint(checkpoint, device="cpu", model_factory=factory)
    return predictor, captured


def test_predictor_restores_ordered_classes_without_yaml(checkpoint: Path) -> None:
    predictor, captured = _build_predictor(checkpoint)

    assert predictor.class_names == ("background", "cat", "dog")
    assert captured == {"name": "fake-detector", "num_classes": 3, "weights": "none", "params": {"size": 320}}


def test_single_prediction_writes_json_and_png_with_overwrite_protection(checkpoint: Path, tmp_path: Path) -> None:
    predictor, _ = _build_predictor(checkpoint)
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (20, 10), "white").save(image)
    output = tmp_path / "single"

    result = predictor.predict_single(image, output, score_threshold=0.5, display_limit=10)

    assert result.image == str(image)
    assert result.detections[0].class_name == "dog"
    assert result.detections[0].box_xyxy == (1.0, 2.0, 9.5, 8.5)
    assert (output / "sample.json").is_file()
    assert (output / "sample.png").is_file()
    payload = json.loads((output / "sample.json").read_text(encoding="utf-8"))
    assert payload["detections"][0]["score"] == pytest.approx(0.875)

    with pytest.raises(FileExistsError):
        predictor.predict_single(image, output, score_threshold=0.5, display_limit=10)
    predictor.predict_single(image, output, score_threshold=0.5, display_limit=10, overwrite=True)


def test_directory_prediction_is_sorted_and_keeps_valid_results_on_corrupt_input(
    checkpoint: Path, tmp_path: Path
) -> None:
    predictor, _ = _build_predictor(checkpoint)
    input_dir = tmp_path / "images"
    (input_dir / "nested").mkdir(parents=True)
    Image.new("RGB", (20, 10), "white").save(input_dir / "b.PNG")
    Image.new("RGB", (20, 10), "white").save(input_dir / "a.jpg")
    Image.new("RGB", (20, 10), "white").save(input_dir / "nested" / "c.JpEg")
    (input_dir / "bad.png").write_text("not an image", encoding="utf-8")
    (input_dir / "ignored.gif").write_text("ignored", encoding="utf-8")
    output = tmp_path / "batch"

    result = predictor.predict_directory(input_dir, output, score_threshold=0.5, display_limit=10)

    assert [Path(item.image).relative_to(input_dir).as_posix() for item in result.predictions] == [
        "a.jpg",
        "b.PNG",
        "nested/c.JpEg",
    ]
    assert [Path(item.image).relative_to(input_dir).as_posix() for item in result.errors] == ["bad.png"]
    payload = json.loads((output / "predictions.json").read_text(encoding="utf-8"))
    assert [item["image"] for item in payload["predictions"]] == ["a.jpg", "b.PNG", "nested/c.JpEg"]
    assert payload["errors"][0]["image"] == "bad.png"
    assert not any("ignored.gif" in str(item) for item in result.predictions)
