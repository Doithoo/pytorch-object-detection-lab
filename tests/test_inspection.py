from pathlib import Path

import pytest
from PIL import Image

import object_detector.data.inspection as inspection
from object_detector.data.dataset import VocDetectionDataset
from object_detector.data.inspection import render_detection_preview
from tests.conftest import PreparedVoc


def test_preview_writes_a_nonempty_rgb_image(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    dataset = VocDetectionDataset.from_manifests(prepared_voc.manifests, "train", training=False)
    output = tmp_path / "preview.png"

    render_detection_preview([dataset[0], dataset[1]], ("background", *prepared_voc.metadata.class_names), output)

    with Image.open(output) as preview:
        assert preview.mode == "RGB"
        assert preview.width > 20
        assert preview.height >= 10
    assert output.stat().st_size > 100


def test_inspection_summarizes_prepared_targets(prepared_voc: PreparedVoc) -> None:
    report = inspection.inspect_prepared_data(
        prepared_voc.manifests,
        split="train",
        data_dir=prepared_voc.voc_root.parent.parent,
        limit=2,
    )

    assert report == {
        "dataset": "voc2007",
        "identity": prepared_voc.metadata.identity,
        "split": "train",
        "total_images": 2,
        "inspected_images": 2,
        "ordinary_objects": 2,
        "difficult_objects": 1,
        "empty_images": 0,
        "images_with_difficult": 1,
        "class_counts": {
            "ordinary": {"cat": 1, "dog": 1},
            "difficult": {"dog": 1},
        },
        "image_size": {
            "min_height": 10,
            "max_height": 10,
            "min_width": 20,
            "max_width": 20,
        },
        "boxes": {
            "count": 3,
            "min_width": 9.0,
            "max_width": 16.0,
            "min_height": 8.0,
            "max_height": 9.0,
            "min_area": 72.0,
            "max_area": 144.0,
        },
    }


def test_inspection_rejects_a_nonpositive_limit(prepared_voc: PreparedVoc) -> None:
    with pytest.raises(ValueError, match="inspection limit must be positive"):
        inspection.inspect_prepared_data(prepared_voc.manifests, split="train", limit=0)
