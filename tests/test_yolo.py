from pathlib import Path

import pytest
import yaml

from object_detector.data.yolo import export_yolo_dataset
from tests.conftest import PreparedVoc


def test_export_yolo_dataset_writes_normalized_labels(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    output = tmp_path / "yolo"

    result = export_yolo_dataset(
        prepared_voc.manifests,
        prepared_voc.voc_root.parent.parent,
        output,
    )

    assert result.image_count == 4
    assert result.object_count == 3
    assert result.class_names == ("cat", "dog", "person")
    first_label = (output / "labels/train/train-1.txt").read_text(encoding="utf-8").strip().split()
    assert first_label[0] == "1"
    assert [float(value) for value in first_label[1:]] == pytest.approx([0.4, 0.55, 0.8, 0.9])
    second_labels = (output / "labels/train/train-2.txt").read_text(encoding="utf-8").splitlines()
    assert len(second_labels) == 1
    assert second_labels[0].startswith("0 ")
    metadata = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert metadata["names"] == {0: "cat", 1: "dog", 2: "person"}
    assert metadata["train"] == "train.txt"
    assert (output / "images/test/test-1.jpg").is_file()

    with pytest.raises(FileExistsError):
        export_yolo_dataset(prepared_voc.manifests, prepared_voc.voc_root.parent.parent, output)
