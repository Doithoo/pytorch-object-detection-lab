import json
from pathlib import Path

import pytest
from PIL import Image

from object_detector.data.dataset import VocDetectionDataset
from object_detector.data.manifest import (
    ManifestError,
    load_dataset_metadata,
    prepare_coco,
    prepare_voc2007,
    verify_prepared_data,
)
from tests.fixtures.voc import build_voc_tree


def test_prepare_coco_writes_manifest_and_normalizes_categories(tmp_path: Path) -> None:
    data_dir = tmp_path / "coco"
    images_dir = data_dir / "images"
    annotations_dir = data_dir / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)
    categories = [{"id": 17, "name": "cat"}, {"id": 3, "name": "dog"}]
    annotation_files = {}
    for split, image_id, class_id in (("train", 1, 17), ("valid", 2, 3), ("test", 3, 17)):
        file_name = f"{split}.jpg"
        Image.new("RGB", (10, 8), "white").save(images_dir / file_name)
        annotation_path = annotations_dir / f"instances_{split}.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "images": [{"id": image_id, "file_name": file_name, "width": 10, "height": 8}],
                    "annotations": [
                        {
                            "id": image_id,
                            "image_id": image_id,
                            "category_id": class_id,
                            "bbox": [1, 1, 4, 3],
                            "iscrowd": 0,
                        }
                    ],
                    "categories": categories,
                }
            ),
            encoding="utf-8",
        )
        annotation_files[split] = annotation_path

    metadata = prepare_coco(data_dir, tmp_path / "manifests", annotation_files)

    assert metadata.class_names == ("cat", "dog")
    assert metadata.label_by_name == {"cat": 1, "dog": 2}
    assert metadata.annotation_format == "coco"
    dataset = VocDetectionDataset.from_manifests(
        tmp_path / "manifests",
        "train",
        data_dir=data_dir,
        training=False,
    )
    _, target = dataset[0]
    assert target["labels"].tolist() == [1]


def test_prepare_uses_official_splits_and_stable_hash(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")

    first = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests", expected_split_counts=None)
    second = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests-2", expected_split_counts=None)

    assert first.split_counts == {"train": 2, "valid": 1, "test": 1}
    assert first.identity == second.identity
    assert first.class_names == ("cat", "dog", "person")
    assert first.class_names[0] == "cat"
    assert (tmp_path / "manifests" / "dataset.yaml").is_file()
    assert (tmp_path / "manifests" / "source.yaml").is_file()


def test_overlap_does_not_replace_existing_manifests(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    output = tmp_path / "manifests"
    prepare_voc2007(voc_root.parent.parent, output, expected_split_counts=None)
    original = (output / "dataset.yaml").read_bytes()
    (voc_root / "ImageSets/Main/val.txt").write_text("train-1\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="split overlap"):
        prepare_voc2007(voc_root.parent.parent, output, expected_split_counts=None)

    assert (output / "dataset.yaml").read_bytes() == original


def test_production_counts_are_checked(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")

    with pytest.raises(ManifestError, match="train split has 2 images; expected 2501"):
        prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests")


def test_prepare_infers_custom_voc_classes_and_stable_labels(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    for annotation_path in (voc_root / "Annotations").glob("*.xml"):
        content = annotation_path.read_text(encoding="utf-8").replace("<name>dog</name>", "<name>fox</name>")
        annotation_path.write_text(content, encoding="utf-8")

    metadata = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests", expected_split_counts=None)

    assert metadata.class_names == ("cat", "fox", "person")
    assert metadata.label_by_name == {"cat": 1, "fox": 2, "person": 3}
    assert metadata.identity
    assert load_dataset_metadata(tmp_path / "manifests").class_names == metadata.class_names


def test_prepare_rejects_image_dimensions_that_disagree_with_xml(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    Image.new("RGB", (21, 10), "white").save(voc_root / "JPEGImages" / "train-1.jpg")

    with pytest.raises(ManifestError, match="dimensions"):
        prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests", expected_split_counts=None)


def test_loaded_metadata_rejects_changed_label_mapping(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    output = tmp_path / "manifests"
    prepare_voc2007(voc_root.parent.parent, output, expected_split_counts=None)
    metadata_path = output / "dataset.yaml"
    content = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(content.replace("  cat: 1", "  cat: 2"), encoding="utf-8")

    with pytest.raises(ManifestError, match="label_by_name"):
        load_dataset_metadata(output)


def test_loaded_metadata_rejects_changed_manifest_rows(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    output = tmp_path / "manifests"
    prepare_voc2007(voc_root.parent.parent, output, expected_split_counts=None)
    train = output / "train.csv"
    train.write_text(train.read_text(encoding="utf-8").replace("train-1", "changed-id", 1), encoding="utf-8")

    with pytest.raises(ManifestError, match="hash"):
        load_dataset_metadata(output)


def test_source_verification_rejects_changed_source_bytes(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    output = tmp_path / "manifests"
    metadata = prepare_voc2007(voc_root.parent.parent, output, expected_split_counts=None)
    Image.new("RGB", (20, 10), "black").save(voc_root / "JPEGImages" / "train-1.jpg")

    with pytest.raises(ManifestError, match="source files"):
        verify_prepared_data(voc_root.parent.parent, metadata, output)


def test_manifest_identity_changes_when_source_content_changes(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")
    first = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests-1", expected_split_counts=None)
    Image.new("RGB", (20, 10), "black").save(voc_root / "JPEGImages" / "train-1.jpg")

    second = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests-2", expected_split_counts=None)

    assert first.identity != second.identity
