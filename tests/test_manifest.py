from pathlib import Path

import pytest

from object_detector.data.manifest import ManifestError, prepare_voc2007
from tests.fixtures.voc import build_voc_tree


def test_prepare_uses_official_splits_and_stable_hash(tmp_path: Path) -> None:
    voc_root = build_voc_tree(tmp_path / "raw")

    first = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests", expected_split_counts=None)
    second = prepare_voc2007(voc_root.parent.parent, tmp_path / "manifests-2", expected_split_counts=None)

    assert first.split_counts == {"train": 2, "valid": 1, "test": 1}
    assert first.identity == second.identity
    assert first.class_names[0] == "aeroplane"
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
