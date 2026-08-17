from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from object_detector.data.schema import VOC_CLASSES
from object_detector.data.voc import VocFormatError, parse_voc_annotation

VOC2007_SPLIT_COUNTS = {"train": 2501, "valid": 2510, "test": 4952}
COORDINATE_CONVENTION = "zero-based continuous xyxy; xmax/ymax are exclusive pixel boundaries"


class ManifestError(ValueError):
    """Raised when source data cannot produce a trustworthy manifest."""


@dataclass(frozen=True)
class DatasetMetadata:
    name: str
    dataset_root: str
    class_names: tuple[str, ...]
    label_by_name: dict[str, int]
    split_counts: dict[str, int]
    split_hashes: dict[str, str]
    identity: str
    coordinate_convention: str


@dataclass(frozen=True)
class ManifestRow:
    image_id: str
    image_path: str
    annotation_path: str


def prepare_voc2007(
    data_dir: Path,
    manifest_dir: Path,
    expected_split_counts: Mapping[str, int] | None = VOC2007_SPLIT_COUNTS,
) -> DatasetMetadata:
    voc_root = data_dir / "VOCdevkit" / "VOC2007"
    split_files = {
        "train": voc_root / "ImageSets" / "Main" / "train.txt",
        "valid": voc_root / "ImageSets" / "Main" / "val.txt",
        "test": voc_root / "ImageSets" / "Main" / "test.txt",
    }
    split_ids = {name: _read_split(path, name) for name, path in split_files.items()}
    _validate_disjoint(split_ids)
    if expected_split_counts is not None:
        for split_name, expected in expected_split_counts.items():
            actual = len(split_ids[split_name])
            if actual != expected:
                raise ManifestError(f"{split_name} split has {actual} images; expected {expected}")

    rows = {name: tuple(_validate_sample(voc_root, image_id) for image_id in ids) for name, ids in split_ids.items()}
    split_hashes = {name: _rows_hash(items) for name, items in rows.items()}
    identity_data = {
        "name": "voc2007",
        "classes": VOC_CLASSES,
        "coordinate_convention": COORDINATE_CONVENTION,
        "split_hashes": split_hashes,
    }
    identity = hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()
    metadata = DatasetMetadata(
        name="voc2007",
        dataset_root="VOCdevkit/VOC2007",
        class_names=VOC_CLASSES,
        label_by_name={name: index for index, name in enumerate(VOC_CLASSES, start=1)},
        split_counts={name: len(items) for name, items in rows.items()},
        split_hashes=split_hashes,
        identity=identity,
        coordinate_convention=COORDINATE_CONVENTION,
    )
    _write_manifests(manifest_dir, rows, metadata)
    return metadata


def load_dataset_metadata(manifest_dir: Path) -> DatasetMetadata:
    path = manifest_dir / "dataset.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read dataset metadata {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path}: metadata root must be a mapping")
    try:
        return DatasetMetadata(
            name=str(raw["name"]),
            dataset_root=str(raw["dataset_root"]),
            class_names=tuple(raw["class_names"]),
            label_by_name={str(key): int(value) for key, value in raw["label_by_name"].items()},
            split_counts={str(key): int(value) for key, value in raw["split_counts"].items()},
            split_hashes={str(key): str(value) for key, value in raw["split_hashes"].items()},
            identity=str(raw["identity"]),
            coordinate_convention=str(raw["coordinate_convention"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"{path}: invalid dataset metadata: {exc}") from exc


def read_manifest(path: Path) -> tuple[ManifestRow, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["image_id", "image_path", "annotation_path"]:
                raise ManifestError(f"{path}: invalid manifest columns")
            return tuple(
                ManifestRow(
                    image_id=row["image_id"],
                    image_path=row["image_path"],
                    annotation_path=row["annotation_path"],
                )
                for row in reader
            )
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc


def _read_split(path: Path, split_name: str) -> tuple[str, ...]:
    try:
        image_ids = tuple(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError as exc:
        raise ManifestError(f"cannot read {split_name} split {path}: {exc}") from exc
    if len(set(image_ids)) != len(image_ids):
        raise ManifestError(f"{split_name} split contains duplicate image IDs")
    return image_ids


def _validate_disjoint(split_ids: Mapping[str, tuple[str, ...]]) -> None:
    names = tuple(split_ids)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = sorted(set(split_ids[left]) & set(split_ids[right]))
            if overlap:
                preview = ", ".join(overlap[:3])
                raise ManifestError(f"split overlap between {left} and {right}: {preview}")


def _validate_sample(voc_root: Path, image_id: str) -> ManifestRow:
    image_rel = Path("JPEGImages") / f"{image_id}.jpg"
    annotation_rel = Path("Annotations") / f"{image_id}.xml"
    image_path = voc_root / image_rel
    annotation_path = voc_root / annotation_rel
    if not image_path.is_file():
        raise ManifestError(f"missing image for {image_id}: {image_path}")
    if not annotation_path.is_file():
        raise ManifestError(f"missing annotation for {image_id}: {annotation_path}")
    try:
        annotation = parse_voc_annotation(annotation_path)
    except VocFormatError as exc:
        raise ManifestError(str(exc)) from exc
    if annotation.filename != image_path.name:
        raise ManifestError(f"{annotation_path}: filename {annotation.filename!r} does not match {image_path.name!r}")
    return ManifestRow(image_id=image_id, image_path=image_rel.as_posix(), annotation_path=annotation_rel.as_posix())


def _rows_hash(rows: tuple[ManifestRow, ...]) -> str:
    normalized = "".join(f"{row.image_id},{row.image_path},{row.annotation_path}\n" for row in rows)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _write_manifests(
    manifest_dir: Path,
    rows_by_split: Mapping[str, tuple[ManifestRow, ...]],
    metadata: DatasetMetadata,
) -> None:
    manifest_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{manifest_dir.name}.stage-", dir=manifest_dir.parent))
    backup: Path | None = None
    try:
        for split_name, rows in rows_by_split.items():
            with (stage / f"{split_name}.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["image_id", "image_path", "annotation_path"])
                writer.writeheader()
                writer.writerows(asdict(row) for row in rows)
        (stage / "dataset.yaml").write_text(
            yaml.safe_dump(asdict(metadata), sort_keys=False),
            encoding="utf-8",
        )
        (stage / "source.yaml").write_text(
            yaml.safe_dump(
                {
                    "dataset": "Pascal VOC 2007",
                    "dataset_root": metadata.dataset_root,
                    "archives": {
                        "VOCtrainval_06-Nov-2007.tar": "c52e279531787c972589f7e41ab4ae64",
                        "VOCtest_06-Nov-2007.tar": "b6e924de25625d8de591ea690078ad9f",
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        summary = [f"identity: {metadata.identity}"]
        summary.extend(f"{name}: {count}" for name, count in metadata.split_counts.items())
        (stage / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

        if manifest_dir.exists():
            backup = manifest_dir.with_name(f".{manifest_dir.name}.backup-{uuid.uuid4().hex}")
            os.replace(manifest_dir, backup)
        try:
            os.replace(stage, manifest_dir)
        except OSError:
            if backup is not None:
                os.replace(backup, manifest_dir)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        if backup is not None and backup.exists() and not manifest_dir.exists():
            os.replace(backup, manifest_dir)
