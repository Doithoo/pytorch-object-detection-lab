from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

import yaml
from PIL import Image, UnidentifiedImageError

from object_detector.data.schema import VOC_CLASSES
from object_detector.data.voc import VocFormatError, parse_voc_annotation

VOC2007_SPLIT_COUNTS = {"train": 2501, "valid": 2510, "test": 4952}
MANIFEST_SCHEMA_VERSION = 2
MANIFEST_COLUMNS = ["image_id", "image_path", "annotation_path"]
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
    manifest_hashes: dict[str, str]
    identity: str
    coordinate_convention: str
    schema_version: int = MANIFEST_SCHEMA_VERSION


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
    split_hashes = {name: _rows_hash(items, voc_root) for name, items in rows.items()}
    manifest_hashes = {name: _manifest_hash(items) for name, items in rows.items()}
    identity = _identity_for(split_hashes)

    metadata = DatasetMetadata(
        name="voc2007",
        dataset_root="VOCdevkit/VOC2007",
        class_names=VOC_CLASSES,
        label_by_name={name: index for index, name in enumerate(VOC_CLASSES, start=1)},
        split_counts={name: len(items) for name, items in rows.items()},
        split_hashes=split_hashes,
        manifest_hashes=manifest_hashes,
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
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{path}: metadata root must be a mapping")
    try:
        metadata = DatasetMetadata(
            name=str(raw["name"]),
            dataset_root=str(raw["dataset_root"]),
            class_names=tuple(raw["class_names"]),
            label_by_name={str(key): int(value) for key, value in raw["label_by_name"].items()},
            split_counts={str(key): int(value) for key, value in raw["split_counts"].items()},
            split_hashes={str(key): str(value) for key, value in raw["split_hashes"].items()},
            manifest_hashes={str(key): str(value) for key, value in raw["manifest_hashes"].items()},
            identity=str(raw["identity"]),
            coordinate_convention=str(raw["coordinate_convention"]),
            schema_version=int(raw["schema_version"]),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"{path}: invalid dataset metadata: {exc}") from exc
    _validate_metadata(metadata, manifest_dir)
    return metadata


def read_manifest(path: Path) -> tuple[ManifestRow, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != MANIFEST_COLUMNS:
                raise ManifestError(f"{path}: invalid manifest columns")
            rows: list[ManifestRow] = []
            seen_ids: set[str] = set()
            for line_number, row in enumerate(reader, start=2):
                try:
                    item = ManifestRow(
                        image_id=row["image_id"],
                        image_path=row["image_path"],
                        annotation_path=row["annotation_path"],
                    )
                except (KeyError, TypeError) as exc:
                    raise ManifestError(f"{path}:{line_number}: incomplete manifest row") from exc
                if not all((item.image_id, item.image_path, item.annotation_path)):
                    raise ManifestError(f"{path}:{line_number}: manifest values must not be empty")
                if item.image_id in seen_ids:
                    raise ManifestError(f"{path}:{line_number}: duplicate image ID {item.image_id!r}")
                _validate_relative_path(item.image_path, path, line_number)
                _validate_relative_path(item.annotation_path, path, line_number)
                seen_ids.add(item.image_id)
                rows.append(item)
            return tuple(rows)
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc


def _validate_metadata(metadata: DatasetMetadata, manifest_dir: Path) -> None:
    if metadata.schema_version != MANIFEST_SCHEMA_VERSION:
        raise ManifestError(
            f"{manifest_dir / 'dataset.yaml'}: unsupported schema_version {metadata.schema_version!r}; "
            "regenerate manifests"
        )
    if metadata.name != "voc2007":
        raise ManifestError(f"{manifest_dir / 'dataset.yaml'}: unsupported dataset {metadata.name!r}")
    if metadata.dataset_root != "VOCdevkit/VOC2007":
        raise ManifestError(f"{manifest_dir / 'dataset.yaml'}: dataset_root is not the VOC 2007 root")
    if metadata.class_names != VOC_CLASSES:
        raise ManifestError(f"{manifest_dir / 'dataset.yaml'}: class_names do not match the VOC class schema")
    expected_labels = {name: index for index, name in enumerate(VOC_CLASSES, start=1)}
    if metadata.label_by_name != expected_labels:
        raise ManifestError(f"{manifest_dir / 'dataset.yaml'}: label_by_name does not match class_names")
    if metadata.coordinate_convention != COORDINATE_CONVENTION:
        raise ManifestError(f"{manifest_dir / 'dataset.yaml'}: unsupported coordinate_convention")
    expected_splits = set(VOC2007_SPLIT_COUNTS)
    for field_name, values in (
        ("split_counts", metadata.split_counts),
        ("split_hashes", metadata.split_hashes),
        ("manifest_hashes", metadata.manifest_hashes),
    ):
        if set(values) != expected_splits:
            raise ManifestError(f"{manifest_dir / 'dataset.yaml'}: {field_name} must define train, valid, and test")
    expected_identity = _identity_for(metadata.split_hashes)
    if metadata.identity != expected_identity:
        raise ManifestError(f"{manifest_dir / 'dataset.yaml'}: identity does not match metadata contents")
    for split_name in ("train", "valid", "test"):
        rows = read_manifest(manifest_dir / f"{split_name}.csv")
        if len(rows) != metadata.split_counts[split_name]:
            raise ManifestError(
                f"{split_name}.csv has {len(rows)} rows; metadata records {metadata.split_counts[split_name]}"
            )
        actual_hash = _manifest_hash(rows)
        if actual_hash != metadata.manifest_hashes[split_name]:
            raise ManifestError(f"{split_name}.csv hash does not match dataset metadata")


def verify_prepared_data(data_dir: Path, metadata: DatasetMetadata, manifest_dir: Path) -> None:
    """Verify source bytes still match the hashes captured during preparation."""
    rows_by_split = {
        split_name: read_manifest(manifest_dir / f"{split_name}.csv") for split_name in ("train", "valid", "test")
    }
    _validate_disjoint({split_name: tuple(row.image_id for row in rows) for split_name, rows in rows_by_split.items()})
    voc_root = data_dir / metadata.dataset_root
    for split_name, rows in rows_by_split.items():
        try:
            actual_hash = _rows_hash(rows, voc_root)
        except OSError as exc:
            raise ManifestError(f"cannot verify {split_name} source files below {voc_root}: {exc}") from exc
        if actual_hash != metadata.split_hashes[split_name]:
            raise ManifestError(f"{split_name} source files do not match dataset metadata")


def _identity_for(split_hashes: Mapping[str, str]) -> str:
    identity_data = {
        "name": "voc2007",
        "classes": VOC_CLASSES,
        "coordinate_convention": COORDINATE_CONVENTION,
        "split_hashes": dict(split_hashes),
    }
    return hashlib.sha256(json.dumps(identity_data, sort_keys=True).encode()).hexdigest()


def _validate_relative_path(value: str, manifest_path: Path, line_number: int) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ManifestError(
            f"{manifest_path}:{line_number}: manifest paths must be relative and stay below the dataset root"
        )


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
    try:
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            dimensions = image.size
    except (OSError, UnidentifiedImageError) as exc:
        raise ManifestError(f"cannot decode image for {image_id}: {image_path}: {exc}") from exc
    if dimensions != (annotation.width, annotation.height):
        raise ManifestError(
            f"{annotation_path}: dimensions {annotation.width}x{annotation.height} "
            f"disagree with image {dimensions[0]}x{dimensions[1]}"
        )
    return ManifestRow(image_id=image_id, image_path=image_rel.as_posix(), annotation_path=annotation_rel.as_posix())


def _manifest_hash(rows: tuple[ManifestRow, ...]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=MANIFEST_COLUMNS)
    writer.writeheader()
    writer.writerows(asdict(row) for row in rows)
    return hashlib.sha256(output.getvalue().encode()).hexdigest()


def _rows_hash(rows: tuple[ManifestRow, ...], voc_root: Path) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row.image_id},{row.image_path},{row.annotation_path}\n".encode())
        for relative_path in (row.image_path, row.annotation_path):
            path = voc_root / relative_path
            digest.update(relative_path.encode())
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


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
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
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
