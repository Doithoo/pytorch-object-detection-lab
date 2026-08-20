from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from object_detector.data.dataset import VocDetectionDataset


@dataclass(frozen=True)
class YoloExportResult:
    output_dir: Path
    image_count: int
    object_count: int
    class_names: tuple[str, ...]


def export_yolo_dataset(
    manifest_dir: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> YoloExportResult:
    """Export prepared detection data as normalized YOLO text labels."""
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"YOLO output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.stage-", dir=output_dir.parent))
    image_count = 0
    object_count = 0
    class_names: tuple[str, ...] | None = None
    try:
        for split in ("train", "valid", "test"):
            dataset = VocDetectionDataset.from_manifests(
                manifest_dir,
                split,
                data_dir=data_dir,
                training=False,
            )
            current_names = dataset.metadata.class_names
            if class_names is None:
                class_names = current_names
            elif class_names != current_names:
                raise ValueError("prepared splits do not share one class schema")
            image_lines: list[str] = []
            used_names: set[str] = set()
            for index, row in enumerate(dataset.rows):
                image, target = dataset[index]
                source = dataset.dataset_root / row.image_path
                safe_id = _safe_name(row.image_id)
                if safe_id in used_names:
                    raise ValueError(f"YOLO filename collision after normalizing image ID {row.image_id!r}")
                used_names.add(safe_id)
                suffix = source.suffix.lower() or ".jpg"
                image_relative = Path("images") / split / f"{safe_id}{suffix}"
                label_relative = Path("labels") / split / f"{safe_id}.txt"
                image_destination = stage / image_relative
                image_destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, image_destination)
                ordinary = target["iscrowd"] == 0
                boxes = target["boxes"][ordinary]
                labels = target["labels"][ordinary]
                height, width = image.shape[-2:]
                label_lines = []
                for box, label in zip(boxes.tolist(), labels.tolist(), strict=True):
                    xmin, ymin, xmax, ymax = (float(value) for value in box)
                    center_x = ((xmin + xmax) / 2.0) / width
                    center_y = ((ymin + ymax) / 2.0) / height
                    box_width = (xmax - xmin) / width
                    box_height = (ymax - ymin) / height
                    label_lines.append(
                        f"{int(label) - 1} {center_x:.8f} {center_y:.8f} {box_width:.8f} {box_height:.8f}"
                    )
                label_path = stage / label_relative
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
                image_lines.append(image_relative.as_posix())
                image_count += 1
                object_count += len(label_lines)
            (stage / f"{split}.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
        assert class_names is not None
        (stage / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "train.txt",
                    "val": "valid.txt",
                    "test": "test.txt",
                    "names": {index: name for index, name in enumerate(class_names)},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        _publish_directory(stage, output_dir, overwrite=overwrite)
        return YoloExportResult(output_dir, image_count, object_count, class_names)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def _publish_directory(stage: Path, destination: Path, *, overwrite: bool) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite and any(destination.iterdir()):
            raise FileExistsError(f"YOLO output directory already exists: {destination}")
        backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
        os.replace(destination, backup)
    try:
        os.replace(stage, destination)
    except OSError:
        if backup is not None and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _safe_name(value: str) -> str:
    result = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    if not result:
        raise ValueError("image ID cannot produce an empty YOLO filename")
    return result
