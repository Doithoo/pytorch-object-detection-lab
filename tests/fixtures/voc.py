from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from PIL import Image


def build_voc_tree(root: Path) -> Path:
    voc_root = root / "VOCdevkit" / "VOC2007"
    image_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"
    split_dir = voc_root / "ImageSets" / "Main"
    for directory in (image_dir, annotation_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    samples = {
        "train-1": [("dog", (1, 2, 16, 10), False)],
        "train-2": [("cat", (2, 1, 10, 8), False), ("dog", (11, 2, 20, 10), True)],
        "valid-1": [],
        "test-1": [("person", (4, 1, 12, 10), False)],
    }
    for index, (image_id, objects) in enumerate(samples.items()):
        Image.new("RGB", (20, 10), color=(40 * index, 60, 120)).save(image_dir / f"{image_id}.jpg")
        _write_annotation(annotation_dir / f"{image_id}.xml", f"{image_id}.jpg", objects)

    (split_dir / "train.txt").write_text("train-1\ntrain-2\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("valid-1\n", encoding="utf-8")
    (split_dir / "test.txt").write_text("test-1\n", encoding="utf-8")
    return voc_root


def _write_annotation(
    path: Path,
    filename: str,
    objects: list[tuple[str, tuple[int, int, int, int], bool]],
) -> None:
    root = Element("annotation")
    SubElement(root, "filename").text = filename
    size = SubElement(root, "size")
    SubElement(size, "width").text = "20"
    SubElement(size, "height").text = "10"
    SubElement(size, "depth").text = "3"
    for class_name, box, difficult in objects:
        object_node = SubElement(root, "object")
        SubElement(object_node, "name").text = class_name
        SubElement(object_node, "difficult").text = "1" if difficult else "0"
        box_node = SubElement(object_node, "bndbox")
        for field, value in zip(("xmin", "ymin", "xmax", "ymax"), box, strict=True):
            SubElement(box_node, field).text = str(value)
    ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
