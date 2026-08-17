from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from object_detector.data.schema import VOC_CLASSES, VocAnnotation, VocObject


class VocFormatError(ValueError):
    """Raised when a Pascal VOC annotation violates the supported schema."""


def parse_voc_annotation(path: Path) -> VocAnnotation:
    try:
        root = ET.parse(path).getroot()
    except OSError as exc:
        raise VocFormatError(f"{path}: cannot read annotation: {exc}") from exc
    except ET.ParseError as exc:
        raise VocFormatError(f"{path}: invalid XML: {exc}") from exc

    filename = _required_text(root, "filename", path)
    size = root.find("size")
    if size is None:
        raise VocFormatError(f"{path}: missing size")
    width = _positive_integer(_required_text(size, "width", path), path, "size.width")
    height = _positive_integer(_required_text(size, "height", path), path, "size.height")

    objects = tuple(
        _parse_object(node, index, width, height, path) for index, node in enumerate(root.findall("object"))
    )
    return VocAnnotation(filename=filename, width=width, height=height, objects=objects)


def _parse_object(node: ET.Element, index: int, width: int, height: int, path: Path) -> VocObject:
    prefix = f"{path}: object {index}"
    class_name = _required_text(node, "name", path)
    if class_name not in VOC_CLASSES:
        raise VocFormatError(f"{prefix}: unknown class {class_name!r}")

    difficult_text = node.findtext("difficult", default="0").strip()
    if difficult_text not in {"0", "1"}:
        raise VocFormatError(f"{prefix}: difficult must be 0 or 1")

    box_node = node.find("bndbox")
    if box_node is None:
        raise VocFormatError(f"{prefix}: missing bndbox")
    values = tuple(
        _finite_number(_required_text(box_node, field, path), path, f"object {index}.bndbox.{field}")
        for field in ("xmin", "ymin", "xmax", "ymax")
    )
    xmin = min(max(values[0] - 1.0, 0.0), float(width))
    ymin = min(max(values[1] - 1.0, 0.0), float(height))
    xmax = min(max(values[2], 0.0), float(width))
    ymax = min(max(values[3], 0.0), float(height))
    if xmax <= xmin:
        raise VocFormatError(f"{prefix}: box must have positive width after clipping")
    if ymax <= ymin:
        raise VocFormatError(f"{prefix}: box must have positive height after clipping")
    return VocObject(class_name=class_name, box=(xmin, ymin, xmax, ymax), difficult=difficult_text == "1")


def _required_text(node: ET.Element, field: str, path: Path) -> str:
    text = node.findtext(field)
    if text is None or not text.strip():
        raise VocFormatError(f"{path}: missing {field}")
    return text.strip()


def _positive_integer(value: str, path: Path, field: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise VocFormatError(f"{path}: {field} must be an integer") from exc
    if result <= 0:
        raise VocFormatError(f"{path}: {field} must be positive")
    return result


def _finite_number(value: str, path: Path, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise VocFormatError(f"{path}: {field} must be numeric") from exc
    if not math.isfinite(result):
        raise VocFormatError(f"{path}: {field} must be finite")
    return result
