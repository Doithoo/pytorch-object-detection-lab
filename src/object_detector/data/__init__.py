from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_detector.data.dataset import VocDetectionDataset as VocDetectionDataset
    from object_detector.data.dataset import detection_collate as detection_collate
    from object_detector.data.inspection import render_detection_preview as render_detection_preview
    from object_detector.data.manifest import DatasetMetadata as DatasetMetadata
    from object_detector.data.manifest import ManifestError as ManifestError
    from object_detector.data.manifest import prepare_voc2007 as prepare_voc2007
    from object_detector.data.schema import VOC_CLASSES as VOC_CLASSES
    from object_detector.data.schema import VocAnnotation as VocAnnotation
    from object_detector.data.schema import VocObject as VocObject
    from object_detector.data.voc import VocFormatError as VocFormatError
    from object_detector.data.voc import parse_voc_annotation as parse_voc_annotation

__all__ = [
    "VOC_CLASSES",
    "DatasetMetadata",
    "ManifestError",
    "VocDetectionDataset",
    "VocAnnotation",
    "VocFormatError",
    "VocObject",
    "detection_collate",
    "parse_voc_annotation",
    "prepare_voc2007",
    "render_detection_preview",
]

_EXPORTS = {
    "VOC_CLASSES": ("object_detector.data.schema", "VOC_CLASSES"),
    "DatasetMetadata": ("object_detector.data.manifest", "DatasetMetadata"),
    "ManifestError": ("object_detector.data.manifest", "ManifestError"),
    "VocDetectionDataset": ("object_detector.data.dataset", "VocDetectionDataset"),
    "VocAnnotation": ("object_detector.data.schema", "VocAnnotation"),
    "VocFormatError": ("object_detector.data.voc", "VocFormatError"),
    "VocObject": ("object_detector.data.schema", "VocObject"),
    "detection_collate": ("object_detector.data.dataset", "detection_collate"),
    "parse_voc_annotation": ("object_detector.data.voc", "parse_voc_annotation"),
    "prepare_voc2007": ("object_detector.data.manifest", "prepare_voc2007"),
    "render_detection_preview": ("object_detector.data.inspection", "render_detection_preview"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
