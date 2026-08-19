from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_detector.evaluation.comparison import ComparisonReport as ComparisonReport
    from object_detector.evaluation.comparison import compare_runs as compare_runs
    from object_detector.evaluation.metrics import DetectionMetric as DetectionMetric

__all__ = ["ComparisonReport", "DetectionMetric", "compare_runs"]

_EXPORTS = {
    "ComparisonReport": ("object_detector.evaluation.comparison", "ComparisonReport"),
    "DetectionMetric": ("object_detector.evaluation.metrics", "DetectionMetric"),
    "compare_runs": ("object_detector.evaluation.comparison", "compare_runs"),
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
