from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from object_detector.models import torchvision_models as torchvision_models
    from object_detector.models.registry import ModelConfigError as ModelConfigError
    from object_detector.models.registry import build_model as build_model
    from object_detector.models.registry import get_backbone_weight as get_backbone_weight
    from object_detector.models.registry import list_models as list_models

__all__ = ["ModelConfigError", "build_model", "get_backbone_weight", "list_models", "torchvision_models"]

_EXPORTS: dict[str, tuple[str, str | None]] = {
    "ModelConfigError": ("object_detector.models.registry", "ModelConfigError"),
    "build_model": ("object_detector.models.registry", "build_model"),
    "get_backbone_weight": ("object_detector.models.registry", "get_backbone_weight"),
    "list_models": ("object_detector.models.registry", "list_models"),
    "torchvision_models": ("object_detector.models.torchvision_models", None),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = import_module(module_name)
    value = module if attribute_name is None else getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
