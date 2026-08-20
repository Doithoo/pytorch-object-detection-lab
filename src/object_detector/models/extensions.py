from __future__ import annotations

import importlib
import importlib.util
import inspect
from collections.abc import Callable, Iterable
from pathlib import Path
from types import ModuleType

import torch
from torch import nn


class ExtensionError(RuntimeError):
    """An external factory could not satisfy the detection contract."""


def load_factory(path: str, *, required_keywords: Iterable[str]) -> Callable[..., object]:
    if not isinstance(path, str) or path.count(":") != 1:
        raise ExtensionError(f"invalid factory path {path!r}; expected module:function")
    module_name, attribute_name = path.split(":", 1)
    if not module_name or not attribute_name:
        raise ExtensionError(f"invalid factory path {path!r}; expected module:function")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        module = _load_local_module(module_name, exc)
    except (ImportError, OSError) as exc:
        raise ExtensionError(f"could not load factory {path!r}: {exc}") from exc
    try:
        factory = getattr(module, attribute_name)
    except AttributeError as exc:
        raise ExtensionError(f"could not load factory {path!r}: {exc}") from exc
    if not callable(factory):
        raise ExtensionError(f"factory {path!r} is not callable")
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError) as exc:
        raise ExtensionError(f"could not inspect factory {path!r}: {exc}") from exc
    required = set(required_keywords)
    accepts_extra_keywords = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    for keyword in required:
        parameter = parameters.get(keyword)
        if parameter is None and not accepts_extra_keywords:
            raise ExtensionError(f"factory {path!r} does not accept required keyword {keyword!r}")
        if parameter is not None and parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise ExtensionError(f"factory {path!r} requires {keyword!r} positionally")
    for parameter in parameters.values():
        if (
            parameter.default is inspect.Parameter.empty
            and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            and parameter.name not in required
        ):
            raise ExtensionError(f"factory {path!r} has unsupported required parameter {parameter.name!r}")
    return factory


def _load_local_module(module_name: str, original_error: ModuleNotFoundError) -> ModuleType:
    source = Path.cwd().joinpath(*module_name.split(".")).with_suffix(".py")
    if not source.is_file():
        raise ExtensionError(f"could not import module {module_name!r}: {original_error}") from original_error
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ExtensionError(f"could not create an import specification for {source}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        raise ExtensionError(f"could not load local module {source}: {exc}") from exc
    return module


class DetectionFactoryModel(nn.Module):
    """Validate a user detector's torchvision-compatible forward contract."""

    def __init__(self, model: nn.Module, factory_path: str, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.factory_path = factory_path
        self.num_classes = num_classes

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> object:
        output = self.model(images, targets) if self.training else self.model(images)
        if self.training:
            _validate_losses(output, self.factory_path)
        else:
            _validate_predictions(output, len(images), self.factory_path, self.num_classes)
        return output


def build_external_model(
    factory_path: str,
    *,
    num_classes: int,
    weights: str,
    params: dict[str, object],
) -> nn.Module:
    factory = load_factory(factory_path, required_keywords=("num_classes", "weights", *params))
    try:
        model = factory(num_classes=num_classes, weights=weights, **params)
    except ExtensionError:
        raise
    except Exception as exc:
        raise ExtensionError(f"external model factory {factory_path!r} failed: {exc}") from exc
    if not isinstance(model, nn.Module):
        raise ExtensionError(f"external model factory {factory_path!r} did not return torch.nn.Module")
    return DetectionFactoryModel(model, factory_path, num_classes)


def _validate_losses(output: object, factory_path: str) -> None:
    if not isinstance(output, dict) or not output:
        raise ExtensionError(f"external detector {factory_path!r} must return a nonempty loss mapping in train mode")
    if any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor) or value.numel() != 1
        for name, value in output.items()
    ):
        raise ExtensionError(f"external detector {factory_path!r} losses must be scalar tensors")


def _validate_predictions(output: object, image_count: int, factory_path: str, num_classes: int) -> None:
    if not isinstance(output, (list, tuple)) or len(output) != image_count:
        raise ExtensionError(f"external detector {factory_path!r} must return one prediction per image")
    required = {"boxes", "labels", "scores"}
    for prediction in output:
        if not isinstance(prediction, dict) or not required <= prediction.keys():
            raise ExtensionError(f"external detector {factory_path!r} predictions need boxes, labels, and scores")
        boxes = prediction["boxes"]
        labels = prediction["labels"]
        scores = prediction["scores"]
        if (
            not isinstance(boxes, torch.Tensor)
            or boxes.ndim != 2
            or boxes.shape[-1] != 4
            or not isinstance(labels, torch.Tensor)
            or labels.ndim != 1
            or not isinstance(scores, torch.Tensor)
            or scores.ndim != 1
            or len(boxes) != len(labels)
            or len(boxes) != len(scores)
            or (len(labels) and (int(labels.min()) < 1 or int(labels.max()) >= num_classes))
        ):
            raise ExtensionError(f"external detector {factory_path!r} returned an invalid prediction mapping")


__all__ = ["DetectionFactoryModel", "ExtensionError", "build_external_model", "load_factory"]
