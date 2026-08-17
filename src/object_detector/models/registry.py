from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn

from object_detector.models import torchvision_models
from object_detector.models.spec import ModelSpec


class ModelConfigError(ValueError):
    """Raised when a model name, class count, or weight policy is invalid."""


_REGISTRY: dict[str, ModelSpec] = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": ModelSpec(
        name="fasterrcnn_mobilenet_v3_large_320_fpn",
        constructor=torchvision_models.build_fasterrcnn_mobilenet,
        family="two_stage",
        backbone_weights={
            "none": None,
            "imagenet1k_v1": torchvision_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        },
    ),
    "fasterrcnn_resnet50_fpn": ModelSpec(
        name="fasterrcnn_resnet50_fpn",
        constructor=torchvision_models.build_fasterrcnn_resnet50,
        family="two_stage",
        backbone_weights={
            "none": None,
            "imagenet1k_v1": torchvision_models.ResNet50_Weights.IMAGENET1K_V1,
        },
    ),
    "ssdlite320_mobilenet_v3_large": ModelSpec(
        name="ssdlite320_mobilenet_v3_large",
        constructor=torchvision_models.build_ssdlite_mobilenet,
        family="one_stage",
        backbone_weights={
            "none": None,
            "imagenet1k_v1": torchvision_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        },
    ),
}


def list_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(list_models())
        raise ModelConfigError(f"unknown model {name!r}; choose one of {choices}") from exc


def build_model(
    name: str,
    num_classes: int,
    weights: str = "none",
    params: Mapping[str, object] | None = None,
) -> nn.Module:
    if num_classes < 2:
        raise ModelConfigError("num_classes must include background and at least one object class")
    spec = get_model_spec(name)
    if weights not in spec.supported_weights:
        raise ModelConfigError(f"{name} does not support weights={weights!r}")
    model_params = dict(params or {})
    reserved = {"weights", "weights_backbone", "num_classes"}
    collision = reserved & model_params.keys()
    if collision:
        raise ModelConfigError(f"model params cannot override {', '.join(sorted(collision))}")
    return spec.constructor(num_classes, weights, model_params)


def get_backbone_weight(name: str, policy: str) -> object | None:
    spec = get_model_spec(name)
    if policy not in spec.backbone_weights:
        raise ModelConfigError(f"{name} does not define weights={policy!r}")
    return spec.backbone_weights[policy]


def expected_weight_cache_path(name: str, policy: str) -> Path:
    weight = get_backbone_weight(name, policy)
    if weight is None:
        raise ModelConfigError(f"{name} has no cache path for weights={policy!r}")
    url = getattr(weight, "url", None)
    if not isinstance(url, str):
        raise ModelConfigError(f"{name} weight policy {policy!r} has no download URL")
    return Path(torch.hub.get_dir()) / "checkpoints" / os.path.basename(urlparse(url).path)
