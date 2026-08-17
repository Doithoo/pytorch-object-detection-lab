from __future__ import annotations

from collections.abc import Mapping

import torch.nn as nn
from torchvision import models as torchvision_models
from torchvision.models import MobileNet_V3_Large_Weights, ResNet50_Weights
from torchvision.models import detection as detection_models


def build_fasterrcnn_mobilenet(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> nn.Module:
    backbone_weights = _mobile_weights(weights)
    return detection_models.fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        **dict(params),
    )


def build_fasterrcnn_resnet50(num_classes: int, weights: str, params: Mapping[str, object]) -> nn.Module:
    backbone_weights = _resnet_weights(weights)
    return detection_models.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        **dict(params),
    )


def build_ssdlite_mobilenet(num_classes: int, weights: str, params: Mapping[str, object]) -> nn.Module:
    backbone_weights = _mobile_weights(weights)
    return detection_models.ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        **dict(params),
    )


def _mobile_weights(policy: str) -> MobileNet_V3_Large_Weights | None:
    if policy == "none":
        return None
    if policy == "imagenet1k_v1":
        return MobileNet_V3_Large_Weights.IMAGENET1K_V1
    raise ValueError(f"unsupported MobileNetV3 weight policy: {policy}")


def _resnet_weights(policy: str) -> ResNet50_Weights | None:
    if policy == "none":
        return None
    if policy == "imagenet1k_v1":
        return ResNet50_Weights.IMAGENET1K_V1
    raise ValueError(f"unsupported ResNet50 weight policy: {policy}")


__all__ = [
    "MobileNet_V3_Large_Weights",
    "ResNet50_Weights",
    "build_fasterrcnn_mobilenet",
    "build_fasterrcnn_resnet50",
    "build_ssdlite_mobilenet",
    "detection_models",
    "torchvision_models",
]
