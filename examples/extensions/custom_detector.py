from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class SingleScaleBackbone(nn.Module):
    """Expose one feature tensor and its channel count to Faster R-CNN."""

    def __init__(self, features: nn.Module, out_channels: int) -> None:
        super().__init__()
        self.features = features
        self.out_channels = out_channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.features(image)


def build_detector(
    *,
    num_classes: int,
    weights: str,
    width_mult: float = 0.5,
    min_size: int = 320,
    max_size: int = 640,
) -> nn.Module:
    """Build Faster R-CNN around a single-scale MobileNet V3 Small backbone."""
    if weights == "none":
        backbone_weights = None
    elif weights == "imagenet1k_v1":
        backbone_weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    else:
        raise ValueError(f"unsupported weight policy: {weights}")

    features = mobilenet_v3_small(weights=backbone_weights, width_mult=width_mult).features
    backbone = SingleScaleBackbone(features, features[-1].out_channels)
    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128, 256),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    return FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=min_size,
        max_size=max_size,
    )
