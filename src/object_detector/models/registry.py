from __future__ import annotations

import difflib
import os
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from object_detector.models.spec import ModelSpec

if TYPE_CHECKING:
    import torch.nn as nn


class ModelConfigError(ValueError):
    """Raised when a model name, class count, or weight policy is invalid."""


class _LazyBackboneWeights(Mapping[str, object | None]):
    def __init__(self, resolver: Callable[[], object]) -> None:
        self._resolver = resolver

    def __getitem__(self, policy: str) -> object | None:
        if policy == "none":
            return None
        if policy == "imagenet1k_v1":
            return self._resolver()
        raise KeyError(policy)

    def __iter__(self) -> Iterator[str]:
        return iter(("none", "imagenet1k_v1"))

    def __len__(self) -> int:
        return 2


def _build_fasterrcnn_mobilenet(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> nn.Module:
    from object_detector.models.torchvision_models import build_fasterrcnn_mobilenet

    return build_fasterrcnn_mobilenet(num_classes, weights, params)


def _build_fasterrcnn_resnet50(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> nn.Module:
    from object_detector.models.torchvision_models import build_fasterrcnn_resnet50

    return build_fasterrcnn_resnet50(num_classes, weights, params)


def _build_retinanet_resnet50(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> nn.Module:
    from object_detector.models.torchvision_models import build_retinanet_resnet50

    return build_retinanet_resnet50(num_classes, weights, params)


def _build_fcos_resnet50(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> nn.Module:
    from object_detector.models.torchvision_models import build_fcos_resnet50

    return build_fcos_resnet50(num_classes, weights, params)


def _build_ssdlite_mobilenet(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> nn.Module:
    from object_detector.models.torchvision_models import build_ssdlite_mobilenet

    return build_ssdlite_mobilenet(num_classes, weights, params)


def _mobilenet_weight() -> object:
    from object_detector.models.torchvision_models import MobileNet_V3_Large_Weights

    return MobileNet_V3_Large_Weights.IMAGENET1K_V1


def _resnet50_weight() -> object:
    from object_detector.models.torchvision_models import ResNet50_Weights

    return ResNet50_Weights.IMAGENET1K_V1


_REGISTRY: dict[str, ModelSpec] = {
    "fasterrcnn_mobilenet_v3_large_320_fpn": ModelSpec(
        name="fasterrcnn_mobilenet_v3_large_320_fpn",
        constructor=_build_fasterrcnn_mobilenet,
        family="two_stage",
        description="Compact Faster R-CNN baseline with a MobileNet V3 FPN backbone.",
        parameters={
            "min_size": "Shorter image edge used by the internal detector transform.",
            "max_size": "Maximum longer image edge after resizing.",
            "box_score_thresh": "ROI prediction score threshold applied by the model.",
        },
        input_notes=(
            "Accepts a list of float RGB tensors in [0, 1].",
            "Training targets use zero-based continuous xyxy boxes.",
        ),
        backbone_weights=_LazyBackboneWeights(_mobilenet_weight),
    ),
    "fasterrcnn_resnet50_fpn": ModelSpec(
        name="fasterrcnn_resnet50_fpn",
        constructor=_build_fasterrcnn_resnet50,
        family="two_stage",
        description="Faster R-CNN comparison model with a ResNet-50 FPN backbone.",
        parameters={
            "min_size": "Shorter image edge used by the internal detector transform.",
            "max_size": "Maximum longer image edge after resizing.",
            "box_score_thresh": "ROI prediction score threshold applied by the model.",
        },
        input_notes=(
            "Accepts a list of float RGB tensors in [0, 1].",
            "Uses more memory and compute than the MobileNet Faster R-CNN recipe.",
        ),
        backbone_weights=_LazyBackboneWeights(_resnet50_weight),
    ),
    "retinanet_resnet50_fpn": ModelSpec(
        name="retinanet_resnet50_fpn",
        constructor=_build_retinanet_resnet50,
        family="one_stage",
        description="Anchor-based RetinaNet with a ResNet-50 FPN backbone and focal loss.",
        parameters={
            "min_size": "Shorter image edge used by the internal detector transform.",
            "max_size": "Maximum longer image edge after resizing.",
            "score_thresh": "Inference score threshold before NMS.",
            "nms_thresh": "IoU threshold used by non-maximum suppression.",
            "detections_per_img": "Maximum detections returned for one image.",
            "topk_candidates": "Highest-scoring candidates retained before NMS.",
            "fg_iou_thresh": "Anchor IoU threshold for positive training matches.",
            "bg_iou_thresh": "Anchor IoU threshold below which matches are negative.",
        },
        input_notes=(
            "Accepts a list of float RGB tensors in [0, 1].",
            "Uses dense anchors and focal loss instead of an ROI head.",
        ),
        backbone_weights=_LazyBackboneWeights(_resnet50_weight),
    ),
    "fcos_resnet50_fpn": ModelSpec(
        name="fcos_resnet50_fpn",
        constructor=_build_fcos_resnet50,
        family="one_stage",
        description="Anchor-free FCOS detector with a ResNet-50 FPN backbone.",
        parameters={
            "min_size": "Shorter image edge used by the internal detector transform.",
            "max_size": "Maximum longer image edge after resizing.",
            "score_thresh": "Inference score threshold before NMS.",
            "nms_thresh": "IoU threshold used by non-maximum suppression.",
            "detections_per_img": "Maximum detections returned for one image.",
            "topk_candidates": "Highest-scoring candidates retained before NMS.",
            "center_sampling_radius": "Radius used to select positive locations near box centers.",
        },
        input_notes=(
            "Accepts a list of float RGB tensors in [0, 1].",
            "Predicts class, box distances, and centerness without predefined anchors.",
        ),
        backbone_weights=_LazyBackboneWeights(_resnet50_weight),
    ),
    "ssdlite320_mobilenet_v3_large": ModelSpec(
        name="ssdlite320_mobilenet_v3_large",
        constructor=_build_ssdlite_mobilenet,
        family="one_stage",
        description="Single-stage SSDLite comparison model with a MobileNet V3 backbone.",
        parameters={
            "score_thresh": "Detection score threshold applied by the model.",
            "nms_thresh": "IoU threshold used by non-maximum suppression.",
            "detections_per_img": "Maximum detections returned for one image.",
        },
        input_notes=(
            "Accepts a list of float RGB tensors in [0, 1].",
            "The built-in transform resizes inputs to the detector's 320-pixel recipe.",
        ),
        backbone_weights=_LazyBackboneWeights(_mobilenet_weight),
    ),
}


def list_models() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        model_names = list_models()
        choices = ", ".join(model_names)
        suggestion = difflib.get_close_matches(name, model_names, n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        raise ModelConfigError(f"unknown model {name!r}{hint}; choose one of {choices}") from exc


def build_model(
    name: str,
    num_classes: int,
    weights: str = "none",
    params: Mapping[str, object] | None = None,
    *,
    factory: str | None = None,
) -> nn.Module:
    if num_classes < 2:
        raise ModelConfigError("num_classes must include background and at least one object class")
    model_params = dict(params or {})
    if factory is not None:
        from object_detector.models.extensions import ExtensionError, build_external_model

        try:
            return build_external_model(
                factory,
                num_classes=num_classes,
                weights=weights,
                params=model_params,
            )
        except ExtensionError as exc:
            raise ModelConfigError(str(exc)) from exc
    spec = get_model_spec(name)
    if weights not in spec.supported_weights:
        raise ModelConfigError(f"{name} does not support weights={weights!r}")
    reserved = {"weights", "weights_backbone", "num_classes"}
    collision = reserved & model_params.keys()
    if collision:
        raise ModelConfigError(f"model params cannot override {', '.join(sorted(collision))}")
    unknown = sorted(model_params.keys() - spec.parameters.keys())
    if unknown:
        parameter = unknown[0]
        suggestion = difflib.get_close_matches(parameter, spec.parameters, n=1)
        hint = f"; did you mean {suggestion[0]!r}?" if suggestion else ""
        choices = ", ".join(spec.parameters)
        raise ModelConfigError(f"unknown model param {parameter!r} for {name}{hint}; choose one of {choices}")
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
    from torch import hub

    return Path(hub.get_dir()) / "checkpoints" / os.path.basename(urlparse(url).path)
