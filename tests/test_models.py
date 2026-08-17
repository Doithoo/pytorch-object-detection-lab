from __future__ import annotations

import torch.nn as nn

from object_detector.models import torchvision_models
from object_detector.models.registry import build_model, list_models


def test_registry_contains_exact_initial_models() -> None:
    assert set(list_models()) == {
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fasterrcnn_resnet50_fpn",
        "ssdlite320_mobilenet_v3_large",
    }


def test_offline_default_disables_all_weights(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> nn.Module:
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(
        torchvision_models.detection_models,
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        fake_constructor,
    )

    build_model("fasterrcnn_mobilenet_v3_large_320_fpn", 21, "none", {})

    assert captured["weights"] is None
    assert captured["weights_backbone"] is None
    assert captured["num_classes"] == 21


def test_reference_policy_selects_pinned_backbone_weight() -> None:
    from torchvision.models import MobileNet_V3_Large_Weights

    from object_detector.models.registry import get_backbone_weight

    assert get_backbone_weight("fasterrcnn_mobilenet_v3_large_320_fpn", "imagenet1k_v1") is (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
