from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from object_detector.models import torchvision_models
from object_detector.models.registry import ModelConfigError, build_model, get_model_spec, list_models


def test_external_factory_builds_and_validates_detector_contract() -> None:
    model = build_model(
        "custom_detector",
        3,
        "none",
        {"size": 320},
        factory="tests.fixtures.models:build_external_detector",
    )
    images = [torch.zeros(3, 12, 12)]
    targets = [
        {
            "boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor([1]),
        }
    ]

    model.train()
    losses = model(images, targets)
    assert isinstance(losses, dict)
    assert set(losses) == {"loss_classifier", "loss_box_reg"}

    model.eval()
    predictions = model(images)
    assert isinstance(predictions, list)
    assert predictions[0]["labels"].tolist() == [1]


def test_repository_custom_detector_factory_constructs_offline() -> None:
    model = build_model(
        "custom_fasterrcnn_mobilenet_v3_small",
        4,
        "none",
        {"width_mult": 0.25, "min_size": 64, "max_size": 64},
        factory="examples.extensions.custom_detector:build_detector",
    )

    assert isinstance(model, nn.Module)
    assert sum(parameter.numel() for parameter in model.parameters()) > 0


def test_external_factory_errors_are_reported_as_model_config_errors() -> None:
    with pytest.raises(ModelConfigError, match="expected module:function"):
        build_model("custom_detector", 3, factory="invalid")


def test_local_factory_module_loads_from_current_working_directory(tmp_path, monkeypatch) -> None:
    (tmp_path / "local_detector.py").write_text(
        "from tests.fixtures.models import FakeDetector\n"
        "def build(*, num_classes, weights):\n"
        "    return FakeDetector()\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    model = build_model("local", 3, factory="local_detector:build")

    assert isinstance(model, nn.Module)


def test_registry_metadata_import_is_lightweight_in_fresh_process() -> None:
    script = """
import json
import sys

from object_detector.models.registry import get_model_spec, list_models

assert get_model_spec(list_models()[0]).description
forbidden = ("matplotlib", "numpy", "pycocotools", "torch", "torchmetrics", "torchvision")
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden)
)
print(json.dumps(loaded))
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert json.loads(result.stdout) == []
    assert result.stderr == ""


def test_package_public_exports_remain_available() -> None:
    import object_detector.data as data
    import object_detector.evaluation as evaluation
    import object_detector.models as models

    expected_exports = {
        data: [
            "VOC_CLASSES",
            "DatasetMetadata",
            "ManifestError",
            "VocDetectionDataset",
            "VocAnnotation",
            "VocFormatError",
            "VocObject",
            "detection_collate",
            "parse_voc_annotation",
            "prepare_coco",
            "prepare_voc2007",
            "render_detection_preview",
        ],
        evaluation: ["ComparisonReport", "DetectionMetric", "compare_runs"],
        models: ["ModelConfigError", "build_model", "get_backbone_weight", "list_models", "torchvision_models"],
    }
    for package, expected in expected_exports.items():
        assert package.__all__ == expected
        assert all(hasattr(package, name) for name in expected)
    assert models.torchvision_models is torchvision_models


def test_registry_contains_exact_initial_models() -> None:
    assert set(list_models()) == {
        "fasterrcnn_mobilenet_v3_large_320_fpn",
        "fasterrcnn_resnet50_fpn",
        "fcos_resnet50_fpn",
        "retinanet_resnet50_fpn",
        "ssdlite320_mobilenet_v3_large",
    }


def test_registry_exposes_user_facing_model_metadata() -> None:
    for name in list_models():
        spec = get_model_spec(name)
        assert spec.description
        assert spec.parameters
        assert spec.input_notes


def test_unknown_model_suggests_close_match() -> None:
    with pytest.raises(ModelConfigError, match="did you mean 'fasterrcnn_resnet50_fpn'"):
        get_model_spec("fasterrcnn_resnet50_fp")


@pytest.mark.parametrize(
    ("model_name", "parameter", "suggestion"),
    [
        ("fasterrcnn_mobilenet_v3_large_320_fpn", "min_sze", "min_size"),
        ("retinanet_resnet50_fpn", "topk_canddates", "topk_candidates"),
        ("ssdlite320_mobilenet_v3_large", "score_thres", "score_thresh"),
    ],
)
def test_unknown_model_parameter_suggests_maintained_key(
    model_name: str,
    parameter: str,
    suggestion: str,
) -> None:
    with pytest.raises(ModelConfigError, match=rf"unknown model param.*{parameter}.*did you mean '{suggestion}'"):
        build_model(model_name, 21, "none", {parameter: 0.5})


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


@pytest.mark.parametrize(
    ("name", "constructor_name"),
    [
        ("retinanet_resnet50_fpn", "retinanet_resnet50_fpn"),
        ("fcos_resnet50_fpn", "fcos_resnet50_fpn"),
    ],
)
def test_resnet_one_stage_models_construct_offline(
    name: str,
    constructor_name: str,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_constructor(**kwargs: object) -> nn.Module:
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(torchvision_models.detection_models, constructor_name, fake_constructor)

    build_model(name, 21, "none", {})

    assert captured["weights"] is None
    assert captured["weights_backbone"] is None
    assert captured["num_classes"] == 21


def test_reference_policy_selects_pinned_backbone_weight() -> None:
    from torchvision.models import MobileNet_V3_Large_Weights

    from object_detector.models.registry import get_backbone_weight

    assert get_backbone_weight("fasterrcnn_mobilenet_v3_large_320_fpn", "imagenet1k_v1") is (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
