from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch.nn as nn

from object_detector.models import torchvision_models
from object_detector.models.registry import ModelConfigError, build_model, get_model_spec, list_models


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


def test_reference_policy_selects_pinned_backbone_weight() -> None:
    from torchvision.models import MobileNet_V3_Large_Weights

    from object_detector.models.registry import get_backbone_weight

    assert get_backbone_weight("fasterrcnn_mobilenet_v3_large_320_fpn", "imagenet1k_v1") is (
        MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
