from pathlib import Path

import torch
from PIL import Image

import object_detector.evaluation.visualization as visualization


def test_evidence_visualization_writes_legend_for_empty_predictions(tmp_path: Path) -> None:
    image = torch.zeros((3, 24, 32))
    target = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "iscrowd": torch.empty((0,), dtype=torch.int64),
    }
    prediction = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "scores": torch.empty((0,)),
    }
    output = tmp_path / "evidence.png"

    visualization.render_detection_evidence(
        image,
        prediction,
        target,
        ("background", "dog"),
        output,
        score_threshold=0.5,
    )

    with Image.open(output) as rendered:
        assert rendered.mode == "RGB"
        assert rendered.width == 32
        assert rendered.height > 24
    assert output.stat().st_size > 100
