import torch

import object_detector.evaluation.errors as errors


def test_error_analysis_matches_greedily_and_handles_difficult_targets() -> None:
    target = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        ),
        "labels": torch.tensor([1, 1, 1]),
        "iscrowd": torch.tensor([0, 0, 1]),
    }
    prediction = {
        "boxes": torch.tensor(
            [
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 24.0, 30.0],
                [60.0, 60.0, 70.0, 70.0],
                [40.0, 40.0, 50.0, 50.0],
            ]
        ),
        "labels": torch.tensor([1, 1, 1, 1]),
        "scores": torch.tensor([0.9, 0.8, 0.7, 0.6]),
    }

    result = errors.analyze_image_errors(
        "sample",
        prediction,
        target,
        ("background", "dog"),
        score_threshold=0.5,
        iou_threshold=0.5,
    )

    assert [item.kind for item in result] == ["localization", "false_positive", "ignored", "missed"]
    assert result[0].iou == torch.tensor(0.4).item()
    assert result[0].score == torch.tensor(0.8).item()
    assert result[1].iou == 0.0
    assert result[2].iou == 1.0
    assert result[3].score is None
    assert result[3].box == (20.0, 20.0, 30.0, 30.0)


def test_equal_scores_keep_original_prediction_order() -> None:
    prediction = {
        "boxes": torch.tensor([[20.0, 20.0, 21.0, 21.0], [10.0, 10.0, 11.0, 11.0]]),
        "labels": torch.tensor([1, 1]),
        "scores": torch.tensor([0.7, 0.7]),
    }
    target = {
        "boxes": torch.empty((0, 4)),
        "labels": torch.empty((0,), dtype=torch.int64),
        "iscrowd": torch.empty((0,), dtype=torch.int64),
    }

    result = errors.analyze_image_errors(
        "sample",
        prediction,
        target,
        ("background", "dog"),
        score_threshold=0.5,
        iou_threshold=0.5,
    )

    assert [item.box for item in result] == [
        (20.0, 20.0, 21.0, 21.0),
        (10.0, 10.0, 11.0, 11.0),
    ]
