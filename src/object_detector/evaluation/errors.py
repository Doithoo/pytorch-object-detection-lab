from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torchvision.ops import box_iou

ErrorKind = Literal["localization", "false_positive", "missed", "ignored"]


@dataclass(frozen=True)
class DetectionError:
    image_id: str
    kind: ErrorKind
    class_name: str
    score: float | None
    iou: float
    box: tuple[float, float, float, float]


def analyze_image_errors(
    image_id: str,
    prediction: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    class_names: Sequence[str],
    score_threshold: float,
    iou_threshold: float,
) -> tuple[DetectionError, ...]:
    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    target_boxes = target["boxes"].detach().cpu()
    target_labels = target["labels"].detach().cpu()
    iscrowd = target.get("iscrowd", torch.zeros_like(target_labels)).detach().cpu()

    prediction_order = sorted(
        (index for index, score in enumerate(scores.tolist()) if score >= score_threshold),
        key=lambda index: (-float(scores[index]), index),
    )
    matched_targets: set[int] = set()
    result: list[DetectionError] = []

    for prediction_index in prediction_order:
        label = int(labels[prediction_index])
        class_name = _class_name(label, class_names)
        ordinary = [
            index
            for index in range(len(target_boxes))
            if int(target_labels[index]) == label and int(iscrowd[index]) == 0 and index not in matched_targets
        ]
        best_ordinary_index, best_ordinary_iou = _best_target(boxes[prediction_index], target_boxes, ordinary)
        if best_ordinary_index is not None and best_ordinary_iou >= iou_threshold:
            matched_targets.add(best_ordinary_index)
            continue

        difficult = [
            index
            for index in range(len(target_boxes))
            if int(target_labels[index]) == label and int(iscrowd[index]) != 0
        ]
        _, best_difficult_iou = _best_target(boxes[prediction_index], target_boxes, difficult)
        if best_difficult_iou >= iou_threshold:
            result.append(
                _prediction_error(
                    image_id,
                    "ignored",
                    class_name,
                    boxes[prediction_index],
                    scores[prediction_index],
                    best_difficult_iou,
                )
            )
        elif best_ordinary_iou > 0.0:
            result.append(
                _prediction_error(
                    image_id,
                    "localization",
                    class_name,
                    boxes[prediction_index],
                    scores[prediction_index],
                    best_ordinary_iou,
                )
            )
        else:
            result.append(
                _prediction_error(
                    image_id,
                    "false_positive",
                    class_name,
                    boxes[prediction_index],
                    scores[prediction_index],
                    0.0,
                )
            )

    for target_index, box in enumerate(target_boxes):
        if int(iscrowd[target_index]) != 0 or target_index in matched_targets:
            continue
        label = int(target_labels[target_index])
        result.append(
            DetectionError(
                image_id=image_id,
                kind="missed",
                class_name=_class_name(label, class_names),
                score=None,
                iou=0.0,
                box=_box_tuple(box),
            )
        )
    return tuple(result)


def _best_target(
    prediction_box: torch.Tensor,
    target_boxes: torch.Tensor,
    candidate_indices: Sequence[int],
) -> tuple[int | None, float]:
    if not candidate_indices:
        return None, 0.0
    ious = box_iou(prediction_box.reshape(1, 4), target_boxes[list(candidate_indices)]).reshape(-1)
    candidate_position = int(torch.argmax(ious))
    return candidate_indices[candidate_position], float(ious[candidate_position])


def _prediction_error(
    image_id: str,
    kind: ErrorKind,
    class_name: str,
    box: torch.Tensor,
    score: torch.Tensor,
    iou: float,
) -> DetectionError:
    return DetectionError(
        image_id=image_id,
        kind=kind,
        class_name=class_name,
        score=float(score),
        iou=iou,
        box=_box_tuple(box),
    )


def _class_name(class_id: int, class_names: Sequence[str]) -> str:
    if class_id < 0 or class_id >= len(class_names):
        raise ValueError(f"unknown class ID {class_id}")
    return class_names[class_id]


def _box_tuple(box: torch.Tensor) -> tuple[float, float, float, float]:
    values = box.tolist()
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])
