from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from object_detector.config import AppConfig
from object_detector.data.manifest import DatasetMetadata
from object_detector.models.registry import expected_weight_cache_path


@dataclass(frozen=True)
class PreflightIssue:
    field: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    issues: tuple[PreflightIssue, ...]
    notices: tuple[str, ...]

    def raise_for_issues(self) -> None:
        if self.issues:
            details = "\n".join(f"- {issue.field}: {issue.message}" for issue in self.issues)
            raise PreflightError(f"training preflight failed:\n{details}")


class PreflightError(ValueError):
    """Raised when a training request cannot safely start."""


def validate_training_request(config: AppConfig, metadata: DatasetMetadata) -> PreflightReport:
    issues: list[PreflightIssue] = []
    notices: list[str] = []
    required = ("train.csv", "valid.csv", "test.csv", "dataset.yaml")
    missing = [name for name in required if not (config.data.manifest_dir / name).is_file()]
    if missing:
        issues.append(PreflightIssue("data.manifest_dir", "missing " + ", ".join(missing)))

    actual_num_classes = len(metadata.class_names) + 1
    if config.model.expected_num_classes != actual_num_classes:
        issues.append(
            PreflightIssue(
                "model.expected_num_classes",
                f"expected {config.model.expected_num_classes}, dataset requires {actual_num_classes}",
            )
        )

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        issues.append(PreflightIssue("device", "CUDA was requested but is unavailable"))
    elif config.device == "mps" and not torch.backends.mps.is_available():
        issues.append(PreflightIssue("device", "MPS was requested but is unavailable"))
    elif config.device not in {"auto", "cpu", "mps"} and not config.device.startswith("cuda"):
        issues.append(PreflightIssue("device", f"unsupported device {config.device!r}"))

    if not is_writable_destination(config.output_dir):
        issues.append(PreflightIssue("output_dir", f"cannot write below {config.output_dir}"))

    if config.model.weights != "none" and config.model.factory is None:
        cached = expected_weight_cache_path(config.model.name, config.model.weights)
        if not cached.is_file():
            notices.append(
                f"{config.model.weights} is not cached at {cached}; network access is required for model construction"
            )
    if config.train.amp and config.device == "mps":
        notices.append("AMP is not enabled on MPS; training will use full precision")
    return PreflightReport(tuple(issues), tuple(notices))


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def is_writable_destination(path: Path) -> bool:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            return False
        candidate = candidate.parent
    return candidate.is_dir() and os.access(candidate, os.W_OK)
