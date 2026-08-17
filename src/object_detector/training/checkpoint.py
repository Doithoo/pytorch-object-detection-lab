from __future__ import annotations

import os
import pickle
import platform
import subprocess
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot safely restore a requested run."""


@dataclass(frozen=True)
class ResumeIdentity:
    model_name: str
    class_names: tuple[str, ...]
    manifest_identity: str
    preprocessing: Mapping[str, object]


def save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    serialized = dict(payload)
    serialized.setdefault("schema_version", CHECKPOINT_SCHEMA_VERSION)
    try:
        torch.save(serialized, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, object]:
    try:
        loaded = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError as exc:
        raise CheckpointCompatibilityError(f"cannot load safe tensor-only checkpoint {path}: {exc}") from exc
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointCompatibilityError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise CheckpointCompatibilityError(f"checkpoint {path} must contain a mapping")
    if loaded.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"checkpoint {path} has unsupported schema_version {loaded.get('schema_version')!r}"
        )
    return loaded


def validate_resume_identity(checkpoint: Mapping[str, object], expected: ResumeIdentity) -> None:
    model_data = checkpoint.get("model")
    checkpoint_classes = checkpoint.get("class_names")
    checkpoint_preprocessing = checkpoint.get("preprocessing")
    mismatches: list[str] = []
    if not isinstance(model_data, Mapping) or model_data.get("name") != expected.model_name:
        mismatches.append("model_name")
    normalized_classes = tuple(checkpoint_classes) if isinstance(checkpoint_classes, (tuple, list)) else ()
    if normalized_classes != expected.class_names:
        mismatches.append("class_names")
    if checkpoint.get("manifest_identity") != expected.manifest_identity:
        mismatches.append("manifest_identity")
    if not isinstance(checkpoint_preprocessing, Mapping) or dict(checkpoint_preprocessing) != dict(
        expected.preprocessing
    ):
        mismatches.append("preprocessing")
    if mismatches:
        raise CheckpointCompatibilityError("resume identity mismatch: " + ", ".join(mismatches))


def build_run_metadata(*, device: torch.device, seed: int) -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__),
        "platform": platform.platform(),
        "device": str(device),
        "seed": seed,
        "git_revision": _git_revision(),
    }


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()
