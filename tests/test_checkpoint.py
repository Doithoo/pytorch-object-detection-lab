from __future__ import annotations

from pathlib import Path

import pytest
import torch

from object_detector.training.checkpoint import (
    CheckpointCompatibilityError,
    ResumeIdentity,
    build_run_metadata,
    load_checkpoint,
    save_checkpoint,
    validate_resume_identity,
)


def checkpoint_payload(manifest_identity: str = "manifest") -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {"name": "fake", "params": {}},
        "class_names": ["background", "dog"],
        "preprocessing": {"resize_owner": "model"},
        "manifest_identity": manifest_identity,
        "model_state": {"weight": torch.tensor([1.0])},
        "optimizer_state": {},
        "scheduler_state": None,
        "epoch": 1,
        "best_metric": 0.5,
        "metric_history": [],
    }


def _write_marker(path: str) -> str:
    Path(path).write_text("executed", encoding="utf-8")
    return "executed"


class MaliciousPayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return _write_marker, (str(self.marker),)


def test_checkpoint_round_trip_is_atomic(tmp_path: Path) -> None:
    path = tmp_path / "last.pt"

    save_checkpoint(path, checkpoint_payload())
    loaded = load_checkpoint(path)

    assert loaded["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_rejects_manifest_change() -> None:
    checkpoint = checkpoint_payload(manifest_identity="old")
    expected = ResumeIdentity(
        model_name="fake",
        class_names=("background", "dog"),
        manifest_identity="new",
        preprocessing={"resize_owner": "model"},
    )

    with pytest.raises(CheckpointCompatibilityError, match="manifest_identity"):
        validate_resume_identity(checkpoint, expected)


def test_save_failure_leaves_no_final_or_temporary_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "last.pt"

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint(path, checkpoint_payload())
    assert not path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_run_metadata_records_framework_versions() -> None:
    metadata = build_run_metadata(device=torch.device("cpu"), seed=42)

    assert metadata["torch"] == torch.__version__
    assert isinstance(metadata["torchvision"], str)


def test_checkpoint_loader_does_not_execute_pickle_globals(tmp_path: Path) -> None:
    checkpoint = tmp_path / "malicious.pt"
    marker = tmp_path / "executed.txt"
    torch.save({"schema_version": 1, "payload": MaliciousPayload(marker)}, checkpoint)

    with pytest.raises(CheckpointCompatibilityError, match="safe tensor-only checkpoint"):
        load_checkpoint(checkpoint)

    assert not marker.exists()
