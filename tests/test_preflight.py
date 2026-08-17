from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from object_detector.config import load_config
from object_detector.preflight import validate_training_request
from tests.conftest import PreparedVoc


def test_preflight_aggregates_independent_errors(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    config = load_config()
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")
    config = replace(
        config,
        data=replace(config.data, manifest_dir=tmp_path / "missing"),
        model=replace(config.model, expected_num_classes=3),
        device="cuda",
        output_dir=blocked_parent / "artifacts",
    )

    report = validate_training_request(config, prepared_voc.metadata)

    fields = {issue.field for issue in report.issues}
    assert fields == {"data.manifest_dir", "model.expected_num_classes", "device", "output_dir"}


def test_uncached_reference_weight_is_a_notice(prepared_voc: PreparedVoc, monkeypatch, tmp_path: Path) -> None:
    config = load_config()
    config = replace(
        config,
        data=replace(config.data, manifest_dir=prepared_voc.manifests),
        model=replace(config.model, weights="imagenet1k_v1"),
        output_dir=tmp_path / "artifacts",
    )
    monkeypatch.setattr("object_detector.preflight.expected_weight_cache_path", lambda *_: tmp_path / "missing.pth")

    report = validate_training_request(config, prepared_voc.metadata)

    assert not report.issues
    assert "network access is required" in report.notices[0]
