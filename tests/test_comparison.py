from __future__ import annotations

import csv
import importlib
from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml


def _comparison():
    return importlib.import_module("object_detector.evaluation.comparison")


def _write_run(
    root: Path,
    name: str,
    *,
    model: str,
    identity: str = "shared-manifest",
    rows: tuple[tuple[int, str, str], ...] = ((1, "0.1", "2.0"),),
) -> Path:
    run = root / name
    run.mkdir()
    (run / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"name": model, "weights": "none"},
                "train": {"epochs": len(rows), "lr": 0.005},
                "device": "cpu",
                "output_dir": str(root),
                "run_name": name,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (run / "run.yaml").write_text(
        yaml.safe_dump({"manifest_identity": identity, "device": "cpu"}, sort_keys=False),
        encoding="utf-8",
    )
    with (run / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "valid_map_50_95", "loss_total"])
        writer.writeheader()
        for epoch, metric, loss in rows:
            writer.writerow({"epoch": epoch, "valid_map_50_95": metric, "loss_total": loss})
    return run


def test_compare_runs_selects_and_orders_best_metric_rows(tmp_path: Path) -> None:
    first = _write_run(
        tmp_path,
        "first",
        model="fasterrcnn_resnet50_fpn",
        rows=((1, "0.2", "2.0"), (2, "0.3", "1.5")),
    )
    second = _write_run(
        tmp_path,
        "second",
        model="ssdlite320_mobilenet_v3_large",
        rows=((1, "0.4", "2.5"), (2, "0.35", "1.2")),
    )

    report = _comparison().compare_runs([first, second], metric="valid_map_50_95")

    assert report.metric == "valid_map_50_95"
    assert report.manifest_identity == "shared-manifest"
    assert [(row.run_name, row.epoch, row.metric_value) for row in report.rows] == [
        ("second", 1, 0.4),
        ("first", 2, 0.3),
    ]
    assert report.config_differences["model.name"] == (
        "ssdlite320_mobilenet_v3_large",
        "fasterrcnn_resnet50_fpn",
    )
    assert "run_name" not in report.config_differences


def test_compare_runs_aligns_config_differences_with_reverse_rank_order(tmp_path: Path) -> None:
    lower_ranked = _write_run(
        tmp_path,
        "lower-ranked",
        model="model-lower",
        rows=((1, "0.2", "2.0"),),
    )
    higher_ranked = _write_run(
        tmp_path,
        "higher-ranked",
        model="model-higher",
        rows=((1, "0.8", "1.0"),),
    )

    report = _comparison().compare_runs(
        [lower_ranked, higher_ranked],
        metric="valid_map_50_95",
    )

    assert [row.run_name for row in report.rows] == ["higher-ranked", "lower-ranked"]
    assert report.config_differences["model.name"] == ("model-higher", "model-lower")
    assert '  model.name: higher-ranked="model-higher" | lower-ranked="model-lower"' in _comparison().format_comparison(
        report
    )


def test_compare_runs_treats_loss_as_lower_is_better(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "first", model="model-a", rows=((1, "0.2", "2.0"), (2, "0.3", "1.5")))
    second = _write_run(tmp_path, "second", model="model-b", rows=((1, "0.4", "1.2"),))

    report = _comparison().compare_runs([first, second], metric="loss_total")

    assert [(row.run_name, row.epoch, row.metric_value) for row in report.rows] == [
        ("second", 1, 1.2),
        ("first", 2, 1.5),
    ]


def test_compare_runs_rejects_different_manifest_identities(tmp_path: Path) -> None:
    first = _write_run(tmp_path, "first", model="model-a", identity="manifest-a")
    second = _write_run(tmp_path, "second", model="model-b", identity="manifest-b")

    with pytest.raises(ValueError, match="manifest identities differ"):
        _comparison().compare_runs([first, second], metric="valid_map_50_95")


def test_compare_runs_reports_missing_and_nonfinite_metrics(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "run", model="model-a")

    with pytest.raises(ValueError, match="missing metric column 'valid_map_75'"):
        _comparison().compare_runs([run], metric="valid_map_75")

    (run / "metrics.csv").write_text("epoch,valid_map_50_95\n1,nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be finite"):
        _comparison().compare_runs([run], metric="valid_map_50_95")


def test_comparison_csv_is_atomic_and_does_not_overwrite(tmp_path: Path) -> None:
    run = _write_run(tmp_path, "run", model="model-a")
    report = _comparison().compare_runs([run], metric="valid_map_50_95")
    output = tmp_path / "comparison.csv"

    assert _comparison().write_comparison_csv(report, output) == output
    assert output.read_text(encoding="utf-8").splitlines()[0] == (
        "run_name,run_dir,model_name,epoch,metric,metric_value,device,manifest_identity"
    )
    with pytest.raises(FileExistsError, match="comparison output already exists"):
        _comparison().write_comparison_csv(report, output)


def test_comparison_csv_refuses_destination_created_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_run(tmp_path, "run", model="model-a")
    comparison = _comparison()
    report = comparison.compare_runs([run], metric="valid_map_50_95")
    output = tmp_path / "comparison.csv"
    competitor_content = b"competitor content\n"
    temporary_paths: list[Path] = []
    named_temporary_file = comparison.tempfile.NamedTemporaryFile

    @contextmanager
    def create_competitor_after_close(*args: object, **kwargs: object):
        with named_temporary_file(*args, **kwargs) as handle:
            temporary_paths.append(Path(handle.name))
            yield handle
        output.write_bytes(competitor_content)

    monkeypatch.setattr(comparison.tempfile, "NamedTemporaryFile", create_competitor_after_close)

    with pytest.raises(FileExistsError) as error:
        comparison.write_comparison_csv(report, output)

    assert str(error.value) == f"comparison output already exists: {output}"
    assert output.read_bytes() == competitor_content
    assert temporary_paths and all(not path.exists() for path in temporary_paths)


def test_comparison_csv_preserves_publication_error_when_temp_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_run(tmp_path, "run", model="model-a")
    comparison = _comparison()
    report = comparison.compare_runs([run], metric="valid_map_50_95")
    output = tmp_path / "comparison.csv"
    competitor_content = b"competitor content\n"
    temporary_paths: list[Path] = []
    named_temporary_file = comparison.tempfile.NamedTemporaryFile
    unlink = Path.unlink

    @contextmanager
    def create_competitor_after_close(*args: object, **kwargs: object):
        with named_temporary_file(*args, **kwargs) as handle:
            temporary_paths.append(Path(handle.name))
            yield handle
        output.write_bytes(competitor_content)

    def fail_temporary_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path in temporary_paths:
            raise OSError("temporary cleanup failed")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(comparison.tempfile, "NamedTemporaryFile", create_competitor_after_close)
    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)

    with pytest.raises(FileExistsError) as error:
        comparison.write_comparison_csv(report, output)

    assert str(error.value) == f"comparison output already exists: {output}"
    assert output.read_bytes() == competitor_content


def test_comparison_csv_returns_published_output_when_temp_cleanup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_run(tmp_path, "run", model="model-a")
    comparison = _comparison()
    report = comparison.compare_runs([run], metric="valid_map_50_95")
    output = tmp_path / "comparison.csv"
    temporary_paths: list[Path] = []
    named_temporary_file = comparison.tempfile.NamedTemporaryFile
    unlink = Path.unlink

    @contextmanager
    def capture_temporary_path(*args: object, **kwargs: object):
        with named_temporary_file(*args, **kwargs) as handle:
            temporary_paths.append(Path(handle.name))
            yield handle

    def fail_temporary_unlink(path: Path, *, missing_ok: bool = False) -> None:
        if path in temporary_paths:
            raise OSError("temporary cleanup failed")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(comparison.tempfile, "NamedTemporaryFile", capture_temporary_path)
    monkeypatch.setattr(Path, "unlink", fail_temporary_unlink)

    assert comparison.write_comparison_csv(report, output) == output
    expected = (
        "run_name,run_dir,model_name,epoch,metric,metric_value,device,manifest_identity\n"
        f"run,{run.as_posix()},model-a,1,valid_map_50_95,0.1,cpu,shared-manifest\n"
    ).encode()
    assert output.read_bytes() == expected
