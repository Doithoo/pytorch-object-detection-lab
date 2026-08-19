import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest

from object_detector.cli import build_parser, main
from tests.fixtures.voc import build_voc_tree

_HEAVY_IMPORT_PREFIXES = (
    "matplotlib",
    "numpy",
    "pycocotools",
    "torch",
    "torchmetrics",
    "torchvision",
)


class _CliProbe(NamedTuple):
    command_returncode: int
    loaded_modules: list[str]
    stderr: str


def _loaded_modules_after_cli_dispatch(
    arguments: list[str],
    *,
    unrelated_prefixes: tuple[str, ...],
) -> _CliProbe:
    script = """
import contextlib
import io
import json
import sys

from object_detector.cli import main

arguments = json.loads(sys.argv[1])
prefixes = tuple(json.loads(sys.argv[2]))
with contextlib.redirect_stdout(io.StringIO()):
    try:
        command_returncode = main(arguments)
    except SystemExit as exc:
        command_returncode = int(exc.code or 0)
loaded = sorted(
    name
    for name in sys.modules
    if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
)
print(json.dumps({"command_returncode": command_returncode, "loaded_modules": loaded}))
"""
    prefixes = (*_HEAVY_IMPORT_PREFIXES, *unrelated_prefixes)
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(arguments), json.dumps(prefixes)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return _CliProbe(
        command_returncode=payload["command_returncode"],
        loaded_modules=payload["loaded_modules"],
        stderr=result.stderr,
    )


def test_help_dispatch_does_not_import_command_implementation_stacks() -> None:
    probe = _loaded_modules_after_cli_dispatch(
        ["--help"],
        unrelated_prefixes=(
            "object_detector.data.inspection",
            "object_detector.data.manifest",
            "object_detector.evaluation",
            "object_detector.inference",
            "object_detector.models.registry",
            "object_detector.training",
        ),
    )

    assert probe.command_returncode == 0
    assert probe.loaded_modules == []
    assert probe.stderr == ""


def test_list_models_dispatch_keeps_unrelated_stacks_unloaded() -> None:
    probe = _loaded_modules_after_cli_dispatch(
        ["list-models"],
        unrelated_prefixes=(
            "object_detector.data",
            "object_detector.evaluation",
            "object_detector.inference",
            "object_detector.models.torchvision_models",
            "object_detector.training",
        ),
    )

    assert probe.command_returncode == 0
    assert probe.loaded_modules == []
    assert probe.stderr == ""


def test_model_info_dispatch_keeps_unrelated_stacks_unloaded() -> None:
    probe = _loaded_modules_after_cli_dispatch(
        ["model-info", "fasterrcnn_resnet50_fpn"],
        unrelated_prefixes=(
            "object_detector.data",
            "object_detector.evaluation",
            "object_detector.inference",
            "object_detector.models.torchvision_models",
            "object_detector.training",
        ),
    )

    assert probe.command_returncode == 0
    assert probe.loaded_modules == []
    assert probe.stderr == ""


def test_prepare_data_dispatch_keeps_unrelated_stacks_unloaded(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    build_voc_tree(data_dir)
    probe = _loaded_modules_after_cli_dispatch(
        [
            "prepare-data",
            "--data-dir",
            str(data_dir),
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--allow-nonstandard-counts",
        ],
        unrelated_prefixes=(
            "object_detector.data.dataset",
            "object_detector.data.inspection",
            "object_detector.evaluation",
            "object_detector.inference",
            "object_detector.models",
            "object_detector.training",
        ),
    )

    assert probe.command_returncode == 0
    assert probe.loaded_modules == []
    assert probe.stderr == ""


def test_compare_runs_dispatch_keeps_unrelated_stacks_unloaded(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(
        "model:\n  name: fasterrcnn_resnet50_fpn\nrun_name: run\n",
        encoding="utf-8",
    )
    (run / "run.yaml").write_text("manifest_identity: shared\ndevice: cpu\n", encoding="utf-8")
    (run / "metrics.csv").write_text("epoch,valid_map_50_95\n1,0.25\n", encoding="utf-8")
    probe = _loaded_modules_after_cli_dispatch(
        ["compare-runs", str(run), "--metric", "valid_map_50_95"],
        unrelated_prefixes=(
            "object_detector.data",
            "object_detector.evaluation.evaluate",
            "object_detector.evaluation.metrics",
            "object_detector.inference",
            "object_detector.models",
            "object_detector.training",
        ),
    )

    assert probe.command_returncode == 0
    assert probe.loaded_modules == []
    assert probe.stderr == ""


def test_prepare_data_prints_identity_and_counts(tmp_path: Path, capsys) -> None:
    build_voc_tree(tmp_path / "raw")

    result = main(
        [
            "prepare-data",
            "--data-dir",
            str(tmp_path / "raw"),
            "--manifest-dir",
            str(tmp_path / "manifests"),
            "--allow-nonstandard-counts",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "train=2 valid=1 test=1" in output
    assert "identity=" in output


def test_list_models_prints_stable_registry_metadata(capsys) -> None:
    result = main(["list-models"])

    output = capsys.readouterr().out.splitlines()
    assert result == 0
    assert output[0] == "name\tfamily\tweights"
    assert output[1:] == [
        "fasterrcnn_mobilenet_v3_large_320_fpn\ttwo_stage\tnone,imagenet1k_v1",
        "fasterrcnn_resnet50_fpn\ttwo_stage\tnone,imagenet1k_v1",
        "ssdlite320_mobilenet_v3_large\tone_stage\tnone,imagenet1k_v1",
    ]


def test_model_info_prints_metadata_without_constructing_model(capsys) -> None:
    result = main(["model-info", "fasterrcnn_resnet50_fpn"])

    output = capsys.readouterr().out
    assert result == 0
    assert "name: fasterrcnn_resnet50_fpn\n" in output
    assert "family: two_stage\n" in output
    assert "description:" in output
    assert "weights: none, imagenet1k_v1\n" in output
    assert "parameters:\n" in output
    assert "input_notes:\n" in output


def test_inspect_data_prints_a_serializable_report(prepared_voc, capsys) -> None:
    result = main(
        [
            "inspect-data",
            "--manifest-dir",
            str(prepared_voc.manifests),
            "--data-dir",
            str(prepared_voc.voc_root.parent.parent),
            "--split",
            "train",
            "--limit",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert result == 0
    assert "dataset: voc2007\n" in output
    assert f"identity: {prepared_voc.metadata.identity}\n" in output
    assert "total_images: 2\n" in output
    assert "inspected_images: 1\n" in output


def test_compare_runs_prints_a_factual_table(tmp_path: Path, capsys) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "config.yaml").write_text(
        "model:\n  name: fasterrcnn_resnet50_fpn\nrun_name: run\n",
        encoding="utf-8",
    )
    (run / "run.yaml").write_text("manifest_identity: shared\ndevice: cpu\n", encoding="utf-8")
    (run / "metrics.csv").write_text("epoch,valid_map_50_95\n1,0.25\n", encoding="utf-8")

    result = main(["compare-runs", str(run), "--metric", "valid_map_50_95"])

    output = capsys.readouterr().out
    assert result == 0
    assert "metric: valid_map_50_95\n" in output
    assert "run\tfasterrcnn_resnet50_fpn\t1\t0.25\tcpu\n" in output


def test_train_parser_accepts_runtime_controls() -> None:
    args = build_parser().parse_args(
        [
            "train",
            "--config",
            "configs/learning_minimal.yaml",
            "--set",
            "train.epochs",
            "1",
            "--dry-run",
            "--resume",
            "last.pt",
            "--device",
            "cpu",
        ]
    )

    assert args.dry_run is True
    assert args.overrides == [["train.epochs", "1"]]


def test_train_dry_run_prints_diagnostics_and_applies_device(tmp_path: Path, capsys, monkeypatch) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text("train:\n  epochs: 1\n", encoding="utf-8")
    captured = {}

    def fake_run_training(config, *, resume, dry_run_mode):
        captured.update(config=config, resume=resume, dry_run_mode=dry_run_mode)
        return SimpleNamespace(
            run_dir=tmp_path / "artifacts" / "run",
            dry_run_result=SimpleNamespace(
                image_shapes=((3, 24, 32),),
                target_counts=(2,),
                losses={"loss_total": 1.25, "loss_classifier": 0.5},
            ),
        )

    monkeypatch.setattr("object_detector.training.train.run_training", fake_run_training)

    result = main(["train", "--config", str(config_path), "--dry-run", "--device", "cpu"])

    output = capsys.readouterr().out
    assert result == 0
    assert captured["config"].device == "cpu"
    assert captured["resume"] is None
    assert captured["dry_run_mode"] is True
    assert "image_shapes=((3, 24, 32),)" in output
    assert "target_counts=(2,)" in output
    assert "loss_total=1.25" in output
    assert output.endswith("dry-run OK\n")


def test_train_rejects_dry_run_with_resume_as_a_usage_error(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "run.yaml"
    config_path.write_text("train:\n  epochs: 1\n", encoding="utf-8")

    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--dry-run",
            "--resume",
            str(tmp_path / "missing.pt"),
        ]
    )

    stderr = capsys.readouterr().err
    assert result == 2
    assert "--dry-run cannot be combined with --resume" in stderr
    assert "Traceback" not in stderr


def test_evaluate_parser_accepts_checkpoint_runtime_controls() -> None:
    args = build_parser().parse_args(
        [
            "evaluate",
            "--checkpoint",
            "best.pt",
            "--split",
            "test",
            "--output-dir",
            "evaluation",
            "--device",
            "cpu",
            "--score-threshold",
            "0.25",
            "--overwrite",
        ]
    )

    assert args.checkpoint == Path("best.pt")
    assert args.split == "test"
    assert args.score_threshold == 0.25
    assert args.overwrite is True


def test_evaluate_handler_reports_output_directory(tmp_path: Path, capsys, monkeypatch) -> None:
    captured = {}
    output = tmp_path / "evaluation"

    def fake_evaluate_checkpoint(checkpoint, **kwargs):
        captured.update(checkpoint=checkpoint, **kwargs)
        return SimpleNamespace(output_dir=output)

    monkeypatch.setattr("object_detector.evaluation.evaluate.evaluate_checkpoint", fake_evaluate_checkpoint)

    result = main(
        [
            "evaluate",
            "--checkpoint",
            "best.pt",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert captured["checkpoint"] == Path("best.pt")
    assert captured["split"] == "test"
    assert captured["device"] == "cpu"
    assert capsys.readouterr().out == f"{output}\n"


def test_predict_parser_requires_exactly_one_input_mode() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "predict",
            "--checkpoint",
            "best.pt",
            "--image",
            "sample.jpg",
            "--output-dir",
            "predictions",
            "--display-limit",
            "5",
        ]
    )

    assert args.image == Path("sample.jpg")
    assert args.input_dir is None
    assert args.display_limit == 5
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "predict",
                "--checkpoint",
                "best.pt",
                "--image",
                "sample.jpg",
                "--input-dir",
                "images",
                "--output-dir",
                "predictions",
            ]
        )


def test_predict_handler_runs_single_mode(tmp_path: Path, capsys, monkeypatch) -> None:
    captured = {}
    image = tmp_path / "sample.jpg"
    output = tmp_path / "predictions"

    class FakePredictor:
        def predict_single(self, image_path, output_dir, **kwargs):
            captured.update(image_path=image_path, output_dir=output_dir, **kwargs)

    fake_type = SimpleNamespace(from_checkpoint=lambda checkpoint, device: FakePredictor())
    monkeypatch.setattr("object_detector.inference.predictor.Predictor", fake_type)

    result = main(
        [
            "predict",
            "--checkpoint",
            "best.pt",
            "--image",
            str(image),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    assert captured["image_path"] == image
    assert captured["output_dir"] == output
    assert captured["score_threshold"] == 0.5
    assert capsys.readouterr().out == f"{output}\n"


@pytest.mark.parametrize(
    "command",
    [
        "show-config",
        "list-models",
        "model-info",
        "inspect-data",
        "compare-runs",
        "prepare-data",
        "train",
        "evaluate",
        "predict",
    ],
)
def test_every_subcommand_has_help(command: str, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([command, "--help"])

    assert exit_info.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_known_user_errors_return_nonzero_without_traceback(tmp_path: Path, capsys) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("train:\n  epochz: 1\n", encoding="utf-8")

    result = main(["show-config", "--config", str(bad_config)])

    stderr = capsys.readouterr().err
    assert result == 2
    assert "unknown configuration field: train.epochz" in stderr
    assert "Traceback" not in stderr


def test_missing_prediction_checkpoint_is_a_concise_error(tmp_path: Path, capsys) -> None:
    result = main(
        [
            "predict",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--image",
            str(tmp_path / "image.jpg"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    stderr = capsys.readouterr().err
    assert result == 2
    assert "cannot load checkpoint" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["evaluate", "--checkpoint", "model.pt", "--output-dir", "out", "--score-threshold", "nan"],
        ["evaluate", "--checkpoint", "model.pt", "--output-dir", "out", "--score-threshold", "1.1"],
        [
            "predict",
            "--checkpoint",
            "model.pt",
            "--image",
            "image.jpg",
            "--output-dir",
            "out",
            "--score-threshold",
            "-0.1",
        ],
        [
            "predict",
            "--checkpoint",
            "model.pt",
            "--image",
            "image.jpg",
            "--output-dir",
            "out",
            "--display-limit",
            "-1",
        ],
        ["inspect-data", "--manifest-dir", "manifests", "--limit", "0"],
    ],
)
def test_cli_rejects_invalid_thresholds_before_dispatch(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)
