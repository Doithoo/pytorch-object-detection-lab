from pathlib import Path
from types import SimpleNamespace

import pytest

import object_detector.cli as cli
from object_detector.cli import build_parser, main
from tests.fixtures.voc import build_voc_tree


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

    monkeypatch.setattr(cli, "run_training", fake_run_training, raising=False)

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

    monkeypatch.setattr(cli, "evaluate_checkpoint", fake_evaluate_checkpoint, raising=False)

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
    monkeypatch.setattr(cli, "Predictor", fake_type, raising=False)

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


@pytest.mark.parametrize("command", ["show-config", "prepare-data", "train", "evaluate", "predict"])
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
