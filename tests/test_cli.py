from pathlib import Path
from types import SimpleNamespace

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
