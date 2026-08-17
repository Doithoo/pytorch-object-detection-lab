from pathlib import Path

from object_detector.cli import main
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
