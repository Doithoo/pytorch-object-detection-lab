from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
EXAMPLES = [
    ROOT / "examples" / f"0{index}_{name}.py"
    for index, name in enumerate(
        ("boxes_and_labels", "detection_batch", "detector_losses", "minimal_training_loop", "checkpoint_prediction"),
        start=1,
    )
]


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda path: path.name)
def test_example_help_is_offline_and_executable(example: Path, tmp_path: Path) -> None:
    environment = {**os.environ, "TORCH_HOME": str(tmp_path / "torch"), "MPLCONFIGDIR": str(tmp_path / "mpl")}
    result = subprocess.run(
        [sys.executable, str(example), "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_plot_metrics_writes_png(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.csv"
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "loss_total", "valid_map_50_95"])
        writer.writeheader()
        writer.writerows(
            [
                {"epoch": 1, "loss_total": 2.0, "valid_map_50_95": 0.1},
                {"epoch": 2, "loss_total": 1.0, "valid_map_50_95": 0.2},
            ]
        )
    output = tmp_path / "metrics.png"
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "plot_metrics.py"), "--metrics", str(metrics), "--output", str(output)],
        cwd=ROOT,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert output.is_file()
    assert output.stat().st_size > 100
