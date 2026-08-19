from __future__ import annotations

import importlib
from pathlib import Path

from PIL import Image


def test_generate_doc_assets_is_explicit_and_reproducible(tmp_path: Path) -> None:
    module = importlib.import_module("scripts.generate_doc_assets")
    assert list(tmp_path.iterdir()) == []

    result = module.main(["--output-dir", str(tmp_path)])

    assert result == 0
    expected_sizes = {
        "detection-target-anatomy.png": (640, 432),
        "detection-error-analysis.png": (640, 448),
    }
    assert {path.name for path in tmp_path.iterdir()} == set(expected_sizes)
    for name, size in expected_sizes.items():
        path = tmp_path / name
        with Image.open(path) as image:
            assert image.mode == "RGB"
            assert image.size == size
            colors = image.getcolors(maxcolors=1_000_000)
            assert colors is not None
            assert len(colors) > 16
        assert path.stat().st_size > 1_000
