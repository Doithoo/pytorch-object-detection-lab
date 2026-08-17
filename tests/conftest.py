from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from object_detector.data.manifest import DatasetMetadata, prepare_voc2007
from tests.fixtures.voc import build_voc_tree


@dataclass(frozen=True)
class PreparedVoc:
    voc_root: Path
    manifests: Path
    metadata: DatasetMetadata


@pytest.fixture
def prepared_voc(tmp_path: Path) -> PreparedVoc:
    voc_root = build_voc_tree(tmp_path / "raw")
    manifests = tmp_path / "manifests"
    metadata = prepare_voc2007(voc_root.parent.parent, manifests, expected_split_counts=None)
    return PreparedVoc(voc_root, manifests, metadata)
