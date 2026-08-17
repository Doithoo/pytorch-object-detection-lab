from pathlib import Path

from PIL import Image

from object_detector.data.dataset import VocDetectionDataset
from object_detector.data.inspection import render_detection_preview
from tests.conftest import PreparedVoc


def test_preview_writes_a_nonempty_rgb_image(prepared_voc: PreparedVoc, tmp_path: Path) -> None:
    dataset = VocDetectionDataset.from_manifests(prepared_voc.manifests, "train", training=False)
    output = tmp_path / "preview.png"

    render_detection_preview([dataset[0], dataset[1]], ("background", *prepared_voc.metadata.class_names), output)

    with Image.open(output) as preview:
        assert preview.mode == "RGB"
        assert preview.width > 20
        assert preview.height >= 10
    assert output.stat().st_size > 100
