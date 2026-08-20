from pathlib import Path

import pytest

from object_detector.data.voc import VocFormatError, parse_voc_annotation


def test_voc_box_becomes_zero_based_xyxy(tmp_path: Path) -> None:
    xml = tmp_path / "sample.xml"
    xml.write_text(
        "<annotation><filename>x.jpg</filename><size><width>20</width>"
        "<height>10</height><depth>3</depth></size><object><name>dog</name>"
        "<difficult>1</difficult><bndbox><xmin>1</xmin><ymin>2</ymin>"
        "<xmax>20</xmax><ymax>10</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )

    annotation = parse_voc_annotation(xml)

    assert annotation.filename == "x.jpg"
    assert annotation.size == (20, 10)
    assert annotation.objects[0].box == (0.0, 1.0, 20.0, 10.0)
    assert annotation.objects[0].difficult is True


def test_degenerate_box_is_rejected(tmp_path: Path) -> None:
    xml = tmp_path / "bad.xml"
    xml.write_text(
        "<annotation><filename>x.jpg</filename><size><width>20</width>"
        "<height>10</height><depth>3</depth></size><object><name>dog</name>"
        "<bndbox><xmin>8</xmin><ymin>2</ymin><xmax>7</xmax>"
        "<ymax>6</ymax></bndbox></object></annotation>",
        encoding="utf-8",
    )

    with pytest.raises(VocFormatError, match="positive width"):
        parse_voc_annotation(xml)


def test_custom_class_names_can_be_allowed_explicitly(tmp_path: Path) -> None:
    xml = tmp_path / "custom.xml"
    xml.write_text(
        "<annotation><filename>x.jpg</filename><size><width>20</width>"
        "<height>10</height></size><object><name>dragon</name><bndbox>"
        "<xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )

    annotation = parse_voc_annotation(xml, allowed_classes=("dragon",))

    assert annotation.objects[0].class_name == "dragon"

    xml = tmp_path / "unknown.xml"
    xml.write_text(
        "<annotation><filename>x.jpg</filename><size><width>20</width>"
        "<height>10</height></size><object><name>dragon</name><bndbox>"
        "<xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax>"
        "</bndbox></object></annotation>",
        encoding="utf-8",
    )

    with pytest.raises(VocFormatError, match=r"object 0.*dragon"):
        parse_voc_annotation(xml)
