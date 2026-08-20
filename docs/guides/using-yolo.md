# Export Prepared Data for YOLO

[简体中文](using-yolo.zh-CN.md) | [Dataset format](../reference/dataset-format.md)

The project can export any prepared VOC-shaped or COCO JSON dataset to the
common YOLO detection layout. This keeps the validated split membership and
class mapping while converting project `xyxy` boxes to normalized YOLO
`class cx cy width height` rows.

```bash
uv run detect export-yolo-data \
  --data-dir data/raw \
  --manifest-dir data/manifests \
  --output-dir artifacts/yolo-data
```

The output contains copied images, zero-based text labels, split image lists,
and `data.yaml`:

```text
artifacts/yolo-data/
|-- data.yaml
|-- train.txt
|-- valid.txt
|-- test.txt
|-- images/{train,valid,test}/
`-- labels/{train,valid,test}/
```

VOC difficult and COCO crowd objects are omitted because the common YOLO text
format has no equivalent flag. The exporter publishes atomically and refuses a
nonempty destination unless `--overwrite` is explicit.

## Training engine boundary

This repository does not silently redirect its trainer to a third-party YOLO
engine. YOLO implementations differ in architecture, augmentation, loss,
checkpoint, result format, and license. For example, current Ultralytics
packages use AGPL terms; other implementations may use different terms. Review
the selected implementation and version before installing or distributing it.

The generated data is intentionally engine-neutral. A compatible engine can
consume `data.yaml`, while this repository remains responsible only for the
validated conversion. Results from a separate engine should record its exact
version, configuration, weights, hardware, and metric definition before being
compared with the repository's torchvision runs.
