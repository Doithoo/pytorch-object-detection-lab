# Use Your Own VOC-Shaped Data

[Simplified Chinese](using-your-data.zh-CN.md) | [Tutorial: data and boxes](../tutorial/02-data-and-boxes.md)

This guide is for users whose images and annotations can be arranged like Pascal VOC 2007. The current application does not accept arbitrary classes or annotation formats through configuration. Supporting either requires code changes; there is no stable external dataset plugin API.

## Choose the right path

Use the official path when you need the VOC 2007 protocol and counts. Download the two verified archives and run preparation without an exception flag:

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
```

Use the fixture path only when your controlled data deliberately has VOC's directory, XML, class, and coordinate rules but different split counts:

```bash
uv run detect prepare-data --data-dir data/my-voc --manifest-dir data/my-manifests --allow-nonstandard-counts
```

`--allow-nonstandard-counts` disables only the checks for 2501 train, 2510 valid, and 4952 test images. Every other validation still runs. A manifest prepared this way is VOC-shaped evidence, not an official VOC 2007 result.

## Required source tree

```text
data/my-voc/VOCdevkit/VOC2007/
  JPEGImages/<image-id>.jpg
  Annotations/<image-id>.xml
  ImageSets/Main/train.txt
  ImageSets/Main/val.txt
  ImageSets/Main/test.txt
```

Each nonempty split line is an image ID without an extension. IDs must be unique within a split and disjoint across train, valid (`val.txt`), and test. Every ID must resolve to both a decodable JPEG and an XML file. The XML `filename` must equal `<image-id>.jpg`, and XML width and height must match the decoded image.

Each XML object name must be one of the 20 names in the [dataset format reference](../reference/dataset-format.md). Arbitrary-class datasets are not accepted without changing `VOC_CLASSES`, metadata construction, tests, and the class-count contract. `difficult` is optional and defaults to `0`; when present it must be `0` or `1`.

VOC XML boxes are one-based inclusive. Preparation converts `(xmin, ymin, xmax, ymax)` once to `(xmin - 1, ymin - 1, xmax, ymax)`, clips to image bounds, and rejects a nonpositive box. Runtime targets use zero-based continuous `xyxy` with exclusive maximum boundaries.

## Publish and inspect the manifest

Successful preparation prints an `identity=` SHA-256 and split counts, then atomically publishes `train.csv`, `valid.csv`, `test.csv`, `dataset.yaml`, `source.yaml`, and `summary.txt`. These files contain paths, metadata, hashes, and provenance. They do not copy source JPEG or XML bytes; the runtime still reads below `data.data_dir`.

```bash
uv run detect inspect-data --manifest-dir data/my-manifests --data-dir data/my-voc --split train --limit 16
uv run python scripts/preview_dataset.py data/my-manifests --data-dir data/my-voc --split valid --limit 4 --output artifacts/my-data-preview.png
```

Expected evidence is an inspection YAML report plus `artifacts/my-data-preview.png`. The preview script overwrites an existing PNG at `--output` without prompting, so choose a new path when preserving earlier evidence. Verify class names, empty images, difficult counts, image ranges, and box placement before training. Training removes difficult objects; inspection and evaluation retain them as `difficult=True` and `iscrowd=1`.

## Prove the configured runtime path

Point the learning recipe at the same source and manifest directories. First inspect the resolved values and their `cli` sources:

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set data.data_dir data/my-voc --set data.manifest_dir data/my-manifests --set run_name my-data-check
```

Then cross the dataset, model, loss, and optimizer boundary with one dry-run batch:

```bash
uv run detect train --config configs/learning_minimal.yaml --set data.data_dir data/my-voc --set data.manifest_dir data/my-manifests --set run_name my-data-dry-run --dry-run --device cpu
```

Expected output names image shapes, target counts, finite losses, and `dry-run OK`. Images and XML are read below `data/my-voc`; manifest rows and identity come from `data/my-manifests`. Dry run writes no normal run directory.

To exercise the bounded artifact path, use a distinct name and the same overrides:

```bash
uv run detect train --config configs/learning_minimal.yaml --set data.data_dir data/my-voc --set data.manifest_dir data/my-manifests --set run_name my-data-bounded --device cpu
```

On success this writes the standard run set under `artifacts/my-data-bounded`. The learning recipe still limits samples and epochs, and nonstandard-count data is still not official VOC 2007. These commands provide integration evidence only; they do not produce a full benchmark result.

If any source image, XML, or split changes, rerun preparation and use the new identity. Do not edit CSV rows or hashes to preserve an old identity. Continue with the [dry-run and training tutorial](../tutorial/04-training.md), or read [adding datasets](adding-datasets.md) when the source format is genuinely different.
