# Tutorial 02: Trust VOC Data Before Training

[Simplified Chinese](02-data-and-boxes.zh-CN.md) | [Tutorial index](README.md)

This chapter requires the locked environment from Tutorial 01 and enough local
storage for Pascal VOC 2007. Downloading needs access to the official host;
preparation, inspection, and preview are local operations.

## Download the two official archives

```bash
uv run python scripts/download_data.py --data-dir data/raw
```

The script uses these published archive identities:

| Archive | Published MD5 |
|---|---|
| `VOCtrainval_06-Nov-2007.tar` | `c52e279531787c972589f7e41ab4ae64` |
| `VOCtest_06-Nov-2007.tar` | `b6e924de25625d8de591ea690078ad9f` |

Expected success prints both archive paths under `data/raw/archives` and leaves
the extracted tree at `data/raw/VOCdevkit/VOC2007`. An existing archive is reused
only when its checksum matches; a mismatched existing archive triggers a fresh
download. If the completed transfer still has the wrong checksum, the command
fails and removes its `.part` file. An unsafe tar member is also a hard failure.

## Validate and publish fixed manifests

```bash
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
```

Preparation validates the official split counts (`2501` train, `2510` valid,
`4952` test), disjoint image IDs, image/XML presence, decodability, dimensions,
class names, boxes, and annotation filenames. Expected stdout has an `identity=`
SHA-256 followed by the three counts.

The command publishes `train.csv`, `valid.csv`, `test.csv`, `dataset.yaml`,
`source.yaml`, and `summary.txt` together. Split hashes include row identities and
the bytes of referenced images and annotations. The combined identity also
includes ordered classes and the coordinate convention.

Treat this prepared directory as immutable experiment input. It is not made
read-only by the filesystem: running preparation again atomically replaces the
directory. If source content or split membership changes, the identity changes,
and old checkpoints must not be evaluated as if they used the new data.
`--allow-nonstandard-counts` exists for deliberate VOC-shaped fixtures, but a
run prepared that way is not an official VOC 2007 result.

## Convert VOC coordinates exactly once

VOC XML stores one-based inclusive corners. The parser converts
`(xmin, ymin, xmax, ymax)` to zero-based continuous `xyxy` as:

```text
(xmin - 1, ymin - 1, xmax, ymax)
```

For VOC box `(11, 21, 50, 70)`, the project target is `[10, 20, 50, 70]`.
Its width is `40`, height is `50`, and area is `2000`. The maximum values remain
unchanged because they become exclusive continuous boundaries. The parser clips
to image bounds and rejects a non-positive box after clipping.

Do not subtract one again in a transform, and do not use an inclusive `+1` area
formula after conversion. The tensor contract is the one established in
[Tutorial 00](00-basics.md).

## Difficult objects are preserved evidence

VOC marks some objects `difficult=1`. Their treatment depends on purpose:

- Training removes difficult objects before transforms and loss computation.
- Validation and test keep them with `difficult=True` and `iscrowd=1`.
- Metrics do not count them as ordinary targets; detections matching only a
  difficult target are ignored by error analysis.
- An image can therefore have objects in XML but an empty training target. Its
  boxes must still be shaped `[0, 4]`.

This avoids teaching the model from ambiguous targets while preserving enough
information for honest evaluation and visual inspection.

## Inspect structure before pixels

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
```

Expected YAML identifies the dataset, manifest identity, total and inspected
images, ordinary and difficult object counts, empty images, class counts, image
size ranges, and box width/height/area ranges. `--limit 16` bounds decoded
inspection; it does not claim to summarize every object's distribution. Repeat
on `valid` when checking difficult annotations.

A missing source image, malformed XML, mismatched dimensions, empty split, or
non-positive limit fails with a concise error. Do not continue to training just
because the manifest CSV itself can be opened.

## Inspect pixels and boxes together

```bash
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split valid --limit 4 --output artifacts/dataset_preview.png
```

Expected stdout is `artifacts/dataset_preview.png`. Open the image and check that
the annotations present in the selected rows surround the right objects and have
plausible class names. Ordinary boxes are solid green. Dashed orange difficult
boxes appear only when a selected row contains difficult annotations; the script
does not search ahead to guarantee one. A parser can be internally consistent
and still encode a mistaken custom coordinate convention, so this visual check
is part of the trust boundary.

The same conventions are illustrated without VOC data in this deterministic
synthetic teaching diagram:

![Synthetic detection target anatomy](../assets/detection-target-anatomy.png)

This image is documentation evidence for rendering and target anatomy, not a
model prediction or benchmark result. Unlike the selected-row dataset preview, it
is explicitly synthetic and guarantees one labeled dashed difficult target for
the exercise.

## Common failure boundaries

- Download fails before checksums: network or official-host access is missing.
- An existing archive checksum differs: the script downloads it again. If the
  completed transfer still differs, the command fails and removes `.part`; do
  not prepare from the unverified archive.
- Preparation reports nonstandard counts or split overlap: the source tree does
  not satisfy the official protocol.
- Inspection identity differs from a checkpoint: data provenance changed.
- Preview boxes are shifted by one pixel or have implausible size: revisit the
  one-based inclusive to continuous xyxy conversion.
- Training targets unexpectedly vanish: inspect whether all objects are marked
  difficult before assuming a collate bug.

Next, follow these image and target lists through the maintained torchvision
detector in [Tutorial 03](03-faster-rcnn.md).
