# Dataset and Manifest Format

[Simplified Chinese](dataset-format.zh-CN.md) | [VOC 2007 protocol](voc2007.md)

This reference defines the runtime dataset format in version 0.1. It is for data authors, extension maintainers, and anyone checking experiment identity. The provider accepts the 20 VOC classes only; arbitrary-class datasets require code changes.

## Source tree and XML

Preparation reads `data_dir/VOCdevkit/VOC2007`. It maps `ImageSets/Main/train.txt` to `train`, `val.txt` to `valid`, and `test.txt` to `test`. Split lines are nonempty image IDs without extensions; IDs must be unique within and disjoint across splits. Each ID resolves to `JPEGImages/<id>.jpg` and `Annotations/<id>.xml`.

The XML root must provide nonempty `filename`, positive integer `size/width` and `size/height`, and zero or more `object` elements. Each object requires a supported `name` and `bndbox` with finite numeric `xmin`, `ymin`, `xmax`, `ymax`. Optional `difficult` defaults to `0` and otherwise must be `0` or `1`. The filename and decoded image dimensions must match the XML.

Supported object names, in label order, are: `aeroplane`, `bicycle`, `bird`, `boat`, `bottle`, `bus`, `car`, `cat`, `chair`, `cow`, `diningtable`, `dog`, `horse`, `motorbike`, `person`, `pottedplant`, `sheep`, `sofa`, `train`, `tvmonitor`. Labels are 1 through 20; 0 is background.

VOC coordinates are one-based inclusive. Parsing produces zero-based continuous `xyxy` as `(xmin - 1, ymin - 1, xmax, ymax)`, clips all values to `[0,width]` or `[0,height]`, and rejects nonpositive width or height after clipping. The maximum corner is an exclusive continuous boundary.

## CSV manifests

`train.csv`, `valid.csv`, and `test.csv` have exactly these columns in this order:

| Column | Value |
|---|---|
| `image_id` | source split ID |
| `image_path` | POSIX path `JPEGImages/<id>.jpg`, relative to `dataset_root` |
| `annotation_path` | POSIX path `Annotations/<id>.xml`, relative to `dataset_root` |

Rows preserve split-file order. The CSV files reference source data; they do not contain or copy image/XML bytes.

## `dataset.yaml`

| Key | Type and responsibility |
|---|---|
| `name` | string, currently `voc2007` |
| `dataset_root` | string, currently `VOCdevkit/VOC2007` relative to `data_dir` |
| `class_names` | ordered sequence of the 20 foreground names |
| `label_by_name` | mapping of each name to integer 1 through 20 |
| `split_counts` | mapping `train`, `valid`, `test` to row counts |
| `split_hashes` | mapping each split to a SHA-256 digest of source rows and referenced JPEG/XML bytes |
| `manifest_hashes` | mapping each split to a SHA-256 digest of the exact CSV bytes |
| `identity` | combined SHA-256 experiment identity |
| `coordinate_convention` | exact string `zero-based continuous xyxy; xmax/ymax are exclusive pixel boundaries` |
| `schema_version` | integer manifest schema version, currently `2` |

`split_hashes` cover source content and `manifest_hashes` cover the exact published CSV bytes. Runtime loading validates the schema version, VOC class order, label mapping, split counts, CSV hashes, and metadata identity before constructing a dataset. If any manifest file changes, regenerate the manifests instead of editing it by hand.

For each row, a split hash consumes `image_id,image_path,annotation_path\\n`, each relative path string, then the complete bytes of the referenced image and XML, in row order. The combined identity hashes canonical JSON containing `name`, ordered `classes`, `coordinate_convention`, and `split_hashes`. Thus source bytes, paths/order/membership, classes, or coordinate rules change identity. File timestamps and absolute `data_dir` do not.

## `source.yaml` and `summary.txt`

`source.yaml` contains `dataset: Pascal VOC 2007`, the relative `dataset_root`, and an `archives` mapping from both official tar filenames to published MD5 values. This is generated provenance metadata; for an intentional nonstandard fixture it does not prove that the official archives supplied the bytes.

`summary.txt` is plain text: `identity: <sha256>` followed by `train: <count>`, `valid: <count>`, and `test: <count>`. These two files are informational; runtime loading is driven by the CSV files and `dataset.yaml`.

All six files are written to a staged directory and atomically replace the destination as a set. If validation or publication fails, preparation does not expose a partial new manifest.

## Runtime item format

The loader decodes RGB and returns `image: float32 Tensor[3,H,W]` scaled to `[0,1]`. Its target is:

| Field | Dtype | Shape | Meaning |
|---|---|---|---|
| `boxes` | `float32` | `[N,4]` | zero-based continuous `xyxy` |
| `labels` | `int64` | `[N]` | foreground IDs 1 through 20 |
| `image_id` | `int64` | `[1]` | nonnegative 63-bit integer derived from the first eight SHA-256 bytes of the source ID |
| `area` | `float32` | `[N]` | `(xmax-xmin)*(ymax-ymin)` |
| `iscrowd` | `int64` | `[N]` | 1 exactly when VOC `difficult=1` |
| `difficult` | `bool` | `[N]` | original difficult flag |

Training removes difficult objects before augmentation. Validation, test, inspection, and visualization retain them. Empty and difficult-only training samples remain valid with `boxes [0,4]` and object vectors `[0]`. Horizontal flip updates boxes; degenerate boxes are filtered with all object-aligned fields. Collation returns `list[image]` and `list[target]`, not stacked tensors.

Use `uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16` to inspect a limited number of items, then follow [using your data](../guides/using-your-data.md) for preparation and preview commands.
