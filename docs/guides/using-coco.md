# Use COCO JSON Data

[简体中文](using-coco.zh-CN.md) | [Dataset format](../reference/dataset-format.md)

The COCO provider accepts one standard COCO annotation JSON for each `train`,
`valid`, and `test` split. Images live below one shared directory, and each
JSON contains `images`, `annotations`, and `categories`.

## Layout

```text
my-coco/
|-- images/
|   |-- train-001.jpg
|   |-- valid-001.jpg
|   `-- test-001.jpg
`-- annotations/
    |-- instances_train.json
    |-- instances_valid.json
    `-- instances_test.json
```

Each image record needs integer `id`, `file_name`, `width`, and `height`. Each
annotation needs `image_id`, `category_id`, and a positive `[x, y, width,
height]` `bbox`. `iscrowd` is optional and defaults to zero. Category IDs may
be sparse; the provider sorts category names and writes continuous labels in
`dataset.yaml`.

## Prepare

```bash
uv run detect prepare-data --format coco \
  --data-dir my-coco \
  --images-dir images \
  --manifest-dir data/my-coco-manifests \
  --train-annotations annotations/instances_train.json \
  --valid-annotations annotations/instances_valid.json \
  --test-annotations annotations/instances_test.json
```

The command validates image dimensions, references, categories, boxes, split
overlap, and source bytes before publishing manifests. Then inspect and verify:

```bash
uv run detect verify-data --data-dir my-coco --manifest-dir data/my-coco-manifests
uv run detect inspect-data --data-dir my-coco --manifest-dir data/my-coco-manifests --split train --limit 16
```

Set `model.expected_num_classes` to the number of foreground categories plus
background. Keep the generated manifest directory together with the data when
resuming or evaluating a checkpoint.
