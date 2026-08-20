# Add an Internal Dataset Provider

[Simplified Chinese](adding-datasets.zh-CN.md) | [Dataset format](../reference/dataset-format.md)

This maintainer guide covers adding a new prepared-data provider when VOC-shaped or COCO JSON input is not enough. Existing provider selection is available through `data.name`; a new provider must preserve the shared manifest and target contracts.

## Keep preparation and loading separate

Keep preparation separate from runtime loading:

```text
source files -> full validation -> staged manifests + metadata -> atomic publication
prepared rows -> Dataset -> (float RGB image, target) -> list-based collate
```

A provider must publish fixed train, valid, and test membership, reject duplicates and cross-split overlap, and derive identity from source content, ordered classes, coordinate rules, and split rows. Manifests reference source paths; they do not copy source data. A failed preparation must leave the previous manifest directory intact.

## Implementation ownership

1. Add format parsing and validation under `src/object_detector/data/`. Keep source-coordinate conversion in the parser.
2. Add a preparation function that writes the schemas documented in [dataset format](../reference/dataset-format.md) through staging and atomic replacement.
Add preparation and runtime loading for the new format, then update configuration dispatch, preflight, checkpoint identity, documentation, packaging, and tests together.
4. Update preflight, checkpoint identity, evaluation, examples, and bilingual documentation together.

The canonical item is `image: float32 Tensor[3,H,W]` in RGB `[0,1]` plus:

| Field | Requirement |
|---|---|
| `boxes` | `float32 [N,4]`, zero-based continuous `xyxy` |
| `labels` | `int64 [N]`; 0 is background and object labels start at 1 |
| `image_id` | `int64 [1]`, stable for the source ID |
| `area` | `float32 [N]`, `(xmax-xmin)*(ymax-ymin)` |
| `iscrowd` | `int64 [N]` |
| `difficult` | `bool [N]` |

Object-aligned fields must share `N`. Empty targets keep `[0,4]` and `[0]` shapes. Geometry transforms must update the image, boxes, area, and every aligned field together. `detection_collate` must return lists so variable image sizes and object counts remain valid.

## Validation and tests

Reject missing or corrupt images, malformed annotations, unknown labels, non-finite or degenerate boxes, filename or dimension disagreement, invalid split membership, and metadata/class-count mismatch before training. Decide and document difficult/crowd semantics instead of silently dropping them.

Add focused parser, manifest, dataset, transform, inspection, preflight, and end-to-end tests. Include empty and difficult-only images, stable identity, changed source bytes, atomic failure, collate, and a model dry run. Run:

```bash
uv run pytest tests/test_manifest.py tests/test_dataset.py tests/test_transforms.py tests/test_end_to_end.py -q
```

A provider is not complete merely because one sample loads. Preparation must be reproducible, invalid input must fail without partial output, runtime targets must follow the detector input format, and the data identity must reach run files and checkpoints. For already VOC-shaped sources, use [your own data](using-your-data.md) instead.
