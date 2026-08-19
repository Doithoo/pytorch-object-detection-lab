# Metrics and Evaluation Artifacts

[Simplified Chinese](metrics.zh-CN.md) | [Evaluation tutorial](../tutorial/05-evaluation-and-inference.md)

This reference is for reading training history, AP/AR reports, error rows, and evidence images. Metric values are unitless fractions, not percentages. The [recorded full-VOC run](../recorded-run/README.md) is one concrete example; bounded runs and an unexecuted recipe are not equivalent evidence.

## Training `metrics.csv`

Rows are rewritten atomically after each completed epoch. Columns appear in first-seen insertion order:

| Column | Meaning |
|---|---|
| `epoch` | one-based completed epoch |
| `loss_total` | sample-weighted epoch mean of the sum of model-returned losses |
| model loss names | sample-weighted epoch means for every key returned in train mode |
| `valid_map_50_95`, `valid_map_50`, `valid_map_75` | validation AP fields below |
| `valid_mar_1`, `valid_mar_10`, `valid_mar_100` | validation AR fields below |
| `valid_image_count` | validation images processed |
| `valid_target_count` | ordinary validation targets, excluding `iscrowd=1` difficult objects |
| `valid_prediction_count` | all detections returned by the model before project score filtering |

Faster R-CNN contributes exactly `loss_classifier`, `loss_box_reg`, `loss_objectness`, and `loss_rpn_box_reg`. Other registered detector families may return different loss keys; the trainer records the mapping it receives. Validation sends model-returned predictions directly to the metric backend without applying `evaluation.score_threshold`.

## Aggregate AP and AR

The backend is `torchmetrics.detection.MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)`, backed by `pycocotools`.

| Field | Contract |
|---|---|
| `map_50_95` | mean AP over IoU 0.50, 0.55, ..., 0.95 |
| `map_50` | AP at IoU 0.50 |
| `map_75` | AP at IoU 0.75 |
| `mar_1` | mean AR with at most 1 detection per image |
| `mar_10` | mean AR with at most 10 detections per image |
| `mar_100` | mean AR with at most 100 detections per image |
| `per_class` | represented non-background classes with `class_id`, `class_name`, `map_50_95`, `mar_100` |
| `image_count` | images supplied to the backend |
| `target_count` | ordinary, non-crowd targets |
| `prediction_count` | detections supplied to the backend |

Torchmetrics negative sentinel values, including undefined/missing-class outputs, are clamped to `0.0`. No other normalization, rescaling, smoothing, or percentage conversion occurs. Metric payload values, serialized prediction numbers, and numeric error/per-class row values are rounded to six decimals where applicable. The threshold fields in `evaluation.json` are written directly from their configured floats without that rounding helper. In-memory training values and checkpoint history retain ordinary Python float precision.

Difficult VOC objects enter evaluation targets as `iscrowd=1`. They do not add to ordinary `target_count`. The backend applies its crowd handling; the separate error analyzer treats qualifying difficult-only matches as `ignored` and never marks difficult targets as missed.

## `evaluation.json`

The CLI evaluation writes:

| Key | Value |
|---|---|
| `metrics` | aggregate fields above, including nested `per_class` |
| `backend_versions` | installed `torchmetrics` and `pycocotools` version strings |
| `score_threshold` | CLI threshold for serialized predictions and evidence rendering, default 0.05 |
| `error_score_threshold` | checkpoint config threshold for error candidates, default 0.5 |
| `error_iou_threshold` | checkpoint config same-class match threshold, default 0.5 |
| `max_detections` | supported metric cap, exactly 100 |
| `manifest_identity` | prepared dataset identity |
| `checkpoint_sha256` | complete checkpoint file SHA-256 |
| `split` | `train`, `valid`, or `test` |

Changing CLI `--score-threshold` changes `predictions.json` and rendered blue boxes, not AP/AR or `prediction_count`. Model-owned postprocessing such as `box_score_thresh` or SSDLite `score_thresh` happens inside the model and therefore can change what reaches the backend.

## CSV and JSON evidence

`per_class.csv` columns are exactly `class_id,class_name,map_50_95,mar_100`. `predictions.json` is an array of `{image_id, predictions}`; each retained prediction has `box` (four rounded `xyxy` values), `class_id`, `class_name`, and `score`.

`errors.csv` columns are exactly `image_id,kind,class_name,score,iou,box`. `box` is a JSON array inside the CSV cell. A missed row has an empty score. Error candidates are predictions with score at least `error_score_threshold`, ordered by descending score with original order breaking ties, and greedily matched to unmatched ordinary targets of the same class.

| `kind` | Exact condition |
|---|---|
| `ignored` | no ordinary match, but same-class difficult IoU >= error IoU threshold |
| `localization` | no threshold-qualified match, but positive IoU with an unmatched same-class ordinary target |
| `false_positive` | neither ordinary nor difficult same-class overlap qualifies and ordinary best IoU is 0 |
| `missed` | ordinary target remains unmatched after all candidates; `score` is empty and IoU is 0 |

`visualizations/summary.png` always renders the first evaluated sample. Up to five image IDs with the most `missed` rows and up to five with the most `false_positive` rows are reloaded and written as ranked PNGs; ties use image ID. Green boxes are ordinary targets, dashed orange boxes difficult targets, and blue boxes serialized predictions.

All evaluation files are staged and the output directory is published atomically. A nonempty destination fails unless `--overwrite` is explicit. Use validation artifacts for choices and reserve test for the final fixed decision.
