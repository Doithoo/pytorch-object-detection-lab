# Configuration Reference

[Simplified Chinese](config-reference.zh-CN.md) | [Configuration flow](../concepts/configuration-flow.md)

This reference is for authors of YAML recipes and `--set` overrides. `AppConfig` is strict: unknown sections and fields fail, mappings cannot replace scalar fields, and validated values do not silently fall back.

## Resolution and source tracking

Resolution order is typed dataclass defaults, then the optional YAML file, then repeated `--set KEY VALUE` pairs in command order. Each override value is parsed by `yaml.safe_load`, not kept as raw text. For example, `true` becomes a Boolean, `3` an integer, `0.5` a float, `null` Python `None`, and `[320, 640]` a list. The token `none` remains the string required by `model.weights`; `null` and `~` are null values.

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set train.epochs 3 --set train.amp true --set run_name experiment-a
```

`show-config` prints the complete serialized configuration plus a sorted `sources` mapping. Each leaf is labeled `default`, `yaml`, or `cli`. It performs no data loading, model construction, network access, or artifact write. A `train --device ...` override is applied after `AppConfig` loading and is therefore not represented by `show-config` source tracking.

## Data fields

| Field | Default | Accepted value | Important interaction |
|---|---:|---|---|
| `data.name` | `voc2007` | string, exactly `voc2007` | No other provider is implemented. |
| `data.data_dir` | `data/raw` | path-compatible string | Runtime appends `dataset.yaml.dataset_root`; evaluation uses the path saved in the checkpoint config. |
| `data.manifest_dir` | `data/manifests` | path-compatible string | Must contain `train.csv`, `valid.csv`, `test.csv`, `dataset.yaml`; identity must match the checkpoint for evaluation/resume. |
| `data.num_workers` | `0` | integer >= 0, Boolean rejected | Operational resume field; used by train and evaluation DataLoaders. |
| `data.horizontal_flip` | `0.5` | finite number in [0, 1] | Training only; transforms image and boxes together. |
| `data.max_train_samples` | `null` | `null` or integer >= 1 | Limits the leading train rows in both training and evaluation when the train split is selected; changes training semantics and prevents a full-split claim. |
| `data.max_valid_samples` | `null` | `null` or integer >= 1 | Limits validation during training/evaluation. |
| `data.max_test_samples` | `null` | `null` or integer >= 1 | Limits test evaluation; a bounded score is not full VOC evidence. |

## Model fields

| Field | Default | Accepted value | Important interaction |
|---|---:|---|---|
| `model.name` | `fasterrcnn_mobilenet_v3_large_320_fpn` | nonempty string; registry membership checked at construction | Discover names with `detect list-models`. Resume identity requires the same name. |
| `model.weights` | `none` | `none` or `imagenet1k_v1` | `none` is offline. `imagenet1k_v1` requests only the pinned backbone weight and may need cache/network access. Checkpoint restore always reconstructs with `none`. |
| `model.expected_num_classes` | `21` | integer >= 2, Boolean rejected | Preflight requires background plus manifest classes, which is 21 for VOC. The model uses the metadata-derived count. |
| `model.params` | `{}` | mapping | Passed as torchvision constructor keywords. `weights`, `weights_backbone`, and `num_classes` are reserved. See the [model zoo](model-zoo.md). |

## Training fields

| Field | Default | Accepted value | Important interaction |
|---|---:|---|---|
| `train.epochs` | `2` | integer >= 1 | Resume may change it only upward beyond the saved epoch. |
| `train.batch_size` | `2` | integer >= 1 | Used for training and checkpoint evaluation; a semantic resume field. |
| `train.lr` | `0.005` | finite number > 0 | Applied to SGD or AdamW. |
| `train.momentum` | `0.9` | finite number >= 0 | Used only by SGD; still part of resume identity under AdamW. |
| `train.weight_decay` | `0.0005` | finite number >= 0 | Applied to both supported optimizers. |
| `train.optimizer` | `sgd` | `sgd` or `adamw` | SGD uses momentum; AdamW does not. |
| `train.scheduler` | `none` | `none` or `step` | `step` is fixed `StepLR(step_size=8, gamma=0.1)` and its state is checkpointed. |
| `train.seed` | `42` | integer from `0` through `4294967295` inclusive, Boolean rejected | Seeds Python, NumPy, torch, and the shuffle generator; out-of-range values fail with `train.seed must be between 0 and 4294967295`; RNG state is checkpointed. |
| `train.amp` | `false` | YAML Boolean only | Gradient scaling only on CUDA and autocast only on CPU/CUDA, so a resolved MPS device uses full precision. Preflight emits the MPS notice only when configured `device` is exactly `mps`; `auto` resolving to MPS is silent. |
| `train.grad_clip` | `0.0` | finite number >= 0 | `0` disables clipping; positive values call global norm clipping after backward. |
| `train.best_metric` | `map_50_95` | exactly `map_50_95` | `best.pt` updates only on a strict validation improvement. |

## Evaluation and top-level fields

| Field | Default | Accepted value | Important interaction |
|---|---:|---|---|
| `evaluation.score_threshold` | `0.05` | finite number in [0, 1] | Stored in config but the `evaluate` CLI has its own runtime `--score-threshold`; AP always receives raw predictions. |
| `evaluation.error_score_threshold` | `0.5` | finite number in [0, 1] | Checkpoint evaluation uses it for image-level error analysis. |
| `evaluation.error_iou_threshold` | `0.5` | finite number in [0, 1] | Same-class greedy matching threshold for error kinds. |
| `evaluation.max_detections` | `100` | integer exactly `100` | Matches the supported COCO-style AR cap; other positive values are currently rejected. |
| `device` | `auto` | nonempty string in config; training preflight supports `auto`, `cpu`, `mps`, or strings starting with `cuda` | `auto` resolves CUDA, then MPS, then CPU. Evaluation/prediction device is a runtime argument. |
| `output_dir` | `artifacts` | path-compatible string | Preflight requires a writable existing ancestor. Run directory is `output_dir/run_name`, with `run` used when name is null. |
| `run_name` | `null` | `null` or nonempty string | Fresh runs reject an existing directory. It is an operational resume field. |

Every table field can come from default, YAML, or `--set`; the source does not change validation. Runtime-only CLI arguments such as `--dry-run`, `--resume`, `--overwrite`, `--split`, and prediction inputs are not `AppConfig` leaves. See [configuration flow](../concepts/configuration-flow.md) for command ownership and [checkpoint schema](checkpoint-schema.md) for resume rules.
