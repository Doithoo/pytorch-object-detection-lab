# Configuration reference

Precedence is dataclass defaults, then YAML, then repeated CLI `--set KEY VALUE`; `--device` is the final runtime override where supported. Unknown fields and invalid types/ranges are rejected.

| Section | Field | Meaning |
|---|---|---|
| `data` | `name` | Dataset provider name |
| `data` | `data_dir`, `manifest_dir` | Source data and prepared manifest roots |
| `data` | `num_workers` | Nonnegative loader workers |
| `data` | `horizontal_flip` | Training flip probability |
| `data` | `max_train_samples`, `max_valid_samples`, `max_test_samples` | Optional positive split limits |
| `model` | `name` | Registry key |
| `model` | `weights` | `none` or `imagenet1k_v1` |
| `model` | `expected_num_classes` | Background plus object classes |
| `model` | `params` | Constructor-specific mapping |
| `train` | `epochs`, `batch_size` | Positive run and batch sizes |
| `train` | `lr`, `momentum`, `weight_decay` | Optimizer values |
| `train` | `optimizer`, `scheduler` | `sgd`/`adamw` and `none`/`step` |
| `train` | `seed`, `amp`, `grad_clip` | Reproducibility, mixed precision, clipping |
| `train` | `best_metric` | Checkpoint selection metric |
| `evaluation` | `score_threshold` | Serialized/rendered prediction threshold |
| `evaluation` | `error_score_threshold`, `error_iou_threshold` | Error-analysis filters and match IoU |
| `evaluation` | `max_detections` | Per-image backend cap |
| top level | `device` | `auto`, `cpu`, `cuda`, or `mps` |
| top level | `output_dir`, `run_name` | Artifact root and optional run directory name |

```bash
detect show-config --config configs/learning_minimal.yaml --set train.epochs 1
```
