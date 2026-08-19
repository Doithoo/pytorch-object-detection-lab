# Examples

[Simplified Chinese](README.zh-CN.md) | [Tutorial](../docs/tutorial/README.md)

Run these examples in progression order. They move from tensor-only detection contracts to a real torchvision model contract, one synthetic optimization step, and finally checkpoint-backed prediction. They are teaching probes, not benchmark runs.

## Progression

| Example | Use it to learn | Prerequisites | Network behavior | Expected output or artifact |
|---|---|---|---|---|
| `01_boxes_and_labels.py` | xyxy coordinates, integer labels, and area | Installed PyTorch | Offline | Two boxes, labels, and computed areas on stdout |
| `02_detection_batch.py` | Why detection batches are lists of variable-size images and targets | Package installed | Offline | Two image shapes and target counts `1` and `2` |
| `03_detector_losses.py` | The named scalar loss dictionary returned in training mode | Package installed | Offline | Classification loss, box loss, and their total from a tiny fake detector |
| `03_model_contract.py` | The real torchvision train/eval API boundary | More CPU time and memory than tensor-only examples | Offline because it constructs with `weights: none` | Training loss keys followed by evaluation output keys such as boxes, labels, and scores |
| `04_minimal_training_loop.py` | Zeroing gradients, backpropagation, and one SGD update | Package installed | Offline | A synthetic parameter value before and after one update |
| `05_checkpoint_prediction.py` | Restoring model and classes from a checkpoint | A local self-contained checkpoint and image | Offline; checkpoint restoration forces `weights: none` | `<stem>.json` and `<stem>.png` under the requested output directory |

## Commands

The first five commands need no dataset or input files:

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
uv run python examples/03_detector_losses.py
uv run python examples/03_model_contract.py
uv run python examples/04_minimal_training_loop.py --lr 0.1
```

The last command consumes artifacts from a completed training run and a local image:

```bash
uv run python examples/05_checkpoint_prediction.py --checkpoint artifacts/run/best.pt --image image.jpg --output-dir artifacts/example_prediction
```

Use `03_detector_losses.py` before `03_model_contract.py` when you only need the shape of the API. The former is fast and synthetic. The latter builds a maintained torchvision detector and can be noticeably slower, but it still performs no learning and publishes no score.

## Evidence boundary

Examples 01-04 use synthetic values and prove only local data, model-mode, and optimization contracts. Example 05 proves that a compatible checkpoint can drive local prediction. None downloads VOC, trains on the full dataset, or establishes a full VOC result. Follow the [tutorial](../docs/tutorial/README.md) for the bounded integrated workflow.
