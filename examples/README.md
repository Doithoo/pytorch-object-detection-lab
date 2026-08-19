# Small Examples

[简体中文](README.zh-CN.md) | [Tutorial](../docs/tutorial/README.md)

Each program isolates one idea and can run on a local CPU. The examples use
synthetic data or an existing checkpoint. They do not train a full VOC model or
produce comparable model scores.

| Example | What it explains | Expected output |
|---|---|---|
| `01_boxes_and_labels.py` | xyxy coordinates, classes, and area | Two boxes, labels, and areas |
| `02_detection_batch.py` | Why differently sized images remain a list | Two image shapes and target counts |
| `03_detector_losses.py` | The loss dictionary returned during training | Synthetic classification loss, box loss, and sum |
| `03_model_contract.py` | Real torchvision training and prediction return values | Loss field and prediction field names |
| `04_minimal_training_loop.py` | How gradients update one parameter | Parameter value before and after the update |
| `05_checkpoint_prediction.py` | How to predict from a checkpoint | JSON and an annotated PNG |

The first five examples do not need VOC:

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
uv run python examples/03_detector_losses.py
uv run python examples/03_model_contract.py
uv run python examples/04_minimal_training_loop.py --lr 0.1
```

`03_model_contract.py` constructs a real torchvision detector and is slower
than the synthetic examples, but it still does not train. `contract` remains in
the historical filename; the program simply shows what the model accepts and
returns in train and eval modes.

The final example needs a downloaded checkpoint and image:

```bash
uv run python examples/05_checkpoint_prediction.py --checkpoint kaggle-output/reference-fasterrcnn/best.pt --image image.jpg --output-dir artifacts/example-prediction
```

For real training, continue to the [Kaggle guide](../docs/guides/kaggle.md).
