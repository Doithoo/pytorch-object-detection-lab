# Experiments

Start from `learning_minimal.yaml`, set a unique `run_name`, and change one hypothesis at a time with `--set KEY VALUE`. Preserve resolved config, run metadata, manifest identity, metrics, and checkpoint together. Use validation AP for selection and evaluate the reserved test split only after the choice is fixed.

```bash
detect train --config configs/learning_minimal.yaml --set run_name experiment-01 --set train.epochs 2 --device cpu
```
