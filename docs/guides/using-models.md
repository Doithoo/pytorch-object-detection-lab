# Choose and Use a Registered Model

[Simplified Chinese](using-models.zh-CN.md) | [Model zoo](../reference/model-zoo.md)

This guide is for learners choosing among the three maintained torchvision detectors. It covers discovery, weight policy, dry-run evidence, and comparison. It does not claim a speed or accuracy winner. The one [recorded full-VOC run](../recorded-run/README.md) covers only the Faster R-CNN MobileNet recipe and is not a three-model comparison.

## Discover before editing YAML

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
uv run detect model-info fasterrcnn_resnet50_fpn
uv run detect model-info ssdlite320_mobilenet_v3_large
```

`list-models` prints stable names, `two_stage` or `one_stage`, and supported weight policies without constructing a model or touching the network. `model-info` adds the maintained parameter names and input notes. Use Faster R-CNN MobileNet for the tutorial's two-stage path, Faster R-CNN ResNet-50 to change the backbone while retaining the family, or SSDLite to compare a one-stage detector. These are structural choices, not benchmark rankings.

## Choose a weight policy deliberately

`weights: none` passes both detector weights and backbone weights as `None`. Model construction is offline and random; source data must still be local. Use it for examples, dry runs, contract testing, and experiments meant to isolate architecture behavior.

`weights: imagenet1k_v1` still sets full detector weights to `None`, but requests the pinned torchvision ImageNet backbone enum. Training preflight computes the expected torch hub checkpoint path. If that file exists, no network notice is emitted. If absent, preflight prints a notice that network access is required; model construction then lets torchvision download into the torch cache or fail with its underlying network/cache error. Preflight does not download anything itself and the project has no separate download flag.

Inspect the selected path through a dry run:

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name model-check --dry-run --device cpu
```

Expected output lists image shapes, target counts, finite named losses, and `dry-run OK`. It performs one optimizer update but writes no run directory. For an offline pretrained test, place the exact torchvision file in the cache before model construction; do not rename an unrelated file. Cache locations are derived from `torch.hub.get_dir()/checkpoints`, so they can vary by environment.

## Switch one variable

Use a shipped recipe or override only `model.name` while holding the manifest identity, weight policy, sample limits, seed, optimizer, and epochs constant:

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name faster-mobile --device cpu
uv run detect train --config configs/learning_minimal.yaml --set run_name faster-resnet --set model.name fasterrcnn_resnet50_fpn --device cpu
```

Both configurations require `model.expected_num_classes: 21` for VOC's background plus 20 objects. Model-specific values belong under `model.params`; inspect the supported maintained keys in the [model zoo](../reference/model-zoo.md). `weights`, `weights_backbone`, and `num_classes` are reserved and rejected there.

After both runs finish, compare validation evidence from the same manifest:

```bash
uv run detect compare-runs artifacts/faster-mobile artifacts/faster-resnet --metric valid_map_50_95 --output artifacts/model-comparison.csv
```

The command is read-only except for the optional new CSV. It rejects differing manifest identities and reports semantic configuration differences. Use validation to make choices, then evaluate test once after the choice is fixed. See [experiments](experiments.md) for the complete discipline and [adding models](adding-models.md) for internal extension work.
