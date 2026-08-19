# Add an Internal Model

[Simplified Chinese](adding-models.zh-CN.md) | [Model contract tutorial](../tutorial/03-faster-rcnn.md)

This maintainer guide is for adding a detector that teaches a distinct family or controlled tradeoff. Version 0.1 has no stable external plugin API and does not load arbitrary `module:function` factories. Adding a model means changing and testing the repository; checkpoints never serialize executable user code.

## Implement the constructor contract

Put torchvision-specific construction in `src/object_detector/models/torchvision_models.py`:

```python
def build_detector(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> torch.nn.Module: ...
```

`num_classes` includes background. `weights="none"` must pass `weights=None` and `weights_backbone=None` and must not contact the network. Every named policy must map to one pinned torchvision enum and expose a URL from which preflight can derive the expected cache filename. Pass `params` as constructor keywords only after project-owned arguments are fixed.

The returned module must follow torchvision detection modes:

| Mode | Input | Output |
|---|---|---|
| `train()` | `list[Tensor[3,H,W]]`, `list[target]` | nonempty mapping of finite scalar loss tensors |
| `eval()` | image list only | one mapping per image with `boxes`, `labels`, `scores` |

Do not adapt this contract in the CLI or trainer. The checkpoint-only prediction path rebuilds the registered architecture with `weights="none"`, applies saved state, and depends on stable name and parameter semantics.

## Register metadata

Add one `ModelSpec` in `src/object_detector/models/registry.py` with a stable lowercase name, constructor, `two_stage` or `one_stage` family, factual description, maintained parameter guidance, input notes, supported policies, and policy-to-backbone-weight mapping. Do not create a new name for every parameter combination.

`model.params` cannot contain `weights`, `weights_backbone`, or `num_classes`. The current registry passes other keys to torchvision; document only the keys the project intends to maintain, their types, defaults, and effects. A typo must fail during model construction instead of silently falling back.

## Prove the extension

Add tests for registry ordering and metadata, close-name errors, offline `none` construction, pinned weight mapping, invalid/reserved parameters, one synthetic training forward and update, evaluation outputs, checkpoint save/restore with no download, and CLI discovery without construction.

```bash
uv run pytest tests/test_models.py tests/test_model_smoke.py tests/test_checkpoint.py tests/test_inference.py -q
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

Add a YAML recipe only when it represents a coherent comparison, then run `show-config` and a CPU dry run. Do not publish a performance claim from successful construction, finite losses, or a bounded run. Update the bilingual [model zoo](../reference/model-zoo.md), guide links, packaging declarations when needed, and the ADR if the extension changes the external-code boundary.
