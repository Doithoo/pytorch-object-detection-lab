# Add an Internal Model

[Simplified Chinese](adding-models.zh-CN.md) | [Model behavior tutorial](../tutorial/03-faster-rcnn.md)

This maintainer guide covers registered built-in detectors and explicit external
`module:function` factories. Checkpoints never serialize executable user code;
external checkpoints record the factory path and require that same importable
factory at restore time.

## Implement the constructor

For a built-in detector, put torchvision-specific construction in `src/object_detector/models/torchvision_models.py`:

```python
def build_detector(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> torch.nn.Module: ...
```

For an external detector, set `model.factory` to an importable
`module:function`. The factory receives `num_classes`, `weights`, and every key
under `model.params`, and must return a `torch.nn.Module` using the same
list-of-images / target-list contract described above. The project validates
loss mappings in train mode and `boxes`/`labels`/`scores` mappings in eval mode.

```yaml
model:
  name: my_detector
  factory: my_package.models:build_detector
  weights: none
  expected_num_classes: 4
  params:
    width: 32
```

External factories are deliberately explicit. A checkpoint can be safely read
without executing factory code, but prediction, evaluation, or resume will
import the recorded path before reconstructing the model. A missing or changed
factory must fail clearly.

`num_classes` includes background. `weights="none"` must pass `weights=None` and `weights_backbone=None` and must not contact the network. Every named policy must map to one pinned torchvision enum and expose a URL from which preflight can derive the expected cache filename. Pass `params` as constructor keywords only after project-owned arguments are fixed.

The returned module must follow torchvision detection modes:

| Mode | Input | Output |
|---|---|---|
| `train()` | `list[Tensor[3,H,W]]`, `list[target]` | nonempty mapping of finite scalar loss tensors |
| `eval()` | image list only | one mapping per image with `boxes`, `labels`, `scores` |

Do not reinterpret this interface in the CLI or trainer. Checkpoint-only prediction rebuilds the registered architecture with `weights="none"`, applies saved state, and depends on stable names and parameter meanings.

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

Add a YAML configuration only when it represents a coherent comparison, then run `show-config` and a CPU dry run. Do not publish a performance claim from successful construction, finite losses, or a small run. Update the bilingual [model reference](../reference/model-zoo.md), guide links, packaging declarations when needed, and the ADR if the extension changes how external code interacts with the project.
