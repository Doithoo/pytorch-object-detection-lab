# Documentation

[Simplified Chinese](README.zh-CN.md)

Start with the [tutorial](tutorial/README.md) if this is your first pass. It follows the exact workflow `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict` and tells you what each stage can and cannot prove. Python 3.10-3.12, uv, and a cloned repository are the basic prerequisites. The data stages also require a local Pascal VOC 2007 tree.

## Choose a route

| Goal | Start here | Use it when | Expected result |
|---|---|---|---|
| Learn the complete workflow | [Tutorial index](tutorial/README.md) and [learning path](tutorial/learning-path.md) | You want commands in dependency order | Prepared manifests, a dataset preview, dry-run diagnostics, bounded run artifacts, evaluation reports, and prediction files |
| Choose and inspect a model | [Using models](guides/using-models.md) and [model zoo](reference/model-zoo.md) | You need weight policies, model metadata, or a comparison starting point | Registry metadata from `detect list-models` or `detect model-info fasterrcnn_mobilenet_v3_large_320_fpn`; no model is trained by these commands |
| Configure a run | [Configuration flow](concepts/configuration-flow.md), [configuration reference](reference/config-reference.md), and [config index](../configs/README.md) | You need precedence, validation, or a shipped recipe | Resolved YAML from `detect show-config`, including the source of each value |
| Prepare or replace data | [Using your data](guides/using-your-data.md), [dataset format](reference/dataset-format.md), and [VOC 2007 reference](reference/voc2007.md) | You need split, coordinate, difficult-object, or manifest rules | Validated CSV manifests and `dataset.yaml`; preparation alone does not prove model quality |
| Understand detector behavior | [Detection flow](concepts/detection-flow.md) and [How Faster R-CNN works](concepts/how-faster-rcnn-works.md) | A batch, loss dictionary, prediction, or metric is surprising | A traceable contract from source annotations to checkpoint-backed output |
| Run and compare experiments | [Experiments](guides/experiments.md), [metrics](reference/metrics.md), and [checkpoint schema](reference/checkpoint-schema.md) | You are testing one hypothesis at a time or comparing compatible runs | Preserved run provenance, validation metrics, checkpoints, and an optional comparison CSV |
| Train on Kaggle | [Kaggle guide](guides/kaggle.md) and [recorded run](recorded-run/README.md) | The full reference recipe is too slow on local CPU | A T4-backed run plus downloadable training and evaluation artifacts |
| Diagnose a failure | [Troubleshooting](guides/troubleshooting.md) and [code tour](concepts/code-tour.md) | A command fails or an artifact looks wrong | The smallest focused command and the owning module or test layer |
| Extend the project | [Adding datasets](guides/adding-datasets.md) or [adding models](guides/adding-models.md) | You are changing a provider or registry contract | Focused offline tests plus matching English and Chinese documentation |
| Review reproducibility decisions | [Architecture decision 0001](architecture/0001-reproducible-voc-detection-contracts.md) | You need the reason behind data, weight, checkpoint, and evidence boundaries | An architectural rationale, not a benchmark result |

## Evidence boundaries

Examples and most tests use synthetic tensors, temporary VOC-shaped fixtures, or fake detectors. They prove API, geometry, serialization, and orchestration contracts. The learning recipe is a bounded run with sample limits; it can prove that learning machinery is connected, but its metrics are not a full-dataset benchmark.

The repository publishes one evidence-complete reference run: 26 epochs on the official splits, validation-selected epoch 18, and test `map_50_95 = 0.322312` on all 4,952 test images. The [recorded run](recorded-run/README.md) preserves its scope, environment, metrics, checkpoint hash, and failure images. Synthetic examples and bounded learning runs remain teaching evidence and must not be presented as that full result.

## Supporting indexes

- [Examples](../examples/README.md): progressive executable contracts, from boxes to checkpoint prediction.
- [Configurations](../configs/README.md): every shipped recipe, its network policy, and its artifact scope.
- [Scripts](../scripts/README.md): download, visualization, plotting, and documentation-asset tools.
- [Tests](../tests/README.md): focused test layers and the complete offline suite.
