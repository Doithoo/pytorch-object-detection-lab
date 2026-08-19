# Tutorial 04: One Update, Then a Deliberate Run

[Simplified Chinese](04-training.zh-CN.md) | [Tutorial index](README.md)

You need prepared `data/manifests`, matching source images under `data/raw`, and
the model/data contracts from Tutorials 00-03. Begin on CPU unless a CUDA or MPS
dry run has already passed.

## Read one optimizer step in isolation

```bash
uv run python examples/04_minimal_training_loop.py --lr 0.1
```

Expected output is `scale: 1.0000 -> 0.7500`. The fake detector makes the update
visible, but it has only one parameter and two synthetic losses. Its literal
order is `forward -> zero_grad -> sum -> backward -> step`. Clearing gradients
must happen before `backward`; doing so after this simple forward is valid because
the forward pass has not accumulated parameter gradients.

The production `dry_run` clears gradients before its forward pass, then sums the
losses, runs backward, and steps the optimizer. It also moves each image and
target dictionary to the selected device, checks every returned loss is a finite
scalar, and optionally clips gradients. Full epochs average losses by image count.

## Dry run: prove one integrated update

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

The command loads one real training batch, runs the configured torchvision model
in train mode, sums its losses, backpropagates, and updates parameters once.
Expected stdout includes:

```text
image_shapes=((3, H1, W1), (3, H2, W2))
target_counts=(N1, N2)
loss_total=<finite value>
loss_classifier=<finite value>
loss_box_reg=<finite value>
loss_objectness=<finite value>
loss_rpn_box_reg=<finite value>
dry-run OK
```

The actual shapes, counts, and values depend on the selected manifest rows and
model state. `dry-run OK` proves that loading, collation, model forward,
backward, and optimizer update connect for one batch. It writes no normal run
directory and does not measure validation quality.

## Bounded learning run: prove the artifact path

`configs/learning_minimal.yaml` limits train/valid/test to 32/16/16 samples, uses
two epochs, random weights, and zero loader workers. Give the run an explicit
name:

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --device cpu
```

On success stdout prints `artifacts/first-detector`. Inspect its files in this
order:

1. `config.yaml`: the complete resolved recipe, including sample limits.
2. `run.yaml`: environment, device, seed, manifest identity, split hashes, and
   ordered class names.
3. `metrics.csv`: one row per completed epoch, with `loss_total`, the four named
   detector losses, and `valid_` AP/AR/count columns.
4. `best.pt`: the epoch that strictly improved validation `map_50_95`.
5. `last.pt`: the most recently completed epoch, including optimizer, scheduler,
   history, and RNG state for resume.

For a fresh run, any existing resolved run directory is rejected, even when it
is empty. Checkpoints and text artifacts are published atomically so a finished
file does not contain a partially written replacement.

This bounded run proves the integrated learning and artifact workflow. Its
metrics describe only the bounded samples, epochs, seed, and configuration. They
are not a complete VOC benchmark.

## Resume without changing experiment semantics

To extend the same bounded run from two to three epochs:

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --set train.epochs 3 --resume artifacts/first-detector/last.pt --device cpu
```

The command above resumes the existing run in place from its current `last.pt`.
You may instead select a new, empty run directory; cross-directory resume carries
forward the compatible sibling `best.pt`. When an in-place `last.pt` already
exists, resuming from `best.pt` or an older copy is rejected to avoid overwriting
newer history. If `last.pt` is missing, the matching `best.pt` may recover that
same directory; no renamed or copied checkpoint is accepted in place. Resume
also requires finite values for the configured validation metric and a
`best_metric` equal to their full-history maximum, plus the same model, classes,
preprocessing contract, manifest identity, and semantic configuration.
Operational fields such as total epochs, workers, and device may change, but the
requested epoch count must exceed the saved epoch.

## Full training is a separate evidence level

`configs/reference_fasterrcnn.yaml` removes sample limits, requests 26 epochs,
uses the official prepared splits, and selects `imagenet1k_v1` backbone weights.
Before considering it, verify the complete dataset identity, device capacity,
weight cache/network policy, output storage, and
[recorded-run evidence gate](../recorded-run/README.md).

The corresponding command is:

```bash
uv run detect train --config configs/reference_fasterrcnn.yaml
```

This is not a quick tutorial command, and listing it is not evidence that it has
been executed. A separate [recorded Kaggle run](../recorded-run/README.md)
preserves one real 26-epoch execution, including its resolved CUDA/AMP config,
runtime, validation selection, test result, checkpoint hash, and images.

## Common failure boundaries

- Preflight reports missing manifests, class-count mismatch, unavailable device,
  or unwritable output: fix that boundary before model construction.
- A pretrained-weight notice appears: the selected model may need network access
  because the expected cache file is absent.
- A loss is NaN or infinite: the trainer reports the loss name and image IDs;
  inspect those samples before changing optimization settings.
- `best.pt` differs from `last.pt`: this is normal when the final validation AP
  did not strictly improve.
- A resume config changes batch size, optimizer, learning rate, augmentation, or
  sample limits: the trainer rejects changed semantics; start a new run.
- A two-epoch bounded metric looks high or low: it is still bounded evidence and
  cannot be promoted to a full VOC claim.

Continue to [Tutorial 05](05-evaluation-and-inference.md) to evaluate a chosen
checkpoint, inspect error evidence, and run checkpoint-only prediction.
