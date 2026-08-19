# Kaggle-First Documentation Design

## Goal

Make Kaggle the primary training environment for beginners and make the
documentation feel calm, direct, and easy to finish. Keep local training as an
optional path for readers who already have suitable hardware.

## Audience

The primary reader knows basic Python but may be new to PyTorch, object
detection, GPUs, and Kaggle. Contributor and implementation details remain
available, but they must not interrupt the beginner path.

## Information Architecture

The beginner path is:

1. Understand the task and expected outcome.
2. Open and configure Kaggle.
3. Learn the VOC data and bounding-box format.
4. Learn the detector interface and Faster R-CNN at a practical level.
5. Train the reference configuration on Kaggle.
6. Read the metrics, visualizations, and prediction examples.

The root README, documentation index, tutorial index, and learning path all
lead to this sequence. The Kaggle guide becomes a complete start-to-finish
training guide rather than an optional platform note.

Local instructions are limited to environment checks, small dry runs, and a
clearly marked advanced path for readers with a compatible local GPU. The
local commands and project capabilities remain supported.

## Training Results

All published training claims refer to the successful Kaggle T4 run:

- 26 completed epochs
- best validation epoch: 18
- VOC 2007 test images: 4,952
- test `map_50_95`: `0.322312`
- test `map_50`: `0.609917`
- training time: `3025.660` seconds
- complete Kaggle notebook time: `3223.9` seconds

The recorded-run files remain the source of the detailed metrics, class
results, failure examples, configuration, and checkpoint hash. Dry runs,
synthetic examples, and unit-test fixtures are described as teaching or
verification material, never as training results.

## Writing Style

Beginner pages use plain, concrete language and short sections. Each step says
what to do, what the reader should see, and where to go next. Detailed
implementation safeguards, hashes, serialization rules, and contributor
decisions move to reference or architecture pages.

The project name remains unchanged where it identifies the repository or
package, but prose uses terms such as "project", "tutorial", or "example"
instead of repeatedly calling the project a "lab". "Recipe" becomes
"configuration". "Contract" becomes "input and output rules", "interface
behavior", or a direct description unless it is a precise maintainer term.
"Evidence boundary" becomes a direct explanation of what a result does and
does not show.

Warnings are reserved for actions that can waste significant time or produce
incorrect results. Troubleshooting text is factual and task-oriented rather
than defensive.

## Scope

Update both English and Simplified Chinese beginner-facing documentation:

- root README files and documentation indexes
- tutorial index, learning path, and tutorial chapters
- Kaggle, experiment, troubleshooting, model, data, and configuration guides
- configuration and example indexes where they direct training choices
- recorded Kaggle run summaries and relevant reference-page links

Maintainer ADRs and schema references may retain precise technical terms when
changing them would reduce accuracy. Source code, CLI behavior, configuration
schemas, model behavior, and recorded result files do not change.

## Verification

Verification will check:

- English and Chinese navigation agree on the Kaggle-first path.
- Beginner pages do not present local full training as the default.
- Only the recorded Kaggle run is presented as a completed training result.
- Repetitive terms such as "lab", "recipe", "contract", and their Chinese
  equivalents are removed or retained only where technically necessary.
- Documentation links and existing documentation tests pass.

