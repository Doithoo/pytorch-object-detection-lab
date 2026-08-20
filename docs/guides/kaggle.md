# Complete Training on Kaggle

[简体中文](kaggle.zh-CN.md) | [Completed training run](../recorded-run/README.md)

This is the recommended way to train the project. Kaggle provides a ready-to-use
GPU, so a beginner does not have to solve local CUDA, driver, and memory issues
first. The supplied script downloads VOC 2007, prepares the data, checks one
batch, trains for 26 epochs, selects the best checkpoint, and evaluates the test
set.

The completed T4 run spent about 50 minutes training and about 54 minutes on the
entire Kaggle job, including downloads and evaluation.

## Before you start

You need:

- A Kaggle account. If the GPU option is unavailable, complete any account or
  phone verification requested by Kaggle.
- Network access to Kaggle and the Oxford VOC download host.
- A local copy of this project to submit the prepared runner.

The runner contains a source snapshot and downloads the two official VOC
archives directly. Therefore:

- You do not create or attach a Kaggle Dataset.
- You do not need to install or call `kagglehub`.
- Internet must be enabled for the Kaggle job.

## 1. Install and sign in with the Kaggle CLI

Install the CLI as a separate tool rather than a project dependency:

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

The login command opens a browser for authorization. If the web page says you
are signed in but API requests are rejected, refresh the authorization:

```bash
kaggle auth login --force
```

## 2. Move the job to your account

Open [`../recorded-run/kaggle/kernel-metadata.json`](../recorded-run/kaggle/kernel-metadata.json):

```json
{
  "id": "your-username/pytorch-object-detection-lab-voc2007-gpu",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4"
}
```

Replace only the account name in `id`. Keep `code_file` pointing to
`run_kaggle.py` and leave the data-source lists empty.

## 3. Submit the job

From the repository root:

```bash
kaggle kernels push -p docs/recorded-run/kaggle
```

The command returns the Kaggle page URL. You can also query the status:

```bash
kaggle kernels status <your-username>/pytorch-object-detection-lab-voc2007-gpu
```

After the first submission, open Settings on the Kaggle page and confirm:

- The accelerator is a T4 or newer NVIDIA GPU.
- Internet is enabled.
- The job is Running rather than Error.

Kaggle may display `GPU T4 x2`. This is a single-device project and uses only
`cuda:0`; an idle second GPU is expected and does not require a code change or
resubmission.

## 4. Read the log

A healthy log moves through entries like:

```text
{"project": "/kaggle/working/project", ...}
{"phase": "download_voc2007", "status": "started"}
{"phase": "download_voc2007", "status": "completed", ...}
{"phase": "training", "status": "running", "elapsed_seconds": ...}
...
{"phase": "evaluation", "status": "running", "elapsed_seconds": ...}
```

Training and evaluation print a heartbeat every 60 seconds. If heartbeats and
epoch lines continue to appear, the job is working. Training an object detector
on full VOC takes tens of minutes; do not stop the job because no new metric
appeared for a few minutes.

## 5. Confirm completion

After the status becomes `COMPLETE` or the page says `Successfully ran`, inspect
the final log summary:

- `completed_epochs` is `26`.
- train / valid / test counts are `2501 / 2510 / 4952`.
- `best_epoch` is present.
- test evaluation used all 4,952 images.

The completed reference run reported `3223.9s` total. Your time can vary with
Kaggle hardware and network conditions.

## 6. Download the training artifacts

The complete output also contains about 1.7 GB of temporary VOC data. Usually
you only need `artifacts`:

```bash
kaggle kernels output <your-username>/pytorch-object-detection-lab-voc2007-gpu --file-pattern 'artifacts/.*' -p kaggle-output
```

Start with:

| File | What to inspect first |
|---|---|
| `reference-fasterrcnn/metrics.csv` | Per-epoch loss, validation mAP, and best epoch |
| `reference-fasterrcnn/config.yaml` | CUDA, AMP, and paths actually used on Kaggle |
| `reference-fasterrcnn/best.pt` | Validation-selected model for prediction |
| `reference-fasterrcnn/last.pt` | Final epoch and resume state |
| `reference-fasterrcnn/evaluation/evaluation.json` | Test summary metrics |
| `reference-fasterrcnn/evaluation/per_class.csv` | Results for all 20 classes |
| `reference-fasterrcnn/evaluation/visualizations/` | Predictions, false positives, and misses |
| `kaggle-run-summary.json` | Runtime, split counts, and final metrics |

## Three failures seen during the real setup

### Project archive not found

An early runner expected an external source archive that was not attached to
the non-interactive job:

```text
FileNotFoundError: expected one project archive, found []
```

The current runner embeds its exact source and needs no manual archive
upload. Confirm that metadata points to the current repository
`run_kaggle.py`, not an older copy.

### A new Dataset cannot be attached non-interactively

An earlier runner called `kagglehub.dataset_download` while running and Kaggle
returned:

```text
New Datasets cannot be attached in non-interactive sessions
```

The current runner uses neither a Kaggle Dataset nor `kagglehub`; it downloads
VOC from the official host. Leave `dataset_sources` empty.

### P100 is incompatible with the current PyTorch build

Tesla P100 has compute capability `sm_60`, while the current Kaggle PyTorch
build contains kernels for `sm_70` and newer. It fails with:

```text
CUDA error: no kernel image is available for execution on the device
```

Choose a T4 or newer GPU. This is not caused by the data or model code, and
changing training parameters cannot fix it.

## Continue learning

Read the [training tutorial](../tutorial/04-training.md) while the job runs.
After completion, use [evaluation and prediction](../tutorial/05-evaluation-and-inference.md)
to inspect metrics and images. Compare with the project's
[completed Kaggle run](../recorded-run/README.md) when useful.
