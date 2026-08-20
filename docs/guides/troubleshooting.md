# Troubleshooting

[简体中文](troubleshooting.zh-CN.md) | [Kaggle guide](kaggle.md)

Start with the earliest error in the log. Later nbconvert warnings usually come
from Kaggle generating the results page and are not the training failure.

## Kaggle fails immediately after submission

### `expected one project archive, found []`

This is an old runner that expects a manually attached source archive. The
current [`run_kaggle.py`](../recorded-run/kaggle/run_kaggle.py) embeds its
source. Confirm that metadata points to this file and submit again.

### `New Datasets cannot be attached in non-interactive sessions`

This is an earlier runner that calls `kagglehub.dataset_download` while running.
The current runner needs neither a Dataset nor `kagglehub`; `dataset_sources`
should be empty.

### `no kernel image is available for execution on the device`

The job received a Tesla P100. The current Kaggle PyTorch build does not support
P100 `sm_60`. Select a T4 or newer GPU in Settings and submit the job again.

### The page shows T4 x2, but only one GPU works

This is expected. The project trains on one GPU and uses only `cuda:0`. Do not
stop a job that continues to print heartbeats.

## Kaggle remains Running

The full run takes about 50-60 minutes. If the log prints
`{"phase": "training", "status": "running"}` every 60 seconds, the job is
healthy. Act only when heartbeats stop and the page reports an error; use the
first traceback.

## Kaggle CLI problems

- `kaggle: command not found`: run `uv tool install kaggle` and ensure the uv
  tool directory is on PATH.
- The API returns unauthorized: run `kaggle auth login --force`.
- Kernel ID is not found: make the username in metadata and the query identical.
- GPU is unavailable: complete Kaggle account verification and check your
  weekly GPU quota.

## VOC download or preparation fails

- Download cannot connect: confirm Kaggle Internet is enabled; the official
  host may also be temporarily unavailable.
- MD5 differs: do not bypass the check; download the archive again.
- Split counts are not `2501 / 2510 / 4952`: confirm complete official VOC 2007.
- An image or XML file is missing: rerun download and preparation rather than
  editing generated CSV files to hide it.
- Custom data has different counts: use `--allow-nonstandard-counts` and state
  clearly that it is not an official VOC result.

## Local environment problems

- `uv sync --locked` fails: use Python 3.10-3.12 and keep the committed `uv.lock`.
- Another `object_detector` is imported: inspect `uv run python -c "import object_detector; print(object_detector.__file__)"`.
- Local CUDA is unavailable: use Kaggle; reserve CPU for examples and dry runs.
- An operation fails on MPS: retry with `--device cpu` and
  `data.num_workers=0` to isolate the device issue.

## Training fails

- Pretrained weight download fails: confirm Internet or place the expected
  weight in the Torch Hub cache.
- A loss is NaN / Inf: inspect the reported image IDs, boxes, labels, and
  learning rate; reproduce in full precision first.
- GPU memory is exhausted: reduce `train.batch_size` or image size and use a new
  run name for changed settings.
- `best.pt` differs from `last.pt`: the final epoch is not always the best
  validation epoch; this is normal.
- Output directory already exists: change `run_name` for a new training run.

## Resume fails

- Prefer `last.pt` from the same run; it contains the newest optimizer,
  scheduler, and random state.
- Model, classes, data identity, batch size, learning rate, or augmentation
  differs: start a new run for those changes.
- Requested epoch total is not greater than completed epochs: set
  `train.epochs` to a larger target.
- Checkpoint version is unsupported or corrupt: do not disable safe loading;
  use a valid file produced by this project.

## Evaluation or prediction fails

- Data identity differs during evaluation: use the prepared data that trained
  the checkpoint.
- Prediction does not need VOC, but the checkpoint must contain a supported
  model name, classes, and preprocessing information.
- Output directory contains files: choose a new directory; use `--overwrite`
  only after deciding the old result is no longer needed.
- JSON has more predictions than PNG: `--display-limit` only limits drawing.

## Metrics look very low

Near-zero values from random weights or a small dry run are normal and are not
training scores. For a complete run, inspect `metrics.csv`, `evaluation.json`,
`per_class.csv`, `errors.csv`, and prediction images in that order. Confirm
completed epochs, best epoch, and data counts before diagnosing the model.
Compare with the [completed Kaggle record](../recorded-run/README.md) when useful.
