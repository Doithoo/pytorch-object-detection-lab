# Train on Kaggle

[Simplified Chinese](kaggle.zh-CN.md) | [Recorded run](../recorded-run/README.md)

Kaggle is useful when the full VOC reference recipe is too slow on a local
CPU. The official CLI handles authentication, submission, status, logs, and
output download. `kagglehub` is not needed for the recorded runner because it
downloads the two official VOC archives directly and embeds its source
snapshot.

## Install and authenticate

Install the CLI as a user tool rather than a project runtime dependency:

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

If an expired OAuth cache reports that you are logged in while API calls reject
authentication, refresh it with `kaggle auth login --force`.

## Use a compatible GPU

Request `NvidiaTeslaT4` or newer. The current Kaggle PyTorch 2.10 CUDA 12.8
build supports compute capability 7.0 and newer; Tesla P100 is `sm_60` and
fails with `no kernel image is available for execution on the device`. The
recorded T4 x2 allocation used only `cuda:0`; the project does not implement
multi-GPU training.

Internet must remain enabled so the runner can download official VOC 2007 and
the pinned ImageNet backbone weight. The run needs no attached dataset.

## Submit and monitor

[`../recorded-run/kaggle/run_kaggle.py`](../recorded-run/kaggle/run_kaggle.py)
is the exact self-contained v7 runner. Its adjacent metadata requests a T4 and
internet. Change its `id` before publishing a copy under another account, then:

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7
```

The runner prints a heartbeat every 60 seconds. Do not edit or resubmit a
running kernel merely because the second allocated T4 is idle.

## Download only results

After status becomes `KernelWorkerStatus.COMPLETE`, avoid downloading the
temporary 1.7 GB VOC tree:

```bash
kaggle kernels output yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7 --file-pattern 'artifacts/.*' -p kaggle-output
```

Verify `completed_epochs`, split counts, the best validation epoch, test image
count, and the SHA-256 of `best.pt` before quoting metrics. Keep large
checkpoints outside Git unless the project deliberately publishes them through
a release or model host.
