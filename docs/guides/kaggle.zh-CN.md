# 在 Kaggle 上完成训练

[English](kaggle.md) | [已完成的训练](../recorded-run/README.zh-CN.md)

这是本项目推荐的训练方式。Kaggle 提供可直接使用的 GPU，避免初学者先处理本地 CUDA、
驱动和显存问题。项目提供的脚本会依次下载 VOC 2007、准备数据、检查一个 batch、训练 26
轮、选择最佳 checkpoint，并评估测试集。

已经完成的 T4 运行训练约 50 分钟，包含下载和评估的整个任务约 54 分钟。

## 开始前

你需要：

- 一个 Kaggle 账户；如果 GPU 选项不可用，按 Kaggle 页面要求完成账户或手机验证。
- 能访问 Kaggle 和 Oxford VOC 下载地址的网络。
- 本项目的本地副本，用于提交已经准备好的 runner。

runner 已经包含项目源码快照，并直接下载两份官方 VOC 压缩包。因此：

- 不需要创建或挂载 Kaggle Dataset。
- 不需要安装或调用 `kagglehub`。
- Kaggle 任务必须开启 Internet。

## 1. 安装并登录 Kaggle CLI

推荐把 CLI 安装为独立工具，不加入项目依赖：

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

登录命令会打开浏览器完成授权。如果网页显示已登录但 API 仍拒绝请求，重新运行：

```bash
kaggle auth login --force
```

## 2. 把任务改成你的账户

打开 [`../recorded-run/kaggle/kernel-metadata.json`](../recorded-run/kaggle/kernel-metadata.json)：

```json
{
  "id": "你的用户名/pytorch-object-detection-lab-voc2007-gpu-run-v7",
  "enable_gpu": "true",
  "enable_internet": "true",
  "machine_shape": "NvidiaTeslaT4"
}
```

只替换 `id` 中的账户名。`code_file` 应继续指向 `run_kaggle.py`，数据源列表保持为空。

## 3. 提交任务

在仓库根目录运行：

```bash
kaggle kernels push -p docs/recorded-run/kaggle
```

命令成功后会返回 Kaggle 页面地址。也可以查询状态：

```bash
kaggle kernels status <你的用户名>/pytorch-object-detection-lab-voc2007-gpu-run-v7
```

第一次提交后，在网页的 Settings 中确认：

- Accelerator 是 T4 或更新的 NVIDIA GPU。
- Internet 已开启。
- 任务状态是 Running，而不是 Error。

Kaggle 可能显示 `GPU T4 x2`。本项目是单设备训练，只使用 `cuda:0`，第二张卡空闲是正常
现象，不需要修改代码或重新提交。

## 4. 看懂日志

正常日志会依次出现：

```text
{"project": "/kaggle/working/project", ...}
{"phase": "download_voc2007", "status": "started"}
{"phase": "download_voc2007", "status": "completed", ...}
{"phase": "training", "status": "running", "elapsed_seconds": ...}
...
{"phase": "evaluation", "status": "running", "elapsed_seconds": ...}
```

训练和评估期间每 60 秒输出一次心跳。只要心跳和 epoch 日志继续更新，任务就仍在工作。
目标检测在完整 VOC 上训练几十分钟很正常，不要因为几分钟没有新指标就停止任务。

## 5. 确认完成

状态显示 `COMPLETE` 或网页显示 `Successfully ran` 后，先在最后一段日志中检查：

- `completed_epochs` 是 `26`。
- train / valid / test 数量是 `2501 / 2510 / 4952`。
- `best_epoch` 已记录。
- 测试评估使用 4,952 张图像。

已发布 v7 的网页总时间是 `3223.9s`。你的运行会因 Kaggle 机器和网络略有变化。

## 6. 下载训练产物

完整输出中还包含约 1.7 GB 的临时 VOC 数据。通常只需下载 `artifacts`：

```bash
kaggle kernels output <你的用户名>/pytorch-object-detection-lab-voc2007-gpu-run-v7 --file-pattern 'artifacts/.*' -p kaggle-output
```

下载后重点查看：

| 文件 | 先看什么 |
|---|---|
| `reference-fasterrcnn/metrics.csv` | 每轮 loss、验证 mAP、最佳轮次 |
| `reference-fasterrcnn/config.yaml` | Kaggle 实际使用的 CUDA、AMP 和路径 |
| `reference-fasterrcnn/best.pt` | 验证集选出的模型，可用于预测 |
| `reference-fasterrcnn/last.pt` | 最后一轮和续训状态 |
| `reference-fasterrcnn/evaluation/evaluation.json` | 测试集汇总指标 |
| `reference-fasterrcnn/evaluation/per_class.csv` | 20 类各自表现 |
| `reference-fasterrcnn/evaluation/visualizations/` | 预测、误检和漏检图 |
| `kaggle-run-summary.json` | 运行时间、划分数量和最终指标 |

## 已经遇到过的三种失败

### 找不到项目压缩包

早期 runner 期待一个外部源码压缩包，但非交互任务中没有附加该文件，日志显示：

```text
FileNotFoundError: expected one project archive, found []
```

当前 v7 runner 已内嵌精确源码，不需要手动上传压缩包。确认 metadata 的 `code_file` 指向
仓库内当前 `run_kaggle.py`，不要使用旧版本脚本。

### 非交互任务不能临时挂载新 Dataset

另一版 runner 在运行时调用 `kagglehub.dataset_download`，Kaggle 返回：

```text
New Datasets cannot be attached in non-interactive sessions
```

当前 runner 不使用 Kaggle Dataset 或 `kagglehub`，会直接从官方地址下载 VOC。保持
`dataset_sources` 为空即可。

### P100 与当前 PyTorch 不兼容

Tesla P100 的计算能力是 `sm_60`，当前 Kaggle PyTorch 构建只包含 `sm_70` 及以上内核，
会报：

```text
CUDA error: no kernel image is available for execution on the device
```

选择 T4 或更新 GPU。这个错误不是数据或模型代码造成的，换训练参数不能解决。

## 下一步

训练运行时阅读[训练教程](../tutorial/04-training.zh-CN.md)。任务完成后，按
[评估与预测](../tutorial/05-evaluation-and-inference.zh-CN.md)查看指标和图像。需要对照时，
打开项目的[已完成 Kaggle 运行](../recorded-run/README.zh-CN.md)。
