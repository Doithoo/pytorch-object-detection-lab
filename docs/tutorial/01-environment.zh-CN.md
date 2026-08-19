# 教程 01：选择训练环境

[English](01-environment.md) | [教程索引](README.zh-CN.md)

本教程推荐在 Kaggle 训练。Kaggle 已经准备好 Python、PyTorch 和 NVIDIA GPU，可以把
时间放在数据、模型和结果上，而不是先处理本地 CUDA 与驱动。

## 推荐：Kaggle GPU

项目提供可以直接提交的 runner。先在本地安装 Kaggle CLI：

```bash
uv tool install kaggle
kaggle auth login
```

然后按 [Kaggle 训练指南](../guides/kaggle.zh-CN.md)修改任务账户并提交。网页中需要：

- T4 或更新的 NVIDIA GPU。不要选择 P100，当前 PyTorch 构建不支持它的 `sm_60`。
- Internet 开启，用于下载官方 VOC 2007 和 ImageNet backbone 权重。
- 约 60 分钟运行时间。

Kaggle 显示 T4 x2 时，本项目仍只使用 `cuda:0`。第二张卡空闲不影响训练。

## 可选：在本地查看项目

本地只需要 Python 3.10-3.12 和 [uv](https://docs.astral.sh/uv/)。在仓库根目录运行：

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect --help
uv run detect list-models
```

版本应为 `0.1.0`，帮助中应包含 `prepare-data`、`inspect-data`、`train`、`evaluate` 和
`predict`。这些命令不会开始训练或下载权重。

查看 Kaggle 参考配置：

```bash
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

`show-config` 只显示最终配置及其来源，不构造模型。参考配置使用
`imagenet1k_v1` backbone；Kaggle runner 会在联网环境中下载对应权重。

## 可选：检查本地设备

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available()); print('cpu', True)"
uv run python -c "from object_detector.preflight import resolve_device; print(resolve_device('auto'))"
```

`device: auto` 按 CUDA、Apple MPS、CPU 的顺序选择。本地没有 CUDA 并不是问题，可以继续
使用 Kaggle。CPU 适合小例子和 dry run，不适合完整的 26 轮 VOC 训练。

## 可选：完成一次 CPU dry run

第 02 章准备好本地数据后，可以运行：

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

它读取一个 batch，执行前向传播、反向传播和一次参数更新，最后打印 `dry-run OK`。它不
保存 checkpoint，也不代表模型已经训练完成。

## 常见问题

- `kaggle` 命令不存在：重新运行 `uv tool install kaggle`，并确认 uv 工具目录在 PATH。
- Kaggle API 拒绝认证：运行 `kaggle auth login --force`。
- Kaggle 页面没有 GPU：完成平台要求的账户验证，并检查 GPU 配额。
- 本地 CUDA 不可用：直接使用 Kaggle；不需要为了学习本项目重装整个本地环境。
- `uv sync --locked` 失败：确认 Python 版本在 3.10-3.12 范围内。
- P100 运行时报 `no kernel image`：换成 T4 或更新 GPU。

下一步阅读 [VOC 数据与边界框](02-data-and-boxes.zh-CN.md)。
