# 教程 01：环境、设备与信任边界

[English](01-environment.md) | [教程索引](README.zh-CN.md)

目标是建立可重复、网络与硬件决策都清楚可见的环境。前提是 Python 3.10-3.12、`uv` 和
本仓库的克隆副本。训练试运行之前的检查都不需要 VOC。

## 严格安装锁定环境

在仓库根目录运行：

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect --help
```

`uv sync --locked` 安装 `uv.lock` 中已经解析的依赖；若锁文件与环境不兼容，它会失败，
而不是悄悄改变解析结果。`--extra dev` 加入本项目的测试和文档工具。当前版本输出是
`0.1.0`；帮助信息会列出 `prepare-data`、`inspect-data`、`train`、`evaluate`、`predict`
等命令，而且不会加载模型。

确认 Python、PyTorch 和本项目包来自同一个环境：

```bash
uv run python -c "import sys, torch, object_detector; print(sys.version); print(torch.__version__); print(object_detector.__file__)"
```

最后一行应指向本仓库的 `src/object_detector`。如果它指向另一份工作副本，应先解决环境
问题，而不是调整数据或模型。

## 检查 CPU、CUDA 与 Apple MPS

```bash
uv run python -c "import torch; print('cuda', torch.cuda.is_available()); print('mps', torch.backends.mps.is_available()); print('cpu', True)"
uv run python -c "from object_detector.preflight import resolve_device; print(resolve_device('auto'))"
```

项目的 `device: auto` 按 CUDA、MPS、CPU 的顺序选择。排查集成流程时先显式使用
`--device cpu`。训练预检会拒绝不可用的显式 `cuda` 或 `mps`。MPS 使用全精度，AMP
缩放器只在 CUDA 上启用。本项目是单设备训练，不实现分布式训练。

第 02 章准备数据后，用下面命令检查完整训练链路：

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

预期输出包含原始图像形状、目标数和有限的具名损失项，并以 `dry-run OK` 结束。试运行会
在内存中执行一次优化器更新，但不发布运行目录或检查点。

## 构造模型前解析配置

```bash
uv run detect show-config --config configs/learning_minimal.yaml
```

预期 YAML 包含 `weights: none`、训练/验证/测试的 `32/16/16` 样本上限、两轮训练、
`num_workers: 0` 和 `device: auto`，并标出每个最终值来自默认值还是 YAML。实验使用
`--set KEY VALUE` 后，也应先用这条命令核对最终配置。

## 权重策略就是网络边界

`configs/learning_minimal.yaml` 使用 `model.weights: none`。模型注册表会把检测器和
骨干网络权重都传为 `None`，因此模型构造不会请求预训练权重。这是可靠的离线教学路径。

`model.weights: imagenet1k_v1` 则不同。训练预检会检查预期的本地 Torch Hub 检查点
路径；文件不存在时，它会提示模型构造需要网络。本仓库不保证网络可用，也不保证缓存中
恰好有正确文件。必须在运行前明确决定策略，不能根据模型名称猜测。

评估和预测从本项目的自包含检查点读取状态，以 `weights=none` 重建模型，再加载
`model_state`。两者需要本地检查点；评估还需要身份标识匹配的准备数据，但都不需要重新
下载骨干网络权重。

## 数据下载是另一条信任边界

模型可以离线构造，并不代表 VOC 已经存在。第 02 章的 `scripts/download_data.py` 只在本地
没有通过校验和的归档文件时访问 Oxford 官方 VOC HTTP 地址。传输内容先写入 `.part`，
随后校验官方 MD5，并拒绝不安全的 tar 条目。网络是否可用不属于仓库承诺。

## 常见失败边界

- `uv sync --locked` 报锁文件不兼容：检查 Python 版本和已提交的锁文件，不要删掉
  `--locked` 来掩盖问题。
- 其他 Python 环境能导入包，但 `uv run detect` 不存在：当前环境或工作副本不对。
- CUDA 不可用：先在项目外核对 PyTorch 构建和驱动，再修改检测器设置。
- MPS 某项操作失败：用 CPU 和 `num_workers: 0` 复现，区分设备后端与数据问题。
- 训练提示预训练权重缓存缺失：当前权重策略在本机不是离线保证。
- 试运行提示缺少 `dataset.yaml`：基础环境通过了，但还没有跨过数据准备边界。

下一步阅读[教程 02](02-data-and-boxes.zh-CN.md)，下载、校验、固定、检查并预览后续运行
实际使用的数据。
