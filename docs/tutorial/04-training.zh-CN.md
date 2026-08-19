# 教程 04：先完成一次参数更新，再决定怎样训练

[English](04-training.md) | [教程索引](README.zh-CN.md)

前提是已经准备好 `data/manifests`，`data/raw` 下也有匹配的源图，并理解教程 00-03 中的
模型和数据契约。除非 CUDA 或 MPS 试运行已经通过，第一次应先使用 CPU。

## 单独读懂一次优化器更新

```bash
uv run python examples/04_minimal_training_loop.py --lr 0.1
```

预期输出 `scale: 1.0000 -> 0.7500`。这个假检测器只有一个参数和两项合成损失，因此参数
变化很容易看清。示例代码的实际顺序是“前向传播 -> `zero_grad` -> 求和 -> 反向传播 ->
更新参数”。清空梯度必须发生在 `backward` 之前；这里放在简单前向传播之后仍然有效，
因为前向传播本身不会累积参数梯度。

生产代码中的 `dry_run` 会先清空梯度，再进行前向传播、求和、反向传播和优化器更新。
它还会把每张图像和标注字典移动到指定设备，检查每项损失都是有限标量，并按配置选择
是否裁剪梯度。完整训练轮次会按图像数平均损失。

## 试运行：证明一次集成更新

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

命令读取一个真实训练批次，以训练模式运行配置中的 torchvision 模型，对损失求和，
反向传播并更新一次参数。预期标准输出包含：

```text
image_shapes=((3, H1, W1), (3, H2, W2))
target_counts=(N1, N2)
loss_total=<有限值>
loss_classifier=<有限值>
loss_box_reg=<有限值>
loss_objectness=<有限值>
loss_rpn_box_reg=<有限值>
dry-run OK
```

实际形状、数量和数值取决于所选清单行与模型状态。`dry-run OK` 证明一个批次的数据加载、
整理、前向传播、反向传播和优化器更新已经连通。它不写正常运行目录，也不测量验证质量。

## 有界学习：证明产物链路

`configs/learning_minimal.yaml` 将 train/valid/test 限为 `32/16/16` 个样本，使用两轮训练、
随机权重和零个数据加载工作进程。显式指定运行名称：

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --device cpu
```

成功时，标准输出为 `artifacts/first-detector`。按下面顺序检查：

1. `config.yaml`：完整解析后的配置，包括样本上限。
2. `run.yaml`：环境、设备、随机种子、数据身份标识、划分哈希值和类别顺序。
3. `metrics.csv`：每轮一行，包含 `loss_total`、四项检测器损失，以及以 `valid_` 开头的
   AP、AR 和计数字段。
4. `best.pt`：严格提升验证集 `map_50_95` 的轮次。
5. `last.pt`：最近完成的轮次，并包含优化器、调度器、历史记录和续训所需的随机数状态。

对于全新运行，只要解析后的运行目录已经存在，命令就会拒绝继续，即使该目录为空也是
如此。检查点与文本产物都通过原子方式写入，因此已经完成的单个文件不会是只写了一半
的替换内容。

这次有界运行只证明配置子集上的学习和产物链路。指标只描述这些样本、轮次、随机种子和
配置，不是完整 VOC 基准。

## 续训不能改变实验语义

把同一运行从两轮延长到三轮：

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --set train.epochs 3 --resume artifacts/first-detector/last.pt --device cpu
```

上面的命令从当前 `last.pt` 原位延长已有运行。也可以选择新的空运行目录；跨目录续训会
带入兼容的同级 `best.pt`。原目录已有 `last.pt` 时，从 `best.pt` 或旧副本恢复会被拒绝，
避免覆盖较新的历史；若 `last.pt` 缺失，则只能用原始精确路径的 `best.pt` 恢复同一目录，
重命名或复制的其他检查点不能原位恢复。续训还要求配置的验证指标均为有限数值，且
`best_metric` 等于完整历史最大值；模型、类别、预处理契约、数据身份标识和实验语义配置
仍必须一致。总轮数、数据加载工作进程数和设备等操作字段可以调整，但请求的总轮数必须
大于检查点保存的轮次。

## 完整训练属于另一层证据

`configs/reference_fasterrcnn.yaml` 不设样本上限，要求 26 轮训练、官方准备的数据划分和
`imagenet1k_v1` 骨干网络权重。考虑运行前，必须核对完整数据身份标识、设备容量、权重
缓存或网络策略、输出空间，以及[运行记录证据门槛](../recorded-run/README.zh-CN.md)。

对应命令是：

```bash
uv run detect train --config configs/reference_fasterrcnn.yaml
```

这不是快速教程命令；列出命令也不代表已经执行。独立的
[Kaggle 实测运行](../recorded-run/README.zh-CN.md)保存了一次真实 26 轮执行，包括解析后的
CUDA/AMP 配置、耗时、验证集选择、测试结果、checkpoint 哈希和图像。

## 常见失败边界

- 训练预检报告清单缺失、类别数不符、设备不可用或输出位置不可写：构造模型前先修复。
- 出现预训练权重提示：预期缓存不存在，所选模型可能需要网络。
- 损失为 NaN 或无穷大：训练器会报告损失名和图像 ID，应先检查样本再修改优化参数。
- `best.pt` 与 `last.pt` 不同：最后一轮验证 AP 没有严格提升时，这是正常现象。
- 续训改变批次大小、优化器、学习率、数据增强或样本上限：训练器会拒绝，应新建运行。
- 两轮有界指标看起来很高或很低：仍不能把它提升为完整 VOC 结论。

下一步进入[教程 05](05-evaluation-and-inference.zh-CN.md)，评估选定检查点、检查错误证据，
并执行只依赖检查点的预测。
