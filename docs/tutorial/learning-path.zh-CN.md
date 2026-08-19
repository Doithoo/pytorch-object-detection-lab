# 目标检测学习路径

[English](learning-path.md) | [教程索引](README.zh-CN.md)

这条路线面向已经理解基本张量和梯度、但还没有完整走过两阶段目标检测流程的学习者。
依次完成 `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`，每进入
下一阶段前，先说清上一阶段究竟证明了什么。

## 0. 建立锁定环境

```bash
uv sync --locked --extra dev
uv run detect --version
uv run detect show-config --config configs/learning_minimal.yaml
```

预期版本为 `0.1.0`。解析后的配置应显示随机权重、两轮训练、`32/16/16` 的样本上限和
自动设备选择。若这些命令失败，先读[教程 01](01-environment.zh-CN.md)，不要同时调整数据
和 GPU 设置。

完成标准：能解释为什么 `weights: none` 是离线构造模型的策略，却不代表 VOC 已经安装。

## 1. 识别检测张量与列表

```bash
uv run python examples/01_boxes_and_labels.py
uv run python examples/02_detection_batch.py
```

先预测面积 `192` 和 `180`，再看示例 01 的输出。面对示例 02，要能解释为什么形状为
`(3,16,20)` 与 `(3,12,24)` 的两张图像要保留在列表中，为什么标注也要放在对应列表中，
为什么框的形状是 `[N,4]`，以及目标类别为何从 1 开始。不清楚时阅读
[教程 00](00-basics.zh-CN.md)。

完成标准：写出合法空标注的张量形状，并手算两个部分重叠框的 IoU。

## 2. 建立可信的数据准备边界

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split valid --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split valid --limit 4 --output artifacts/dataset_preview.png
```

第一条命令跨越网络边界，并校验两个官方 VOC MD5 值。准备命令校验数据，再原子发布固定
清单和由内容计算的数据身份标识。检查命令输出结构化计数和范围。预览命令只渲染所选前几
行中实际存在的普通目标与困难目标；只有这些行含困难目标时，图中才会出现橙色
虚线框。所有需要比较的运行都应把这份数据身份标识视为不可变输入。详见
[教程 02](02-data-and-boxes.zh-CN.md)。

完成标准：解释 `(xmin-1,ymin-1,xmax,ymax)`，在明确标为合成内容的
[标注结构图](../assets/detection-target-anatomy.png)中指出带文字标签的困难目标，并
区分“已检查图像数”和“完整划分图像数”。

## 3. 跨过真实模型的模式边界

```bash
uv run python examples/03_model_contract.py
```

训练模式预期返回 `loss_classifier`、`loss_box_reg`、`loss_objectness`、
`loss_rpn_box_reg`；评估模式预期返回 `boxes`、`labels`、`scores`。示例使用随机权重和
合成输入，只验证 torchvision 的调用契约，不进行学习。张量职责见
[教程 03](03-faster-rcnn.zh-CN.md)。

完成标准：能追踪“图像列表 -> 填充后的图像批次 -> 骨干网络/FPN -> RPN 候选区域 ->
ROI 预测”，并指出两个模块各自对应哪两项损失。

## 4. 完成一次参数更新

```bash
uv run python examples/04_minimal_training_loop.py --lr 0.1
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

小型示例让单个参数的变化清晰可见。随后，试运行会在一批准备好的数据上更新配置中的
检测器，并以 `dry-run OK` 结束。它不写检查点，也不报告模型质量。用
[教程 04](04-training.zh-CN.md)区分这种探针和正式训练证据。

完成标准：指出哪一步清空旧梯度、构造标量损失、计算新梯度，以及真正修改参数。

## 5. 完成一次有界学习运行

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --device cpu
```

依次检查 `artifacts/first-detector` 下的 `config.yaml`、`run.yaml`、`metrics.csv`、
`best.pt`、`last.pt`。`32/16/16` 的样本上限和两轮训练说明它是流程学习，不是完整 VOC
基准。要能解释为什么验证集 `map_50_95` 选择 `best.pt`，以及它为何可能与 `last.pt` 不同。

完成标准：从产物而不是记忆中找出准确的数据身份标识、权重策略、设备、样本上限、被选
轮次和四项损失列。

## 6. 评估证据，然后预测

```bash
uv run detect evaluate --checkpoint artifacts/first-detector/best.pt --split valid --output-dir artifacts/first-detector/evaluation-valid --device cpu
uv run detect predict --checkpoint artifacts/first-detector/best.pt --image docs/assets/detection-target-anatomy.png --output-dir artifacts/prediction --device cpu
```

评估命令需要身份匹配的准备数据，并写入 AP/AR、预测、逐类别记录、分类后的错误和排序
证据图。预测命令只需要检查点与仓库自带的合成教学图，并在
`artifacts/prediction` 下写入 `detection-target-anatomy.json` 和
`detection-target-anatomy.png`；这只检查推理和产物写入机制，不能证明检测器质量。阅读
[教程 05](05-evaluation-and-inference.zh-CN.md)，再用 CSV 行和可视化共同解释一个漏检或
假阳性案例。

完成标准：说明 IoU 阈值与分数阈值为什么不同，困难目标匹配为什么被忽略，以及做选择
时为什么使用验证集而不是测试集。

## 7. 比较一个受控变量

创建两个名称不同的有界运行，保持数据身份标识、随机种子、样本上限和评估协议相同，只
改变一个有意研究的配置字段，然后运行：

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

报告会为每个运行选择最佳记录，验证数据身份标识相同，并列出影响实验语义的配置差异。
它会有意排除操作性字段 `run_name`、`output_dir`、`device` 和 `data.num_workers`。还要把
表格与曲线、视觉错误放在一起比较。两个有界运行只能说明这两次运行中发生了什么。

## 证据边界

合成示例证明局部张量和 API 契约；试运行证明一次集成更新；有界运行证明配置子集上的
产物与评估路径。它们都不是完整 Pascal VOC 基准。独立的
[完整 VOC 实测运行](../recorded-run/README.zh-CN.md)保存了形成这类结论所需的来源、范围、
指标、耗时、checkpoint 哈希和真实图像。
