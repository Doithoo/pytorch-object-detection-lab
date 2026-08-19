# 端到端检测流程

[English](detection-flow.md) | [代码导览](code-tour.zh-CN.md)

本页帮助读者连接命令行值、数据张量、torchvision 模式和最终产物，描述真实训练、评估与预测路径中的职责。

## 从命令到预检查

```text
detect train
  -> 严格的默认值 < YAML < 重复 --set
  -> 最终运行时 --device 覆盖
  -> 加载 dataset.yaml
  -> 预检查清单、类别数、设备、输出和权重缓存
  -> 解析 auto 设备并构造模型
```

配置错误会在数据或模型工作前停止。元数据加载证明 `dataset.yaml` 可以解析；随后预检查要求三个 CSV 和元数据文件，核对 `expected_num_classes == len(class_names)+1`，拒绝不可用的显式加速器和不可写的输出上级目录，并在具名骨干网络权重未缓存时提示。提示表示构造时可能需要网络，不表示已经下载。

## 从源数据到不可混淆的清单标识

```text
VOC 划分标识 + JPEG 字节 + XML 字节
  -> 校验划分互斥、配对、解码、尺寸、类别和检测框
  -> 每个划分的 SHA-256
  -> 标识(名称, 类别, 坐标约定, 划分哈希)
  -> train.csv / valid.csv / test.csv + dataset.yaml + 来源信息
```

准备阶段先暂存全部输出，再原子替换清单目录。CSV 行只引用源文件，不把源数据复制进清单。因此运行时同时需要清单目录和匹配的源根目录。之后修改源内容，必须重新准备并使用新标识。

## 从清单行到可变批次

一行被解码为 `[0,1]` 范围的 RGB `image: float32 [3,H,W]`，以及逐目标字段共享 `N` 的目标映射。VOC 检测框转换成零基连续 `float32 [N,4]`，标签为 `int64 [N]`，0 保留给背景。`image_id`、`area`、`iscrowd` 和 `difficult` 保存标识与评估语义。

训练在随机水平翻转前移除困难目标；评估和检查保留它们。退化框过滤会对全部对齐字段应用同一个保留掩码。空目标仍是有明确形状的有效张量。

```text
样本 1: Tensor[3,H1,W1] + 目标[N1]
样本 2: Tensor[3,H2,W2] + 目标[N2]
  -> detection_collate
图像张量列表 + 目标列表
```

项目不会在此缩放、填充或堆叠批次。torchvision 检测器的变换负责归一化、保持比例的缩放、填充，以及把最终检测框映射回调用者坐标。

## 同一模块的两种接口模式

```text
model.train(); model(images, targets)
  -> 具名标量损失映射
  -> 有限检查 -> 求和 -> 反向传播 -> 可选裁剪 -> 优化器更新

model.eval(); 在 inference_mode 下 model(images)
  -> 每张图像一个 {boxes, labels, scores} 预测映射的列表
  -> 指标 / JSON / 错误分析 / 可视化
```

Faster R-CNN 的损失键是 `loss_classifier`、`loss_box_reg`、`loss_objectness` 和 `loss_rpn_box_reg`；`loss_total` 是项目对它们求和的结果。其他注册家族可以返回不同的具名损失。传入错误参数或按错误模式解释结果，都会违反模型接口。

使用随机合成张量运行两种模式：

```bash
uv run python examples/03_model_contract.py
```

预期输出列出 Faster R-CNN 训练损失键，以及评估键 `boxes`、`labels`、`scores`。它只检查模型构造和输入输出形状，不下载权重、不训练，也不发布指标。

## 训练、验证与原子运行产物

训练器按图像数平均各项损失。每轮之后，验证把模型返回的全部检测交给 AP/AR，不应用项目分数过滤。编排器追加一行历史，更新可选固定 StepLR，并且只有验证 `map_50_95` 严格超过历史最佳时才更新 `best.pt`。`last.pt` 总是记录最新完成轮次。

```text
artifacts/<运行名>/
  config.yaml   解析配置
  run.yaml      环境与清单标识
  metrics.csv   轮次损失、验证 AP/AR 与计数
  best.pt       最佳验证检查点
  last.pt       最新可续训检查点
```

文本文件和检查点都通过临时文件与 `os.replace` 写入。新运行会拒绝已有运行目录。试运行在完成一次集成优化器更新后停止，不写上述正常产物。

## 从检查点到评估或预测

两个消费者都通过 `torch.load(..., weights_only=True)` 加载版本 1，严格校验预处理规则，用 `weights="none"` 重建注册模型，再加载保存状态。

评估还会加载保存的解析配置，要求当前清单标识匹配，读取带标签划分，把模型输出交给指标，再应用独立的序列化阈值和错误阈值，并写出 JSON、CSV 与排序 PNG 图像。预测只需要检查点和新图像，会恢复有序类别与清单来源，但不加载 VOC 数据，也不声明 AP。

目录预测递归处理 `.jpg`、`.jpeg` 和 `.png`，记录不可读图像，并发布完整暂存输出树。单图预测在防覆盖条件下写一个 JSON 与 PNG；JSON 原子写入，PNG 直接保存。

## 每项检查能说明什么

`show-config` 检查值解析，`inspect-data` 检查目标加载，`train --dry-run` 检查一次更新，小规模运行检查输出创建，验证集评估检查配置子集上的指标。它们都不能建立完整 VOC 结果，参考 YAML 也不是一次已完成训练。

可继续阅读[Faster R-CNN 工作原理](how-faster-rcnn-works.zh-CN.md)了解检测器内部职责，[检查点结构](../reference/checkpoint-schema.zh-CN.md)了解恢复，或阅读[指标](../reference/metrics.zh-CN.md)解释输出。
