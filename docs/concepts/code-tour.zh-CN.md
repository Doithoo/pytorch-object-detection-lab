# 从命令到结果阅读代码

[English](code-tour.md) | [检测流程](detection-flow.zh-CN.md)

本页帮助贡献者在修改前确定职责归属。应沿着一条命令阅读包，而不是按字母顺序浏览模块。

```text
detect train
  -> object_detector.cli._train
  -> object_detector.config.load_config
  -> object_detector.training.train.run_training
  -> 清单元数据与预检查
  -> VocDetectionDataset 与列表批处理
  -> 模型注册表与 torchvision 构造器
  -> trainer.train_one_epoch / DetectionMetric
  -> 原子配置、元数据、指标与检查点
```

## 命令与配置

`src/object_detector/cli.py` 负责 argparse 名称、仅运行时参数、简洁标准输出，并把捕获的 `ValueError`、`RuntimeError` 或 `OSError` 转成 `error: ...` 和退出码 2；领域工作交给其他模块。`src/object_detector/config.py` 负责数据类默认值、严格 YAML 合并、有类型的 `--set`、来源跟踪、路径构造和值校验。`src/object_detector/preflight.py` 负责清单文件、类别数、设备、输出、缓存准备情况，以及 `auto` 设备解析。

运行 `uv run detect show-config --config configs/learning_minimal.yaml` 只查看第一部分。预期输出是解析 YAML 和 `sources`，不会读取数据、构造模型、访问网络或写输出。

## 数据加载

按源数据进入顺序阅读：

1. `data/schema.py` 定义 VOC 类别和解析后的标注记录。
2. `data/voc.py` 解析 XML，并转换一基且端点包含的检测框。
3. `data/manifest.py` 校验每个划分样本、对源内容求哈希，并原子发布清单。
4. `data/dataset.py` 把清单行解析成浮点 RGB 张量和 torchvision 目标，只在训练时过滤困难目标。
5. `data/transforms.py` 在几何变化中保持检测框与逐目标字段对齐。
6. `data/inspection.py` 计算有界摘要并渲染目标预览。

`detection_collate` 有意返回两个列表，不填充或堆叠不同大小的图像。类别、困难标记、尺寸或几何出现问题时，应先检查数据流程，再检查模型。

## 模型

`models/spec.py` 定义注册元数据和构造器类型。`models/registry.py` 负责稳定名称、支持的权重策略、缓存路径推导、保留参数和相近名称错误。`models/torchvision_models.py` 是唯一把项目策略转换成 torchvision 构造参数的层。骨干网络、FPN、RPN、ROI Align、预测头、模型自有变换和非极大值抑制仍由 torchvision 负责。

`uv run detect list-models` 和 `uv run detect model-info fasterrcnn_resnet50_fpn` 能在不构造模型时查看元数据。`examples/03_model_contract.py` 使用合成输入构造模型并运行两种状态，但不能衡量学习质量。

## 训练与检查点

`training/trainer.py` 负责移动批次、求和损失、有限标量检查、反向传播、可选梯度裁剪、优化器更新、单轮训练和试运行诊断。`training/train.py` 负责随机种子、数据集与加载器、优化器与调度器构造、验证、最佳与最后检查点选择、续训语义检查、历史和运行产物。`training/checkpoint.py` 负责版本 1 结构、严格预处理规则、受限的张量与容器安全加载（允许安全的基础值、列表、映射和张量）、原子保存、续训标识和环境元数据。

Faster R-CNN 的四个损失键由 torchvision 产生，不是训练器写死的。训练器也能记录其他注册检测器的损失映射。续训恢复模型、优化器、可选调度器、历史和随机数流，不能用来改变实验语义。

## 评估与预测

`evaluation/metrics.py` 把预测和目标适配给 torchmetrics，只把后端负哨兵值截到零。`evaluation/errors.py` 负责确定性的同类别贪心错误标签。`evaluation/visualization.py` 渲染普通、困难和预测框。`evaluation/evaluate.py` 从 checkpoint 重建、验证清单标识、流式处理批次、写 JSON/CSV、只重新加载排序后的错误图像，并发布输出目录。`evaluation/comparison.py` 读取现有运行文件，报告兼容指标和配置差异。

`inference/predictor.py` 用 `weights="none"` 从检查点重建架构和有序类别。单图预测会保护两个输出免遭意外覆盖，其中 JSON 原子写入，PNG 渲染器直接保存；目录预测暂存完整目录树，记录不可读图像错误，再整体发布。它不声明数据集指标，因此不需要 YAML 或清单。

## 修改后检查哪里

可用 `tests/test_end_to_end.py` 阅读可执行的包路径，并针对修改内容运行专门测试。解析器单元测试不覆盖模型集成，dry run 不衡量学习质量，小规模运行也不能建立完整 VOC 结果。[Kaggle 训练记录](../recorded-run/README.zh-CN.md)来自单独的真实执行；`configs/reference_fasterrcnn.yaml` 本身只是一份配置。

接下来可阅读[检测流程](detection-flow.zh-CN.md)了解张量职责，[配置流程](configuration-flow.zh-CN.md)了解优先级，或参考[添加数据集](../guides/adding-datasets.zh-CN.md)与[添加模型](../guides/adding-models.zh-CN.md)中的内部扩展清单。
