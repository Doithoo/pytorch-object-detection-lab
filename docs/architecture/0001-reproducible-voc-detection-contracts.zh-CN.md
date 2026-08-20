# ADR 0001：可复现目标检测契约

[English](0001-reproducible-voc-detection-contracts.md)

- 状态：已接受
- 读者：修改数据、模型、训练、评估或产物兼容性的维护者与贡献者

## 背景

目标检测同时涉及源图像与标注、数据划分、坐标转换、可变尺寸批次、模型模式、优化器状态、随机状态、指标后端和可视化产物。项目把这些边界明确记录，使保存的结果可以独立检查和比较，不依赖隐藏的本地状态。

仓库包含可复现的 VOC 2007 内容、自定义 VOC 形状类别支持、COCO JSON 提供器、五个 torchvision 检测器、显式外部模型工厂、只依赖 checkpoint 的预测，以及独立的 YOLO 数据导出。已完成的 Kaggle 运行是一个具体结果，不是普遍基准。

## 决策

### 准备后的数据具有身份

准备阶段会在发布 `train.csv`、`valid.csv`、`test.csv`、`dataset.yaml`、`source.yaml` 和 `summary.txt` 前校验源样本与划分成员。元数据记录有序类别、标签映射、标注格式、划分数量、源文件哈希、CSV 哈希、坐标规则和组合身份。

VOC 形状数据从通过校验的 XML 中推导非空类别名，官方 VOC 2007 保留发布的类别顺序。COCO JSON 接受不连续的 category ID，并写入稳定的连续标签映射。运行时目标统一使用背景标签 0、从 1 开始的前景标签、零基连续 `xyxy` 框、对齐字段和 `iscrowd` 语义。

训练和评估会在作出指标声明前验证元数据和源字节。清单引用源文件，不是数据集副本。

### Checkpoint 安全、明确且有版本

结构版本 1 保存解析配置、模型名称、可选显式工厂路径、模型参数、权重策略、有序类别、预处理规则、清单身份、划分哈希、模型/优化器/调度器状态、轮次、选定指标、指标历史、环境元数据和随机状态。

加载使用 `weights_only=True`。Checkpoint 不序列化可执行模型代码。内置模型从注册表重建；外部模型记录 `module:function` 路径，并在预测、评估或续训重建模型时要求该路径可导入。

### 模型共享简洁的检测接口

模型接收 `[0,1]` 范围的 RGB 浮点张量列表。训练模式接收对齐目标并返回标量 loss 映射；评估模式接收图像并返回每张图像的 `boxes`、`labels` 和 `scores`。训练器和评估器校验这套接口，模型内部逻辑仍由 torchvision 或显式外部工厂负责。

当前注册模型包括 Faster R-CNN MobileNet、Faster R-CNN ResNet-50、RetinaNet ResNet-50、FCOS ResNet-50 和 SSDLite MobileNet。模型修改示例展示如何替换 backbone 与 anchor generator，同时保持共享接口。

### 指标明确说明口径

评估记录 torchmetrics 提供的 COCO 风格 AP/AR，以及 IoU 0.5 下的 VOC 2007 十一点 AP。验证集可以使用 `map_50_95` 或 `voc_map_50_11` 选择 checkpoint。测试集用于最终固定比较，不用于反复调参。

### 发现和发布默认不修改已有内容

配置显示、模型列表、模型元数据、数据检查和运行比较不会构造模型、下载权重或改写已有产物。数据准备、评估、目录预测和 YOLO 导出都会暂存结果并原子发布；非空目标目录需要显式覆盖。

### YOLO 是数据导出边界

`export-yolo-data` 将已验证的数据转换为归一化 YOLO 文本标签和 `data.yaml`，但不会把仓库训练器替换为第三方引擎。不同 YOLO 实现的架构、loss、checkpoint、结果格式、依赖和许可证可能不同，需单独记录准确版本和许可。

## 结果

准备阶段会读取源文件并产生 I/O 成本，但之后可以用身份验证同一份数据。Checkpoint 因包含优化器、历史和随机状态而更大，但预测与续训不依赖未记录的 YAML。

共享检测接口使架构比较和模型修改更容易检查，但不能保证不同框架、硬件或模型自有后处理产生逐位相同的数值。结果应记录环境、配置、数据身份、指标定义和范围。

外部工厂增加灵活性，同时保留明确的导入边界；工厂路径和依赖也因此成为实验来源的一部分。YOLO 导出提供互操作性，但不会把不同训练引擎伪装成同一系统。

这些决策改善了可检查性和可移植性，但不会把 dry run、有界实验或配置文件变成完整基准。

参见[配置参考](../reference/config-reference.zh-CN.md)、[数据格式](../reference/dataset-format.zh-CN.md)、[checkpoint 结构](../reference/checkpoint-schema.zh-CN.md)、[模型参考](../reference/model-zoo.zh-CN.md)和[指标参考](../reference/metrics.zh-CN.md)。
