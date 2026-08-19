# 选择并使用注册模型

[English](using-models.md) | [模型目录](../reference/model-zoo.zh-CN.md)

本指南帮助学习者在三个由项目维护的 torchvision 检测器之间做选择，内容包括发现、权重策略、试运行证据和对比。本文不会宣称某个模型速度或精度最好；唯一的[完整 VOC 实测运行](../recorded-run/README.zh-CN.md)只覆盖 Faster R-CNN MobileNet 配方，不是三个模型的对比。

## 修改 YAML 前先发现模型

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
uv run detect model-info fasterrcnn_resnet50_fpn
uv run detect model-info ssdlite320_mobilenet_v3_large
```

`list-models` 会打印稳定名称、`two_stage` 或 `one_stage` 家族和支持的权重策略，不会构造模型或访问网络。`model-info` 还会给出项目维护的参数名和输入说明。可用 Faster R-CNN MobileNet 学习教程中的两阶段流程，用 Faster R-CNN ResNet-50 在保留家族的同时更换骨干网络，或用 SSDLite 对比单阶段检测器。这些是结构选择，不是基准排名。

## 有意识地选择权重策略

`weights: none` 会把完整检测器权重和骨干网络权重都设为 `None`。模型随机初始化且构造过程离线，但源数据仍需位于本地。它适合示例、试运行、契约测试，以及需要隔离架构因素的实验。

`weights: imagenet1k_v1` 仍将完整检测器权重设为 `None`，但请求固定的 torchvision ImageNet 骨干网络枚举。训练预检查会计算预期的 torch hub 检查点路径。文件存在时不会显示网络提示；文件缺失时会提示模型构造需要网络，随后由 torchvision 尝试下载到 torch 缓存，或者报告底层网络或缓存错误。预检查本身不会下载，项目也没有单独的模型权重下载参数。

用试运行检查选择的路径：

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name model-check --dry-run --device cpu
```

预期输出包含图像形状、目标数量、有限的具名损失项和 `dry-run OK`。命令会执行一次优化器更新，但不写运行目录。若要离线使用预训练骨干网络，应在构造模型前把准确的 torchvision 文件放入缓存，不要把无关文件改名冒充。缓存路径来自 `torch.hub.get_dir()/checkpoints`，不同环境可能不同。

## 每次只切换一个变量

使用已有配方，或只覆盖 `model.name`，同时保持清单标识、权重策略、样本上限、随机种子、优化器和轮次一致：

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name faster-mobile --device cpu
uv run detect train --config configs/learning_minimal.yaml --set run_name faster-resnet --set model.name fasterrcnn_resnet50_fpn --device cpu
```

VOC 有 20 个目标类别和一个背景类，因此两份配置都需要 `model.expected_num_classes: 21`。模型专用值放在 `model.params` 下；项目维护的参数键见[模型目录](../reference/model-zoo.zh-CN.md)。其中不能覆盖保留项 `weights`、`weights_backbone` 和 `num_classes`。

两次运行完成后，对比相同清单上的验证证据：

```bash
uv run detect compare-runs artifacts/faster-mobile artifacts/faster-resnet --metric valid_map_50_95 --output artifacts/model-comparison.csv
```

除可选的新 CSV 外，该命令只读。它会拒绝不同的清单标识，并报告语义配置差异。模型选择使用验证集，固定选择后再评估一次测试集。完整实验纪律见[实验管理](experiments.zh-CN.md)，内部扩展步骤见[添加模型](adding-models.zh-CN.md)。
