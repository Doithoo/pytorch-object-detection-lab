# 注册模型参考

[English](model-zoo.md) | [使用模型](../guides/using-models.zh-CN.md)

本页定义版本 0.1 的注册表契约，不是排行榜。三个模型都是由本仓库维护调用方式的 torchvision 构造器。项目没有稳定的外部模型插件接口。Faster R-CNN MobileNet 配方已有一次[完整 VOC 实测](../recorded-run/README.zh-CN.md)，但它没有对其他模型进行排名。

## 发现

```bash
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

这些命令只读取注册元数据，不构造模型、不检查权重缓存、不访问网络，也不写产物。未知名称会失败，并可能提示相近的注册名称。

## 注册项

| 名称 | 家族 | 骨干网络与输入职责 | 自带配方与对比用途 |
|---|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | `two_stage` | 带 FPN 的 MobileNet V3 Large；检测器变换默认短边 320、长边上限 640 | `configs/learning_minimal.yaml` 的默认教程基线；用于学习骨干网络/FPN、RPN 和 ROI 头 |
| `fasterrcnn_resnet50_fpn` | `two_stage` | 带 FPN 的 ResNet-50；检测器默认短边 800、长边上限 1333 | `configs/fasterrcnn_resnet50_fpn.yaml`；保留 Faster R-CNN 并更换骨干网络，比 MobileNet 配方需要更多内存与计算 |
| `ssdlite320_mobilenet_v3_large` | `one_stage` | MobileNet V3 Large；内置变换缩放到 320 像素 SSDLite 配方 | `configs/ssdlite320_mobilenet_v3.yaml`；用于比较单阶段家族 |

上述用途描述架构和项目维护的配方，不代表相对精度、吞吐量、收敛速度、硬件支持或普遍适用性。

## 权重策略

每个注册项都准确支持 `none` 和 `imagenet1k_v1`。

| 策略 | 完整检测器 | 骨干网络 | 缓存与网络行为 |
|---|---|---|---|
| `none` | `weights=None` | `weights_backbone=None` | 离线构造路径；随机初始化 |
| `imagenet1k_v1` | `weights=None` | 对应骨干网络的固定 torchvision `IMAGENET1K_V1` 枚举 | 使用已有预期缓存文件，否则在构造时由 torchvision 尝试下载 |

两个 MobileNet 模型预期 `torch.hub.get_dir()/checkpoints/mobilenet_v3_large-8738ca79.pth`；ResNet-50 预期 `torch.hub.get_dir()/checkpoints/resnet50-0676ba61.pth`。预检查只检查文件是否存在，缺失时打印网络提示；它不会下载，也不会校验任意替代文件。检查点评估和预测始终用 `none` 重建，再加载模型状态。

## 项目维护的 `model.params`

注册表只接受下列构造器键。值按 YAML 解析并传给 torchvision；项目没有增加范围校验，因此具体取值错误仍由上游构造器或运行时报出。拼写错误或未维护的键会在构造前失败。

| 模型 | 键 | 该构造器的上游默认值 | 类型与效果 |
|---|---|---:|---|
| 两个 Faster R-CNN | `min_size` | MobileNet 320；ResNet-50 800 | 正整数，检测器变换负责的短边目标 |
| 两个 Faster R-CNN | `max_size` | MobileNet 640；ResNet-50 1333 | 正整数，保持长宽比缩放后的长边上限 |
| 两个 Faster R-CNN | `box_score_thresh` | `0.05` | 数值，ROI 推理分数阈值 |
| SSDLite | `score_thresh` | `0.001` | 数值，NMS 前的推理分数阈值 |
| SSDLite | `nms_thresh` | `0.55` | 数值，非极大值抑制的 IoU 阈值 |
| SSDLite | `detections_per_img` | `300` | 正整数，NMS 后每图检测上限 |

示例：

```yaml
model:
  name: fasterrcnn_mobilenet_v3_large_320_fpn
  weights: none
  expected_num_classes: 21
  params:
    min_size: 320
    max_size: 640
    box_score_thresh: 0.05
```

`weights`、`weights_backbone` 和 `num_classes` 是保留项，放入 `model.params` 会失败。当前模型未列出的键同样会失败；可用 `detect model-info MODEL_NAME` 查看项目维护的参数表面。

## 共享输入与模式契约

所有模型都接收 `[0,1]` 范围的 RGB 浮点张量列表 `[3,H,W]`；模型自有变换负责内部归一化、缩放和批处理。训练接收对齐的目标列表，检测框为零基连续 `xyxy`，目标标签为前景类别。训练模式返回非空标量损失映射；评估模式为每张图像返回 `boxes`、`labels` 与 `scores`。Faster R-CNN 的准确四项损失见[Faster R-CNN 工作原理](../concepts/how-faster-rcnn-works.zh-CN.md)。

VOC 元数据提供 20 个前景类别和背景，因此 `model.expected_num_classes` 必须是 21。受控选择时，应保持权重策略、清单标识、样本上限、随机种子、优化器和轮次不变，再按[实验管理](../guides/experiments.zh-CN.md)比较验证产物。
