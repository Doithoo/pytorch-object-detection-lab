# 添加内部模型

[English](adding-models.md) | [模型行为说明](../tutorial/03-faster-rcnn.zh-CN.md)

本指南面向需要加入不同检测器家族或受控对比模型的维护者，也说明显式外部 `module:function` 工厂。检查点不会序列化用户的可执行代码；外部模型会记录工厂路径，并在恢复时要求该路径仍可导入。

## 实现构造器

内置 detector 的 torchvision 专用构造逻辑放在 `src/object_detector/models/torchvision_models.py`：

```python
def build_detector(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> torch.nn.Module: ...
```

外部 detector 可以把 `model.factory` 设置为可导入的 `module:function`。工厂会收到 `num_classes`、`weights` 以及 `model.params` 下的所有键，并返回遵守同一图像列表/目标列表协议的 `torch.nn.Module`。项目会在训练模式校验 loss 映射，在评估模式校验 `boxes`、`labels`、`scores` 映射。

```yaml
model:
  name: my_detector
  factory: my_package.models:build_detector
  weights: none
  expected_num_classes: 4
  params:
    width: 32
```

外部工厂必须显式指定。检查点可以在不执行工厂代码的情况下安全读取，但预测、评估或续训会在重建模型前导入记录的路径。工厂缺失或发生变化时必须明确失败。

`num_classes` 包含背景类。`weights="none"` 必须传入 `weights=None` 和 `weights_backbone=None`，且不能访问网络。每个具名策略必须映射到一个固定的 torchvision 枚举，并提供可供预检查推导缓存文件名的 URL。项目自有参数固定后，才能把 `params` 作为构造器关键字传入。

返回模块必须遵守 torchvision 检测模式：

| 模式 | 输入 | 输出 |
|---|---|---|
| `train()` | `Tensor[3,H,W]` 图像列表、目标列表 | 非空且每项为有限标量张量的损失映射 |
| `eval()` | 仅图像列表 | 每张图像一个包含 `boxes`、`labels`、`scores` 的映射 |

不要在命令行或训练器中重新解释这套接口。仅依赖 checkpoint 的预测会用 `weights="none"` 重建注册模型，再加载保存状态，因此模型名称和参数含义必须稳定。

## 注册元数据

在 `src/object_detector/models/registry.py` 中加入一个 `ModelSpec`，包含稳定的小写名称、构造器、`two_stage` 或 `one_stage` 家族、事实化描述、项目维护的参数说明、输入注意事项、支持的策略，以及策略到骨干网络权重的映射。不要为每种参数组合创建新名称。

`model.params` 不能包含 `weights`、`weights_backbone` 或 `num_classes`。当前注册表会把其他键传给 torchvision；文档只承诺项目准备维护的键，并写清类型、默认值和效果。拼写错误必须在模型构造时失败，不能静默回退。

## 证明扩展有效

测试应覆盖注册顺序与元数据、相近名称提示、离线 `none` 构造、固定权重映射、错误或保留参数、一次合成训练前向和更新、评估输出、无需下载的检查点恢复，以及不构造模型的命令行发现。

```bash
uv run pytest tests/test_models.py tests/test_model_smoke.py tests/test_checkpoint.py tests/test_inference.py -q
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

只有参数确实组成合理对比时才增加 YAML 配置，然后执行 `show-config` 和 CPU dry run。构造成功、loss 有限或小规模运行都不能成为性能结论。同步更新双语[模型参考](../reference/model-zoo.zh-CN.md)、相关链接和必要的打包声明；如果扩展改变外部代码与项目的交互方式，还要更新架构决策记录。
