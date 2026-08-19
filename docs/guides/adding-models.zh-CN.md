# 添加内部模型

[English](adding-models.md) | [模型契约教程](../tutorial/03-faster-rcnn.zh-CN.md)

本指南面向需要加入不同检测器家族或受控对比模型的维护者。版本 0.1 没有稳定的外部插件接口，也不会加载任意 `module:function` 工厂。添加模型意味着修改并测试仓库；检查点不会序列化用户的可执行代码。

## 实现构造器契约

torchvision 专用构造逻辑放在 `src/object_detector/models/torchvision_models.py`：

```python
def build_detector(
    num_classes: int,
    weights: str,
    params: Mapping[str, object],
) -> torch.nn.Module: ...
```

`num_classes` 包含背景类。`weights="none"` 必须传入 `weights=None` 和 `weights_backbone=None`，且不能访问网络。每个具名策略必须映射到一个固定的 torchvision 枚举，并提供可供预检查推导缓存文件名的 URL。项目自有参数固定后，才能把 `params` 作为构造器关键字传入。

返回模块必须遵守 torchvision 检测模式：

| 模式 | 输入 | 输出 |
|---|---|---|
| `train()` | `Tensor[3,H,W]` 图像列表、目标列表 | 非空且每项为有限标量张量的损失映射 |
| `eval()` | 仅图像列表 | 每张图像一个包含 `boxes`、`labels`、`scores` 的映射 |

不要在命令行或训练器中改写这份契约。仅依赖检查点的预测路径会用 `weights="none"` 重建注册架构，再加载保存的状态，因此模型名称和参数语义必须稳定。

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

只有参数确实组成合理对比时才增加 YAML 配方，然后执行 `show-config` 和 CPU 试运行。构造成功、损失有限或有界运行都不能成为性能结论。同步更新双语[模型目录](../reference/model-zoo.zh-CN.md)、相关链接、必要的打包声明；若扩展改变外部代码边界，还要更新架构决策记录。
