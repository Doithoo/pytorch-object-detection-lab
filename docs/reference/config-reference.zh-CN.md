# 配置参考

[English](config-reference.md) | [配置流程](../concepts/configuration-flow.zh-CN.md)

本参考面向 YAML 配置和 `--set` 覆盖的作者。`AppConfig` 使用严格结构：未知区域或字段会失败，映射不能替代标量字段，错误值也不会静默回退。

## 解析顺序与来源跟踪

解析顺序是类型化数据类默认值、可选 YAML 文件、最后按命令顺序应用重复的 `--set KEY VALUE`。每个覆盖值都由 `yaml.safe_load` 解析，不会保留为原始文本。例如，`true` 变为布尔值，`3` 为整数，`0.5` 为浮点数，`null` 为 Python `None`，`[320, 640]` 为列表。`none` 会保留为 `model.weights` 需要的字符串；`null` 与 `~` 才是空值。

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set train.epochs 3 --set train.amp true --set run_name experiment-a
```

`show-config` 会打印完整序列化配置和排序后的 `sources` 映射。每个叶字段标记为 `default`、`yaml` 或 `cli`。命令不会读取数据、构造模型、访问网络或写产物。`train --device ...` 在加载 `AppConfig` 后才应用，因此不会出现在 `show-config` 的来源跟踪里。

## 数据字段

| 字段 | 默认值 | 接受值 | 重要关系 |
|---|---:|---|---|
| `data.name` | `voc2007` | 字符串，必须为 `voc2007` | 尚未实现其他提供器。 |
| `data.data_dir` | `data/raw` | 可转换为路径的字符串 | 运行时会追加 `dataset.yaml.dataset_root`；评估使用检查点配置保存的路径。 |
| `data.manifest_dir` | `data/manifests` | 可转换为路径的字符串 | 必须包含 `train.csv`、`valid.csv`、`test.csv`、`dataset.yaml`；评估与续训时标识必须匹配检查点。 |
| `data.num_workers` | `0` | >= 0 的整数，拒绝布尔值 | 可在续训时改变；用于训练和评估数据加载器。 |
| `data.horizontal_flip` | `0.5` | [0, 1] 内有限数值 | 只用于训练；同步翻转图像和检测框。 |
| `data.max_train_samples` | `null` | `null` 或 >= 1 的整数 | 同时限制训练和评估选择 train 划分时使用的前若干行；改变训练语义，也不能再声明完整划分结果。 |
| `data.max_valid_samples` | `null` | `null` 或 >= 1 的整数 | 限制训练和评估使用的验证样本。 |
| `data.max_test_samples` | `null` | `null` 或 >= 1 的整数 | 限制测试评估；少量样本分数不是完整 VOC 结果。 |

## 模型字段

| 字段 | 默认值 | 接受值 | 重要关系 |
|---|---:|---|---|
| `model.name` | `fasterrcnn_mobilenet_v3_large_320_fpn` | 非空字符串；设置 `model.factory` 时不要求注册表成员 | 用 `detect list-models` 发现内置名称；续训标识要求名称和工厂设置相同。 |
| `model.factory` | `null` | `null` 或 `module:function` | 显式外部检测器工厂。工厂会收到 `num_classes`、`weights` 和 `model.params`；恢复 checkpoint 时会导入记录的路径。 |
| `model.weights` | `none` | `none` 或 `imagenet1k_v1` | `none` 离线；`imagenet1k_v1` 只请求固定骨干网络权重，可能需要缓存或网络。检查点恢复始终用 `none` 重建。 |
| `model.expected_num_classes` | `21` | >= 2 的整数，拒绝布尔值 | 预检查要求等于背景加清单类别；VOC 为 21。模型实际使用元数据推导的数量。 |
| `model.params` | `{}` | 映射 | 作为 torchvision 构造关键字传入；`weights`、`weights_backbone`、`num_classes` 为保留项。见[模型目录](model-zoo.zh-CN.md)。 |

## 训练字段

| 字段 | 默认值 | 接受值 | 重要关系 |
|---|---:|---|---|
| `train.epochs` | `2` | >= 1 的整数 | 续训时只能提高到大于已保存轮次。 |
| `train.batch_size` | `2` | >= 1 的整数 | 用于训练和检查点评估；属于续训语义。 |
| `train.lr` | `0.005` | > 0 的有限数值 | 用于 SGD 或 AdamW。 |
| `train.momentum` | `0.9` | >= 0 的有限数值 | 仅 SGD 使用，但在 AdamW 配置中仍属于续训标识。 |
| `train.weight_decay` | `0.0005` | >= 0 的有限数值 | 两种优化器都使用。 |
| `train.optimizer` | `sgd` | `sgd` 或 `adamw` | SGD 使用动量，AdamW 不使用。 |
| `train.scheduler` | `none` | `none` 或 `step` | `step` 固定为 `StepLR(step_size=8, gamma=0.1)`，其状态写入检查点。 |
| `train.seed` | `42` | 闭区间 `0` 到 `4294967295` 内的整数，拒绝布尔值 | 初始化 Python、NumPy、torch 和打乱生成器；越界时报告 `train.seed must be between 0 and 4294967295`；随机状态写入检查点。 |
| `train.amp` | `false` | 只能是 YAML 布尔值 | 梯度缩放只在 CUDA 启用，自动类型转换只用于 CPU/CUDA，因此解析为 MPS 时使用全精度。只有配置的 `device` 准确为 `mps` 时预检查才提示；`auto` 解析为 MPS 时不会提示。 |
| `train.grad_clip` | `0.0` | >= 0 的有限数值 | `0` 关闭裁剪；正值在反向传播后执行全局范数裁剪。 |
| `train.best_metric` | `map_50_95` | `map_50_95` 或 `voc_map_50_11` | `best.pt` 只在选定验证指标严格提高时更新。 |

## 评估和顶层字段

| 字段 | 默认值 | 接受值 | 重要关系 |
|---|---:|---|---|
| `evaluation.score_threshold` | `0.05` | [0, 1] 内有限数值 | 保存在配置中，但 `evaluate` 有运行时 `--score-threshold`；AP 始终接收原始预测。 |
| `evaluation.error_score_threshold` | `0.5` | [0, 1] 内有限数值 | 检查点评估用它做图像级错误分析。 |
| `evaluation.error_iou_threshold` | `0.5` | [0, 1] 内有限数值 | 同类别贪心匹配的阈值。 |
| `evaluation.max_detections` | `100` | 整数且只能为 `100` | 对应支持的 COCO 风格 AR 上限；当前拒绝其他正数。 |
| `device` | `auto` | 配置中为非空字符串；训练预检查支持 `auto`、`cpu`、`mps` 或以 `cuda` 开头的字符串 | `auto` 依次选择 CUDA、MPS、CPU；评估与预测的设备为运行时参数。 |
| `output_dir` | `artifacts` | 可转换为路径的字符串 | 预检查要求存在的上级目录可写。运行目录为 `output_dir/run_name`；名称为空时使用 `run`。 |
| `run_name` | `null` | `null` 或非空字符串 | 新运行拒绝已存在的目录；续训时可作为操作字段改变。 |

表中每个字段都可来自默认值、YAML 或 `--set`，来源不会改变校验。`--dry-run`、`--resume`、`--overwrite`、`--split` 和预测输入等运行时命令参数不是 `AppConfig` 叶字段。命令职责见[配置流程](../concepts/configuration-flow.zh-CN.md)，续训规则见[检查点结构](checkpoint-schema.zh-CN.md)。
