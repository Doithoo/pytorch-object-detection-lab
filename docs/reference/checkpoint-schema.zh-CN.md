# 检查点结构版本 1

[English](checkpoint-schema.md) | [指标结构](metrics.zh-CN.md)

本参考面向检查点使用者、续训故障排查和产物审计。项目检查点可以独立完成预测，但评估仍需要与之匹配的已准备标注数据。当前只接受结构版本 `1`。

结构版本 `1` 已为首次 `0.1` 发布冻结。更早的本地开发检查点虽然也写有 `schema_version: 1`，但若缺少 `lineage_id` 或 `run_metadata.cuda_device_count`，则有意不兼容；应重新生成，而不是放宽校验。本项目没有需要迁移的已发布检查点规则。

## 顶层映射

| 字段 | 类型 | 职责 |
|---|---|---|
| `schema_version` | 整数 `1` | 选择对应的兼容规则。 |
| `lineage_id` | 非空字符串 | 新训练生成、所有续训后代继承的稳定标识。 |
| `config` | 映射 | 完整解析后的 `AppConfig`，包含路径、上限、模型、训练、评估、设备、输出与运行名。 |
| `model` | 映射 | `name` 注册表名称或用户名称，可选的 `factory` `module:function`，以及重建架构所需的构造器 `params`。 |
| `weight_policy` | 字符串 | 训练来源（`none` 或 `imagenet1k_v1`）；恢复时不会重新下载。 |
| `class_names` | 非空有序序列 | `background` 在首位，随后是前景名称；用于恢复标签含义。 |
| `preprocessing` | 严格映射 | 输入处理所有权和表示，见下文。 |
| `manifest_identity` | 非空 SHA-256 字符串 | 把模型状态绑定到准备后的数据内容与协议。 |
| `split_hashes` | 映射 | 用于来源追踪的 `train`、`valid`、`test` 内容哈希。 |
| `model_state` | 状态映射 | 重建架构后加载的模型张量。 |
| `optimizer_state` | 状态映射 | 续训所需的优化器参数组和张量状态。 |
| `scheduler_state` | 映射或 `null` | 配置步进调度器时保存其状态，否则为空。 |
| `epoch` | 正整数 | 该内容最近完成的轮次。 |
| `best_metric` | 有限数值 | 截至该轮次记录的 `valid_<config.train.best_metric>` 最大值。 |
| `metric_history` | 映射序列 | 截至该轮次的全部 `metrics.csv` 行。 |
| `run_metadata` | 映射 | 环境和运行标识快照。 |
| `rng_state` | 映射 | 续训所需的 Python、NumPy、torch、CUDA 和数据加载器生成器状态。 |

`model` 包含 `name`、`factory`（内置模型为 null，外部模型为显式 `module:function` 路径）和 `params`。检查点评估与预测用保存的名称和参数、类别数以及 `weights="none"` 调用注册构造器或外部工厂，再加载 `model_state`。预测不需要访问模型权重网络，也不需要输入 YAML。外部工厂代码只在重建模型时导入；`load_checkpoint` 不会执行它。

## 严格预处理规则

```yaml
resize_owner: torchvision_model
input_range: [0.0, 1.0]
color_space: RGB
```

该映射必须完全一致。缺少键、修改值或增加键都会使版本 1 校验失败。图像是 `[0,1]` 范围的 RGB 浮点张量；注册的 torchvision 模型负责归一化、缩放和内部批处理。

## 运行与随机数元数据

`run_metadata` 包含 `python`、`torch`、`torchvision`、`platform`、解析后的 `device`、非负 `cuda_device_count` 和 `seed`；无法读取时 `git_revision` 为空，否则是提交字符串。CUDA 设备始终以 `cuda:1` 这类显式索引保存；隐式 CUDA 请求会在保存前用当前设备索引规范化，检查点元数据中的裸 `cuda` 无效。训练还会把 `manifest_identity`、`split_hashes` 和有序 `class_names` 放入该映射，因此无需连接其他文件也能读取环境记录。

`rng_state` 包含 Python 状态元组；NumPy 的 `bit_generator`、整数 `state` 序列、`position`、`has_gauss` 和 `cached_gaussian`；torch CPU 状态张量；以及训练数据加载器的 `loader_generator` 张量。CUDA 序列必须恰好包含 `run_metadata.cuda_device_count` 个非空一维字节张量；CPU 和 MPS 运行记录计数 `0` 和空序列。CUDA 检查点记录正数计数，且从 `run_metadata.device` 解析出的显式来源索引必须落在该序列中。历史最佳与恢复检查点各自独立验证此拓扑，因此 CUDA 历史最佳可以与后续 CPU 检查点组成合法链。当前设备为 CUDA 且可用时，续训只把保存的来源设备状态映射到解析后的当前 CUDA 设备；否则跳过 CUDA 恢复。

## 安全与原子加载

`save_checkpoint` 在目标目录写入唯一命名的临时文件，再通过 `os.replace` 发布；保存失败会删除临时文件。`load_checkpoint` 调用 `torch.load(..., weights_only=True)`，拒绝不在张量与容器允许列表中的 pickle 全局对象，随后要求顶层为映射且结构版本恰好为 `1`。不要为了加载不受信任文件而改用 `weights_only=False`。

```python
from pathlib import Path
from object_detector.training.checkpoint import load_checkpoint

checkpoint = load_checkpoint(Path("artifacts/run/best.pt"))
print(checkpoint["schema_version"], checkpoint["epoch"])
```

## 续训标识与允许变化

续训要求模型名、有序类别、清单标识、预处理，以及除下列项目外的所有解析配置完全一致：

| 可以变化 | 规则 |
|---|---|
| `train.epochs` | 必须大于已保存 `epoch` |
| `data.num_workers` | 数据加载操作设置 |
| `device` | 执行操作设置 |
| `output_dir` | 配置标识允许变化，但仍需通过目标目录安全检查 |
| `run_name` | 配置标识允许变化，但仍需通过目标目录安全检查 |

续训时不能改变批大小、优化器、调度器、学习率、增强、样本上限、随机种子、混合精度、模型参数、权重策略或评估设置。原位续训必须使用已有 `last.pt`；只要它存在，`best.pt` 或旧副本就会被拒绝。若 `last.pt` 缺失，只允许该运行目录中路径完全匹配的 `best.pt` 原位恢复；其他检查点必须使用新的空运行目录。不同的目标运行目录必须为空。跨目录从 `last.pt` 续训还要求同级 `best.pt` 具有相同 `lineage_id`，且模型、类别、预处理、清单与划分哈希语义完全一致；跨目录从 `best.pt` 续训则直接使用该检查点。允许变化的续训设置与执行环境可能不同，因此无需逐项相等比较完整配置和运行元数据快照。每个可续训检查点都要求配置的验证指标历史值为有限且非布尔的数值，并要求 `best_metric` 等于完整历史最大值；允许后续轮次与最大值持平。历史 `best.pt` 还要求末行值等于 `best_metric`，并严格大于所有更早值。验证通过后，该 payload 会原子发布为新运行的 `best.pt`，且不会改写其源轮次、解析配置或运行来源信息。不同实验应创建新运行，不能修改检查点字段。

评估还会加载保存配置，要求当前 `dataset.yaml.identity` 等于 `manifest_identity`，记录 checkpoint SHA-256，并写入新的输出目录。预测只需要 checkpoint 和输入图像。具体错误与处理方式见[排错指南](../guides/troubleshooting.zh-CN.md)。
