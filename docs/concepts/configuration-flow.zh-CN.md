# 配置流程与职责

[English](configuration-flow.md) | [字段参考](../reference/config-reference.zh-CN.md)

本概念页帮助读者追踪某个值为何被选中，或某个命令行参数为何没有进入保存配置。内容对应 `src/object_detector/config.py` 和 `src/object_detector/cli.py` 的实现。

## 从文本到严格 `AppConfig`

```text
数据类默认值
  -> 递归合并已知 YAML 字段
  -> 按命令顺序应用重复的 --set 点分路径
  -> 构造数据类和 Path 值
  -> 校验类型、有限性、范围和选项
  -> 应用命令专用运行时参数
  -> 训练预检查或命令处理器
```

配置没有环境变量层。`TORCH_HOME` 等环境可以影响 torch 缓存位置，但不会成为 `AppConfig` 的值来源。

YAML 根节点必须是映射。普通区域和字段都是封闭结构：未知键会按点分路径报错，标量也不能替代整个区域。`model.params` 是模型专用映射；所选模型的注册表条目会在构造前拒绝保留键、拼写错误和未维护键。

每个 `--set KEY VALUE` 的值按 YAML 解析，因此 `true`、`3`、`0.5`、`null`、列表和映射都会成为有类型的值。PyYAML 会把 `none` 保留为字符串，而 `null` 和 `~` 会变成 Python `None`。合并后，构造阶段把数据与输出路径转成 `Path`；校验会拒绝把布尔值当整数或数值、非有限数值、错误范围或选项，以及空标识。

## 检查值与来源

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set train.epochs 3 --set model.params.min_size 320 --set run_name trace-me
```

预期输出是完整 YAML，末尾包含 `sources` 映射。每个普通叶字段标记为 `default`、`yaml` 或 `cli`。`model.params` 为空时自身是叶字段；配置嵌套参数后，对应路径会标记为 YAML 或命令行来源。该命令只解析配置，不读取清单、不选择设备、不检查输出可写性、不检查权重缓存、不构造模型，也不写产物。

保存的 `artifacts/<运行名>/config.yaml` 包含解析值，但不包含 `sources` 报告。`run.yaml` 另行记录环境、解析设备、随机种子、Git 修订、类别与清单标识。

## 训练职责

`detect train` 从 `--config` 和重复 `--set` 加载 `AppConfig`。若提供命令级 `--device`，`cli._train` 会在配置校验后通过 `dataclasses.replace` 替换设备。最终替换不会出现在 `show-config` 来源跟踪里，但结果会写入运行的 `config.yaml`。

随后 `training.run_training` 加载 `dataset.yaml`，调用 `preflight.validate_training_request`，按 CUDA、MPS、CPU 的顺序解析 `auto`，初始化随机数，构造注册模型并建立数据集与加载器。预检查核对必须的清单文件、类别数、加速器可用性、输出目标可写性，以及具名骨干网络权重是否已缓存。缺少权重只产生提示，不是错误；稍后的 torchvision 模型构造可能下载。

`--dry-run` 和 `--resume` 改变编排流程，不属于配置结构。试运行只消费一个训练批次，不写正常运行产物。续训指向有结构版本的检查点，并与解析配置核对。

## 仅运行时命令参数

| 命令 | 不属于 `AppConfig` 的运行时输入 |
|---|---|
| `prepare-data` | `--data-dir`、`--manifest-dir`、`--allow-nonstandard-counts`（两个路径与配置含义相同，但独立解析） |
| `inspect-data` | `--manifest-dir`、`--data-dir`、`--split`、`--limit` |
| `list-models` | 无；只读取注册元数据 |
| `model-info` | 位置参数模型 `name` |
| `compare-runs` | 运行目录、`--metric`、可选 `--output` |
| `train` | `--config`、重复 `--set`、`--dry-run`、`--resume`、最终 `--device` |
| `evaluate` | `--checkpoint`、`--split`、`--output-dir`、`--device`、`--score-threshold`、`--overwrite` |
| `predict` | `--checkpoint`、`--image`/`--input-dir` 二选一、`--output-dir`、`--device`、`--score-threshold`、`--display-limit`、`--overwrite` |

评估没有 `--config` 或 `--set`：它会校验并加载检查点保存的解析配置，用于数据路径、样本上限、错误阈值、最大检测数、批大小和工作进程数。命令行 `--score-threshold` 是独立的序列化与可视化阈值，不会替换某个 `AppConfig` 中保存的 `evaluation.score_threshold`。预测只使用检查点模型、类别、预处理和运行时输入。

## 每条命令检查什么

`show-config` 检查文本解析，`train --dry-run` 检查一次完整的数据与模型更新，小规模正常运行检查输出创建，评估检查 checkpoint 与匹配清单上的指标。每条命令覆盖流程中的不同部分。

未知字段或错误类型在模型构造前失败。预检查问题会在创建正常运行目录前失败。新训练拒绝已有运行目录。检查点和文本产物分别原子写入；评估和目录预测则暂存并发布完整输出目录。接下来可阅读[代码导览](code-tour.zh-CN.md)了解模块职责，或查看[配置参考](../reference/config-reference.zh-CN.md)中的全部叶字段。
