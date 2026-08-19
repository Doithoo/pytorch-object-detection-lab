# 运行受控实验

[English](experiments.md) | [训练教程](../tutorial/04-training.zh-CN.md)

本指南用于在不丢失数据来源、也不把测试集用于反复调参的前提下比较配方。仓库已经有一次[完整 VOC 实测运行](../recorded-run/README.zh-CN.md)，但这项由验证集选择的单次结果只证明其精确配方，不是排行榜，也不能替代受控对比。

## 固定证据边界

准备一次数据，记录打印的标识，并在训练前检查源数据：

```bash
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
```

源图像字节、XML、类别、坐标或划分成员只要发生变化，重新准备就会产生不同标识。不能把新运行当作使用旧数据的实验来比较；`compare-runs` 会拒绝不同标识。

## 提出一个假设

从 `configs/learning_minimal.yaml` 开始，指定唯一运行名，并只改变一个语义字段。先检查类型化解析结果：

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set run_name baseline
uv run detect show-config --config configs/learning_minimal.yaml --set run_name flip-off --set data.horizontal_flip 0.0
```

创建产物前，分别用试运行证明两条路径：

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name baseline --dry-run --device cpu
uv run detect train --config configs/learning_minimal.yaml --set run_name flip-off --set data.horizontal_flip 0.0 --dry-run --device cpu
```

试运行会执行一次更新并打印有限损失项，但不会写检查点，也不能证明学习质量。契约通过后，去掉 `--dry-run` 才开始正式运行。新运行会拒绝任何已存在的运行目录，因此不要复用名称。

## 把每次运行作为一个整体保存

完成的训练目录包含：

| 产物 | 证据 |
|---|---|
| `config.yaml` | 完整解析配方，而不是简单复制输入 YAML |
| `run.yaml` | Python、框架、平台、设备、随机种子、Git 修订、清单标识、划分哈希和有序类别 |
| `metrics.csv` | 每个完成轮次一行，包含训练损失和验证指标 |
| `best.pt` | 最近一次严格提升验证 `map_50_95` 的检查点 |
| `last.pt` | 最近完成轮次和续训状态 |

这些文件必须一起保留。续训只能用于延长同一实验；它会恢复优化器、可选调度器、指标历史和随机数状态，所有后代还会继承新训练生成的 `lineage_id`。每个续训检查点都必须记录有限的配置验证指标，并令 `best_metric` 等于完整历史最大值。从 `last.pt` 续训到另一个空运行目录时，必须提供同 lineage 的同级 `best.pt`；其严格历史最大值和语义标识通过验证后才会带入新运行。只有目标是新的空运行目录，或原目录缺少 `last.pt` 且使用原始精确路径时，才可直接从 `best.pt` 续训；原位已有 `last.pt` 时必须使用它。只有 `train.epochs`、`data.num_workers`、`device`、`output_dir` 与 `run_name` 可以不同，而且请求轮次必须大于已保存轮次。其他变化必须开始新运行。

## 用验证集选择，最后只报告一次测试集

使用 `valid_map_50_95` 或其他已记录的验证列比较兼容运行：

```bash
uv run detect compare-runs artifacts/baseline artifacts/flip-off --metric valid_map_50_95 --output artifacts/flip-comparison.csv
```

命令会对每次运行的最佳行排序，显示语义配置差异，并忽略操作性字段 `run_name`、`output_dir`、`device` 和 `data.num_workers`。差异值遵循排名后的行顺序，并标注为 `run=value`。名称包含 `loss` 的指标按低值优先，其他指标按高值优先。它不会替用户判断某个配方在所有场景下都更好。

用验证集选择配方和检查点。固定分数与错误分析阈值后，再评估一次保留测试集：

```bash
uv run detect evaluate --checkpoint artifacts/baseline/best.pt --split test --output-dir artifacts/baseline/evaluation-test --device cpu
```

如果配置包含样本上限，测试结果仍只是有界证据，不是完整 VOC 结果。反复查看测试集再改配方，会把测试集变成另一个验证集。阅读[指标参考](../reference/metrics.zh-CN.md)解释产物；发布任何完整 VOC 结论前，还要满足[记录运行门槛](../recorded-run/README.zh-CN.md)。
