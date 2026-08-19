# 教程 05：指标、错误证据与推理

[English](05-evaluation-and-inference.md) | [教程索引](README.zh-CN.md)

前提是已经有第 04 章生成的项目检查点。数据集评估还需要与检查点身份标识
匹配的数据清单和源数据；单图或目录预测只需要检查点与本地图像。

## IoU 阈值与分数阈值回答不同问题

IoU 测量几何重叠。`0.5` 这样的匹配阈值判断同类别预测框与标注框是否重叠到足以匹配。
分数阈值判断模型置信度是否足以让预测进入序列化结果、可视化或错误分析。提高分数阈值
可能同时移除假阳性和真阳性，并不会改善剩余框的几何位置。

评估命令的 `--score-threshold` 只过滤序列化预测和渲染证据，不过滤传入 AP/AR 指标后端
的预测。检查点配置中的 `error_score_threshold` 和 `error_iou_threshold` 分别控制
错误分类所用的分数和 IoU 阈值。

## 把 AP 与 AR 理解为曲线汇总，而不是一次框判断

- `map_50_95` 对 IoU 0.50 到 0.95、步长 0.05 的平均精度（Average Precision）求平均，
  同时考察分类与更严格的定位。
- `map_50`、`map_75` 是固定 IoU 下的 AP。
- `mar_1`、`mar_10`、`mar_100` 是每张图最多保留 1、10、100 个检测时的平均召回率
  （Average Recall）。
- `per_class.csv` 为实际出现的前景类别报告 `map_50_95` 与 `mar_100`。

AP 汇总按分数排序的检测所形成的精确率-召回率曲线；AR 衡量给定检测数量上限时找回了
多少普通目标。它们都不能解释某张图为何失败，所以评估器同时写入汇总和逐图证据。指标
后端的负占位值会归一化为零，JSON/CSV 序列化结果保留六位小数。

## 困难目标不会变成普通错误

valid/test 划分保留 VOC 困难目标，并设置 `iscrowd=1`。普通目标计数不包括它们。
错误分析中，同类别预测若只在配置的 IoU 阈值上匹配困难目标，会标为 `ignored`，
而不是假阳性；困难目标也不会标为漏检。因此评估前丢掉 `iscrowd` 会改变报告含义。

## 做选择时先评估验证集

选择轮次、阈值、模型或超参数时，先评估 valid 划分：

```bash
uv run detect evaluate --checkpoint artifacts/first-detector/best.pt --split valid --output-dir artifacts/first-detector/evaluation-valid --device cpu
```

预期标准输出给出结果目录。若检查点的数据身份标识与当前准备数据不同，命令会拒绝
继续。它以 `weights=none` 重建模型、加载保存状态，并原子写入：

```text
evaluation.json
predictions.json
per_class.csv
errors.csv
visualizations/summary.png
visualizations/missed-*.png              存在漏检时
visualizations/false_positive-*.png      存在假阳性时
```

`evaluation.json` 记录指标、阈值、指标后端版本、数据划分、数据身份标识和检查点
SHA-256。`predictions.json` 是经过分数过滤的逐图输出。`errors.csv` 使用类别、分数、IoU
和框描述 `missed`、`false_positive`、`localization`、`ignored`。

除非显式使用 `--overwrite`，结果目录必须不存在或为空。保留旧证据前不要随手覆盖。

## 从表格回到图像

先看 `visualizations/summary.png`，再把排序后的漏检图、假阳性图与 `errors.csv` 并排阅读。
错误分析先保留达到错误分数阈值的预测，再按分数从高到低处理。每个预测只考虑同类别且
尚未匹配的普通目标；IoU 达到错误 IoU 阈值时，会匹配并消耗其中一个。否则，若与同类别
困难目标的重叠达标，则标为 `ignored`；若与尚未匹配的普通目标有正重叠但未达标，
则标为 `localization`；其余标为 `false_positive`。已经被消耗的普通目标不再是候选，因此
重复预测可能成为假阳性。所有预测处理完后，未匹配的普通目标标为 `missed`；困难
目标永远不会标为漏检。

绿色框是普通目标，橙色虚线框是困难框，蓝色框是预测。下面的图只用合成数据
演示图例，不是声称的模型结果：

![合成检测错误分析图](../assets/detection-error-analysis.png)

先用 CSV 找到案例，再用 PNG 建立假设。单个指标无法区分小目标漏检、类别混淆、定位偏差、
重复检测或背景误检。

## 不需要 YAML 或数据清单的检查点推理

对一张本地图像：

```bash
uv run detect predict --checkpoint artifacts/first-detector/best.pt --image docs/assets/detection-target-anatomy.png --output-dir artifacts/prediction --device cpu --score-threshold 0.5
```

预期生成 `artifacts/prediction/detection-target-anatomy.json` 和
`artifacts/prediction/detection-target-anatomy.png`。输入是合成教学图；让检测器在它上面
运行只能验证检查点恢复、推理与产物写入机制，不能证明检测质量。JSON 包含图像
尺寸、数据身份标识，以及达到分数阈值的全部检测。`--display-limit` 只限制分数过滤后 PNG
绘制的框数，不会截断 JSON 中的检测结果。

目录预测用 `--input-dir` 代替 `--image`，写入 `predictions.json` 和 `visualizations` 目录树。
无法读取的图片会记录为错误，其余 `.jpg`、`.jpeg`、`.png` 文件继续处理。预测会从
检查点恢复模型结构和类别顺序，以 `weights=none` 构造模型，不需要配置 YAML 或
VOC 文件。

## 严格分开验证集与测试集

用 valid 划分选择检查点和操作阈值。所有选择固定后，官方协议的最终报告才可以
评估 test 划分：

```bash
uv run detect evaluate --checkpoint artifacts/first-detector/best.pt --split test --output-dir artifacts/first-detector/evaluation-test --device cpu
```

对于有界学习检查点，这仍然只评估配置中的测试样本上限，不是完整 VOC 成绩。反复
查看测试集后再修改模型，相当于把测试集变成第二个验证集。

## 比较兼容运行但不丢失上下文

两个运行使用同一份不可变数据身份标识后，执行：

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

预期标准输出标出指标和共享的数据身份标识，按每个运行的最佳指标记录排序，并列出影响
实验语义的配置差异。报告会有意排除操作性字段 `run_name`、`output_dir`、`device` 和
`data.num_workers`。损失指标按从低到高排列，其他指标按从高到低排列。缺少产物或列、
数值不是有限值、数据身份标识不同或输出 CSV 已存在都会导致失败。这个排名只比较记录中
的运行，不能证明某项配置普遍更好。

## 常见失败边界

- 检查点与准备数据的身份标识不同：评估停止；预测仍可运行，因为它不声称数据集指标。
- `evaluation-*` 已有文件：选择新目录，或在保留证据后明确覆盖。
- 提高 CLI `--score-threshold` 后指标不变：这是预期行为，因为 AP/AR 指标后端不使用它。
- 困难目标匹配变成普通假阳性：检查标注路径是否保留 `iscrowd`。
- JSON 中的检测比 PNG 多：`--display-limit` 只控制可视化。
- 把有界测试数值写成完整 VOC 证据：无论数值多少，证据范围都不成立。

回到[学习路径](learning-path.zh-CN.md)进行全流程检查，或使用
[指标参考](../reference/metrics.zh-CN.md)核对报告字段。
