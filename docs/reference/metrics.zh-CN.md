# 指标与评估产物

[English](metrics.md) | [评估教程](../tutorial/05-evaluation-and-inference.zh-CN.md)

本参考用于阅读训练历史、AP/AR 报告、错误行和评估图。指标是没有单位的小数，不是百分数。[Kaggle 训练记录](../recorded-run/README.zh-CN.md)提供一个具体案例；小规模运行和未经执行的配置不能与它直接比较。

## 训练 `metrics.csv`

每个轮次完成后原子重写全部行。列按首次出现顺序排列：

| 列 | 含义 |
|---|---|
| `epoch` | 从 1 开始的已完成轮次 |
| `loss_total` | 模型返回损失之和按样本加权的轮次均值 |
| 模型损失名称 | 训练模式返回的每个键按样本加权的轮次均值 |
| `valid_map_50_95`、`valid_map_50`、`valid_map_75` | 下文定义的验证 AP |
| `valid_mar_1`、`valid_mar_10`、`valid_mar_100` | 下文定义的验证 AR |
| `valid_image_count` | 处理的验证图像数 |
| `valid_target_count` | 普通验证目标数，不含 `iscrowd=1` 的困难目标 |
| `valid_prediction_count` | 项目分数过滤前模型返回的全部检测数 |

Faster R-CNN 固定返回 `loss_classifier`、`loss_box_reg`、`loss_objectness` 和 `loss_rpn_box_reg`。其他注册检测器家族可以返回不同损失键；训练器按收到的映射记录。验证会把模型返回预测直接交给指标后端，不应用 `evaluation.score_threshold`。

## 汇总 AP 与 AR

后端是由 `pycocotools` 支持的 `torchmetrics.detection.MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)`。

| 字段 | 规则 |
|---|---|
| `map_50_95` | IoU 0.50、0.55、...、0.95 上的平均 AP |
| `map_50` | IoU 0.50 时的 AP |
| `map_75` | IoU 0.75 时的 AP |
| `mar_1` | 每图最多 1 个检测时的平均 AR |
| `mar_10` | 每图最多 10 个检测时的平均 AR |
| `mar_100` | 每图最多 100 个检测时的平均 AR |
| `per_class` | 后端中出现的非背景类别，含 `class_id`、`class_name`、`map_50_95`、`mar_100` |
| `image_count` | 交给后端的图像数 |
| `target_count` | 普通非拥挤目标数 |
| `prediction_count` | 交给后端的检测数 |

torchmetrics 的负哨兵值，包括未定义或缺失类别输出，会被截到 `0.0`。除此之外不做归一化、缩放、平滑或百分比转换。指标内容、序列化预测数值，以及错误和逐类别行中的数值会在适用处舍入到六位小数。`evaluation.json` 中的阈值字段直接写入配置浮点数，不经过该舍入辅助函数。内存训练值和检查点历史保留普通 Python 浮点精度。

VOC 困难目标以 `iscrowd=1` 进入评估，不增加普通 `target_count`。指标后端应用自身的拥挤处理；独立错误分析器把符合条件、只匹配困难目标的预测记为 `ignored`，且永远不会把困难目标记为漏检。

## `evaluation.json`

命令行评估会写入：

| 键 | 值 |
|---|---|
| `metrics` | 上述汇总字段，包括嵌套 `per_class` |
| `backend_versions` | 已安装 `torchmetrics` 和 `pycocotools` 的版本字符串 |
| `score_threshold` | 序列化预测与图像渲染的命令行阈值，默认 0.05 |
| `error_score_threshold` | 检查点配置中的错误候选阈值，默认 0.5 |
| `error_iou_threshold` | 检查点配置中的同类别匹配阈值，默认 0.5 |
| `max_detections` | 支持的指标上限，固定为 100 |
| `manifest_identity` | 准备数据的清单标识 |
| `checkpoint_sha256` | 完整检查点文件的 SHA-256 |
| `split` | `train`、`valid` 或 `test` |

修改命令行 `--score-threshold` 会改变 `predictions.json` 和渲染的蓝框，不会改变 AP/AR 或 `prediction_count`。模型内部的后处理，例如 `box_score_thresh` 或 SSDLite `score_thresh`，发生在模型内部，因此会改变进入后端的预测。

## CSV 与 JSON 输出

`per_class.csv` 的列依次为 `class_id,class_name,map_50_95,mar_100`。`predictions.json` 是 `{image_id, predictions}` 数组；每条保留预测包含 `box`（四个舍入后的 `xyxy` 值）、`class_id`、`class_name` 和 `score`。

`errors.csv` 的列依次为 `image_id,kind,class_name,score,iou,box`。`box` 在 CSV 单元格内是 JSON 数组；漏检行的分数为空。错误候选为分数不低于 `error_score_threshold` 的预测，按分数降序排列，同分时保留原顺序，再与尚未匹配的同类别普通目标贪心匹配。

| `kind` | 具体条件 |
|---|---|
| `ignored` | 没有普通匹配，但同类别困难目标 IoU >= 错误 IoU 阈值 |
| `localization` | 没有达到阈值的匹配，但与未匹配的同类别普通目标有正 IoU |
| `false_positive` | 普通和困难同类别重叠都不符合条件，且最佳普通 IoU 为 0 |
| `missed` | 全部候选处理后普通目标仍未匹配；`score` 为空，IoU 为 0 |

`visualizations/summary.png` 总是渲染第一个评估样本。系统按错误数量选择最多五个 `missed` 图像标识和最多五个 `false_positive` 图像标识，重新加载并写出排序 PNG；同数时按图像标识排序。绿色框是普通目标，橙色虚线框是困难目标，蓝色框是序列化预测。

全部评估文件先写入暂存目录，再原子发布输出目录。非空目标目录会失败，除非明确使用 `--overwrite`。实验选择使用验证产物，测试集只用于最终固定决策。
