# 指标

`map_50_95` 在 IoU 0.50:0.05:0.95 上平均 AP；`map_50` 与 `map_75` 是固定 IoU 的 AP。`mar_1`、`mar_10`、`mar_100` 限制每图检测数。逐类输出包含 `map_50_95` 与 `mar_100`。后端负 sentinel 转为数值零；内存值不舍入，JSON/CSV 使用六位小数。

AP 接收未应用显示阈值的后端预测。错误分析另以 `error_score_threshold` 过滤，并在 `error_iou_threshold` 上对同类普通目标贪心匹配；只匹配 difficult 的预测会被忽略。用 validation 指标做选择，将 VOC test 保留给一次最终报告。有界或合成分数不是完整 VOC benchmark。
