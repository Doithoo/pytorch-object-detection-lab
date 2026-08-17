# 00 - 检测基础

每张图像是 `[3,H,W]` 浮点张量。target 字典中的 `boxes` 是 `[N,4]` 浮点 xyxy 坐标，`labels` 是 int64 类别 ID。同一 batch 的图像尺寸和目标数可以不同，因此检测使用列表而不是堆叠张量。

运行：`uv run python examples/02_detection_batch.py`

预期：输出两个不同图像尺寸，以及目标数 `1` 和 `2`。
