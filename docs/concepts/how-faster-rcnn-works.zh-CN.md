# Faster R-CNN 如何工作

backbone 提取特征，FPN 提供多个分辨率；RPN 为 anchors 评分并提出区域；ROI Align 对每个区域采样，ROI heads 完成前景/背景分类与框校正；非极大值抑制去除高度重叠的重复预测。

接口会随模式改变。`model.train()` 需要图像和 targets，返回 RPN 与 ROI loss；`model.eval()` 只需要图像，返回 boxes、labels 和 scores。因此用训练模式做评估不仅影响统计量，还会直接改变 API。

torchvision 模型内部 transform 负责归一化、缩放和组 batch。checkpoint 会记录这份预处理契约，避免推理时静默采用另一套约定。

运行：`uv run python examples/03_model_contract.py`

预期：训练模式输出具名 losses，评估模式输出 `boxes`、`labels` 和 `scores`。
