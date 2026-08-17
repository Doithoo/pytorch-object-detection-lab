# 03 - Faster R-CNN

backbone 与 FPN 产生多尺度特征；区域建议网络提出类别无关区域，ROI heads 完成分类与最终框回归。torchvision 负责这些内部结构，本仓库负责类别数、权重策略、数据契约与产物。

运行：`uv run python examples/03_detector_losses.py`

预期：输出具名的分类、框回归损失及总损失。
