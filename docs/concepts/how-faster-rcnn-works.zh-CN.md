# Faster R-CNN 如何工作

backbone 提取特征，FPN 提供多个分辨率；RPN 为 anchors 评分并提出区域；ROI Align 对每个区域采样，ROI heads 完成前景/背景分类与框校正。训练联合优化 proposal 和 ROI 损失；推理时 torchvision 在返回最终检测前完成评分与抑制。
