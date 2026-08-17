# 添加数据集

添加校验源标注的 parser，以及生成确定性划分 manifest 与 metadata identity 的准备步骤。实现与 `VocDetectionDataset` 相同的 `(image, target)` 接口和 source image ID；复用同步变换与 `detection_collate`。为无目标、坐标、difficult/crowd、hash 和预览添加离线测试。
