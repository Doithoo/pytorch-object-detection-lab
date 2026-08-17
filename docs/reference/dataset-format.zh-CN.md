# 数据集格式

准备阶段要求官方 `VOCdevkit/VOC2007` 目录及 train、val、test 划分文件，并生成 CSV manifests 与 metadata，其中包含划分 hash、有序类别、源根目录、坐标约定和组合 identity。

VOC 框是一基且端点包含。转换为零基连续 xyxy 的公式是 `(xmin-1, ymin-1, xmax, ymax)`，最大角为排他的像素边界。训练移除 difficult 目标；验证/测试以 `iscrowd=1` 保留，指标与错误分析忽略其匹配，普通目标计数也排除它们。空 target 使用 `[0,4]` 与 `[0]` 张量。
