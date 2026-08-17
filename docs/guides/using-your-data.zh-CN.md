# 使用自己的数据

把标注转换为 manifest 数据契约：RGB 图像张量、零基连续 xyxy 框、int64 类别、area、image ID 与 `iscrowd`。定义稳定的 train/valid/test 划分和类别顺序；训练前准备并预览完整数据，禁止使用 test 标签选模。
