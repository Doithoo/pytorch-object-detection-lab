# 为 YOLO 导出已准备的数据

[English](using-yolo.md) | [数据集规则](../reference/dataset-format.zh-CN.md)

项目可以把已准备的 VOC 形状数据或 COCO JSON 数据导出为常见 YOLO 检测布局。导出会
保留已经校验的划分成员和类别映射，并把项目 `xyxy` 框转换为归一化的 YOLO
`class cx cy width height` 行。

```bash
uv run detect export-yolo-data \
  --data-dir data/raw \
  --manifest-dir data/manifests \
  --output-dir artifacts/yolo-data
```

输出包含复制的图像、从 0 开始的文本标签、划分图像列表和 `data.yaml`：

```text
artifacts/yolo-data/
|-- data.yaml
|-- train.txt
|-- valid.txt
|-- test.txt
|-- images/{train,valid,test}/
`-- labels/{train,valid,test}/
```

常见 YOLO 文本格式没有与 VOC difficult 或 COCO crowd 对应的字段，因此这些目标不会
进入导出标签。导出目录原子发布；非空目标目录需要显式使用 `--overwrite`。

## 训练引擎边界

本仓库不会在后台把主 trainer 静默替换为第三方 YOLO 引擎。不同 YOLO 实现在架构、增强、
loss、checkpoint、结果格式和许可证上都可能不同。例如，当前 Ultralytics 包采用 AGPL
条款，其他实现可能采用不同条款；安装或分发前应查看所选实现和版本的实际许可。

生成的数据不绑定某个引擎。兼容实现可以读取 `data.yaml`，本仓库只对已校验的数据转换
负责。将独立引擎结果与项目 torchvision 运行比较前，应记录准确版本、配置、权重、硬件和
指标定义。
