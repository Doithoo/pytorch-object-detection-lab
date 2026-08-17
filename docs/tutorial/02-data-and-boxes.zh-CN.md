# 02 - VOC 数据与框

VOC XML 使用从一开始且端点包含的坐标。解析时把 `(xmin,ymin,xmax,ymax)` 转为零基连续 xyxy：最小坐标减一，最大坐标保持。difficult 目标在验证/测试中以 `iscrowd=1` 保留，但从训练 target 与普通目标计数中排除。

运行：`uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --limit 4`

预期：`artifacts/dataset_preview.png` 显示普通框和虚线 difficult 框。
