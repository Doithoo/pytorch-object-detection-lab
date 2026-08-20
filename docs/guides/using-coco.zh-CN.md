# 使用 COCO JSON 数据

[English](using-coco.md) | [数据集规则](../reference/dataset-format.zh-CN.md)

COCO 提供器接受 `train`、`valid`、`test` 三个标准 COCO 标注 JSON。三份 JSON
共享一个图像目录，每份都包含 `images`、`annotations` 和 `categories`。

## 目录结构

```text
my-coco/
|-- images/
|   |-- train-001.jpg
|   |-- valid-001.jpg
|   `-- test-001.jpg
`-- annotations/
    |-- instances_train.json
    |-- instances_valid.json
    `-- instances_test.json
```

每个图像记录需要整数 `id`、`file_name`、`width` 和 `height`。每个标注需要
`image_id`、`category_id` 和正数 `[x, y, width, height]` `bbox`。`iscrowd`
可以省略，默认值为 0。类别 ID 可以是不连续的；提供器按类别名称排序，并在
`dataset.yaml` 中生成连续标签。

## 准备

```bash
uv run detect prepare-data --format coco \
  --data-dir my-coco \
  --images-dir images \
  --manifest-dir data/my-coco-manifests \
  --train-annotations annotations/instances_train.json \
  --valid-annotations annotations/instances_valid.json \
  --test-annotations annotations/instances_test.json
```

命令会在发布清单前校验图像尺寸、引用、类别、框、划分重叠和源字节。然后检查并验证：

```bash
uv run detect verify-data --data-dir my-coco --manifest-dir data/my-coco-manifests
uv run detect inspect-data --data-dir my-coco --manifest-dir data/my-coco-manifests --split train --limit 16
```

将 `model.expected_num_classes` 设置为前景类别数量加背景。续训或评估 checkpoint
时，保留生成清单目录和数据目录的对应关系。
