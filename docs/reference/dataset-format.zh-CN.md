# 数据集与清单规则

[English](dataset-format.md) | [VOC 2007 协议](voc2007.zh-CN.md)

本参考定义版本 0.1 唯一的运行时数据集规则，适合数据作者、扩展维护者和实验标识审计者。提供器接受 Pascal VOC 形状目录和任意非空类别名。

## 源目录与 XML

准备阶段读取 `data_dir/VOCdevkit/VOC2007`，把 `ImageSets/Main/train.txt` 映射为 `train`、`val.txt` 映射为 `valid`、`test.txt` 映射为 `test`。划分文件的非空行是无扩展名的图像标识；同一划分内不能重复，划分之间不能重叠。每个标识对应 `JPEGImages/<id>.jpg` 和 `Annotations/<id>.xml`。

类别会从全部通过校验的 XML 中收集，并按字典序分配标签。官方 VOC 2007 仍使用发布的 `VOC_CLASSES` 顺序。

XML 根节点必须提供非空 `filename`、正整数 `size/width` 和 `size/height`，并可包含零个或多个 `object`。每个目标需要非空的 `name`，以及含有限数值 `xmin`、`ymin`、`xmax`、`ymax` 的 `bndbox`。可选的 `difficult` 默认为 `0`，若存在只能是 `0` 或 `1`。文件名和解码图像尺寸必须与 XML 一致。

VOC 坐标一基且端点包含。解析后得到零基连续 `xyxy`：`(xmin - 1, ymin - 1, xmax, ymax)`；所有值裁剪到 `[0,width]` 或 `[0,height]`，裁剪后宽或高非正则拒绝。最大角是排他的连续边界。

VOC 坐标一基且端点包含。解析后得到零基连续 `xyxy`：`(xmin - 1, ymin - 1, xmax, ymax)`；所有值裁剪到 `[0,width]` 或 `[0,height]`，裁剪后宽或高非正则拒绝。最大角是排他的连续边界。

## CSV 清单

`train.csv`、`valid.csv` 和 `test.csv` 严格按下列顺序包含三列：

| 列 | 值 |
|---|---|
| `image_id` | 源划分标识 |
| `image_path` | 相对 `dataset_root` 的 POSIX 路径 `JPEGImages/<id>.jpg` |
| `annotation_path` | 相对 `dataset_root` 的 POSIX 路径 `Annotations/<id>.xml` |

行顺序与划分文件一致。CSV 只引用源数据，不包含或复制图像与 XML 字节。

## `dataset.yaml`

| 键 | 类型与职责 |
|---|---|
| `name` | 字符串，当前为 `voc2007` |
| `dataset_root` | 字符串，当前为相对 `data_dir` 的 `VOCdevkit/VOC2007` |
| `class_names` | 推导出的前景类别有序序列 |
| `label_by_name` | 每个名称到从 1 开始的整数标签映射 |
| `split_counts` | `train`、`valid`、`test` 到行数的映射 |
| `split_hashes` | 各划分源行和引用的 JPEG/XML 字节的 SHA-256 摘要映射 |
| `manifest_hashes` | 各划分精确 CSV 字节的 SHA-256 摘要映射 |
| `identity` | 组合后的 SHA-256 实验标识 |
| `coordinate_convention` | 精确字符串 `zero-based continuous xyxy; xmax/ymax are exclusive pixel boundaries` |
| `schema_version` | 清单格式版本整数，当前为 `2` |

`split_hashes` 覆盖源内容，`manifest_hashes` 覆盖已发布 CSV 的精确字节。运行时会在构造数据集前校验格式版本、类别顺序与标签映射、划分数量、CSV 摘要和元数据标识。如果清单文件发生变化，请重新生成清单，不要手工编辑。

每行的划分哈希依次加入 `image_id,image_path,annotation_path\\n`、两个相对路径字符串，以及对应图像和 XML 的完整字节。组合标识对包含 `name`、有序 `classes`、`coordinate_convention` 与 `split_hashes` 的规范 JSON 求哈希。因此源字节、路径、顺序、成员、类别或坐标规则变化都会改变标识；文件时间戳和绝对 `data_dir` 不参与。

## `source.yaml` 与 `summary.txt`

`source.yaml` 包含 `dataset: Pascal VOC 2007`、相对 `dataset_root`，以及从两个官方 tar 文件名到发布 MD5 的 `archives` 映射。它是生成的来源元数据；对于有意构造的非标准夹具，它不能证明字节确实来自官方压缩包。

`summary.txt` 是纯文本：先写 `identity: <sha256>`，再写 `train: <count>`、`valid: <count>` 和 `test: <count>`。这两个文件用于说明；运行时加载由 CSV 和 `dataset.yaml` 驱动。

六个文件先写入暂存目录，再作为整体原子替换目标。校验或发布失败时，不会暴露部分新清单。

## 运行时样本规则

加载器解码 RGB，并返回缩放到 `[0,1]` 的 `image: float32 Tensor[3,H,W]`。目标结构为：

| 字段 | 数据类型 | 形状 | 含义 |
|---|---|---|---|
| `boxes` | `float32` | `[N,4]` | 零基连续 `xyxy` |
| `labels` | `int64` | `[N]` | 从 1 开始的前景标识；0 保留给背景 |
| `image_id` | `int64` | `[1]` | 根据源标识 SHA-256 的前八个字节生成的非负 63 位整数 |
| `area` | `float32` | `[N]` | `(xmax-xmin)*(ymax-ymin)` |
| `iscrowd` | `int64` | `[N]` | 仅当 VOC `difficult=1` 时为 1 |
| `difficult` | `bool` | `[N]` | 原始困难标记 |

训练在增强前移除困难目标；验证、测试、检查和可视化保留它们。空图像和仅困难目标的训练样本仍有效，`boxes` 形状为 `[0,4]`，逐目标向量为 `[0]`。水平翻转会更新检测框；退化框会连同全部对齐字段一起过滤。批处理返回图像列表与目标列表，而不是堆叠张量。

可运行 `uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16` 查看少量样本的结构，再参考[使用自己的数据](../guides/using-your-data.zh-CN.md)执行准备和预览。
