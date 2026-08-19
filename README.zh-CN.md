# PyTorch 目标检测实验室

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![CI](https://github.com/Yashowhoo/pytorch-object-detection-lab/actions/workflows/ci.yml/badge.svg)](.github/workflows/ci.yml)

**英文：[README.md](README.md)**

这是一个面向初学者、可复现的 PyTorch 与 torchvision 目标检测实验室，围绕
Pascal VOC 2007 展示一条完整工作流。项目不会把所有步骤藏进单个训练脚本，而是明确
展示数据、配置、模型、优化、评估、检查点与推理边界。

它适合已经了解 Python、Tensor、损失和梯度，但还没有完整做过目标检测项目的学习者。
主线将讲清 VOC 边界框和困难目标、可变尺寸检测批次、Faster R-CNN 训练/评估接口、
具名损失项、固定清单标识、基于验证集的选模，以及只依赖检查点的预测。

> **实测结果：** 参考 Faster R-CNN 配方已在官方 VOC 2007 划分上完成一次 26 轮训练。
> 保留的 4,952 张测试图像得到 `map_50_95 = 0.322312`、`map_50 = 0.609917`；单张
> Tesla T4 训练耗时 3,025.660 秒。完整配置、指标、失败案例和适用边界见
> [运行证据](docs/recorded-run/README.zh-CN.md)。这只是一次可复现项目实测，不是通用
> benchmark 结论。仓库不提交 145 MB checkpoint，其 SHA-256 已随评估记录。

![带目标框的 VOC 2007 测试集实测预测](docs/recorded-run/evaluation/visualizations/summary.png)

*这是测试图像 `000001` 与已记录最佳 checkpoint 的真实输出；证据页同时保留了误检和
漏检案例。*

![图像、xyxy 边界框、标签、困难标记和目标张量的合成示意图](docs/assets/detection-target-anatomy.png)

*这是由仓库代码生成的确定性合成教学图，不是模型输出。上面的实测结果单独保存了真实
配置、checkpoint 哈希和评估产物。*

## 第一次完整运行

先完整走通下面的路径，再更换模型或移除样本上限：

```text
下载 -> 准备 -> 检查 -> 试运行 -> 训练 -> 评估 -> 预测
```

### 1. 安装并发现 CLI

安装 Python 3.10-3.12、[uv](https://docs.astral.sh/uv/) 和精确锁定的开发环境。如果仓库
和全部锁定依赖尚未存在于本地或缓存中，`git clone` 与
`uv sync --locked --extra dev` 可能访问网络：

```bash
git clone https://github.com/Yashowhoo/pytorch-object-detection-lab.git
cd pytorch-object-detection-lab
uv sync --locked --extra dev
uv run detect --version
uv run detect list-models
uv run detect model-info fasterrcnn_mobilenet_v3_large_320_fpn
```

应识别以下稳定输出：

```text
0.1.0

name	family	weights
fasterrcnn_mobilenet_v3_large_320_fpn	two_stage	none,imagenet1k_v1
fasterrcnn_resnet50_fpn	two_stage	none,imagenet1k_v1
ssdlite320_mobilenet_v3_large	one_stage	none,imagenet1k_v1

name: fasterrcnn_mobilenet_v3_large_320_fpn
family: two_stage
description: Compact Faster R-CNN baseline with a MobileNet V3 FPN backbone.
weights: none, imagenet1k_v1
parameters:
  min_size: Shorter image edge used by the internal detector transform.
  max_size: Maximum longer image edge after resizing.
  box_score_thresh: ROI prediction score threshold applied by the model.
input_notes:
  - Accepts a list of float RGB tensors in [0, 1].
  - Training targets use zero-based continuous xyxy boxes.
```

模型发现只读取注册表元数据，不会构造检测器、下载权重或写入产物。

### 2. 下载并准备官方 VOC 2007

安装完成后，这是数据集网络边界。命令下载官方训练/验证与测试压缩包，在解压前
校验官方发布的 MD5，并把文件存入 `data/raw`。仓库本身不包含完整数据集。

```bash
uv run python scripts/download_data.py --data-dir data/raw
uv run detect prepare-data --data-dir data/raw --manifest-dir data/manifests
```

下载器只复用校验和正确的压缩包，并拒绝不安全的压缩包成员或与现有解压文件冲突的
内容。成功后会依次精确打印：

```text
data/raw/archives/VOCtrainval_06-Nov-2007.tar
data/raw/archives/VOCtest_06-Nov-2007.tar
```

准备命令校验官方划分数量和每一对 JPEG/XML，然后原子写入 `train.csv`、
`valid.csv`、`test.csv`、`dataset.yaml`、`source.yaml` 和 `summary.txt`。标准输出会
给出各划分数量和基于内容生成的清单标识。它先暂存完整结果，再原子替换选定的现有清单
目录，不需要 `--overwrite`。如果原内容仍需保留，请先保存该目录或选择另一个
`--manifest-dir`。使用 `--allow-nonstandard-counts` 前先阅读
[VOC 2007 协议](docs/reference/voc2007.zh-CN.md)。

### 3. 训练前检查数据

先检查结构化数量与范围，再关闭训练期随机变换，渲染训练清单前四行：

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split train --limit 4 --output artifacts/dataset_preview.png
```

`inspect-data` 区分完整划分大小与实际检查行数，并报告普通/困难目标、类别、图像尺寸和
边界框范围。打开 `artifacts/dataset_preview.png`，检查边界框位置与标签。同一清单和
参数会生成确定性预览；重复运行会替换指定 PNG。

### 4. 解析配置并完成一次更新

构造模型前，检查完整类型化配置及每个值的来源：

```bash
uv run detect show-config --config configs/learning_minimal.yaml --set run_name first-detector
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --dry-run --device cpu
```

`show-config` 不会加载数据、构造模型、访问网络或写入运行目录。试运行使用一批已准备
数据，并打印：

```text
image_shapes=((3, H, W), ...)
target_counts=(N, ...)
loss_total=...
loss_classifier=...
loss_box_reg=...
loss_objectness=...
loss_rpn_box_reg=...
dry-run OK
```

数值和形状取决于当前批次。`loss_total` 是具名模型损失项之和。这里列出的四个损失键
属于 Faster R-CNN；其他模型家族可能返回不同名称。打印的每个具名损失都必须是有限
标量。只有完成损失验证和一次真实优化器更新后，才会出现 `dry-run OK`。该命令不会
写入运行目录、检查点或其他产物，也不能证明检测器已经学会有用的边界框。

### 5. 训练有界学习配置

```bash
uv run detect train --config configs/learning_minimal.yaml --set run_name first-detector --device cpu
```

该入门配方运行 2 个训练轮次，并把训练、验证、测试样本分别限制为 32、16、16 张。
它有意保持有界，方便学习者走通机制；其中的指标不是完整 VOC 基准。训练会写入
`artifacts/first-detector`，并拒绝把新运行混入已有运行目录。下一个实验应使用新的
`run_name`。

### 6. 评估验证证据并预测本地图像

还在调整选择时，应评估验证集，而不是反复查看测试集：

```bash
uv run detect evaluate --checkpoint artifacts/first-detector/best.pt --split valid --output-dir artifacts/first-detector/evaluation-valid --device cpu
uv run detect predict --checkpoint artifacts/first-detector/best.pt --image docs/assets/detection-target-anatomy.png --output-dir artifacts/first-detector/prediction --device cpu
```

评估会写入汇总和逐类 AP/AR、序列化预测、分门别类的错误记录与排序后的证据图。预测会写入
`detection-target-anatomy.json` 和 `detection-target-anatomy.png`；使用仓库自带的合成
示意图使命令保持本地、可复现，但输出仍不是质量证据。两个命令默认拒绝产物冲突。
`--overwrite` 表示明确替换，使用前应先检查目标目录。

## 选择模型与权重策略

注册表恰好维护以下三个名称。下表只复述注册表元数据，不暗示已经测得的速度或精度排名。

| 注册名称 | 家族 | 维护用途与取舍 |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | `two_stage` | 使用 MobileNet V3 FPN 骨干网络的紧凑 Faster R-CNN 基线，也是教程默认模型。 |
| `fasterrcnn_resnet50_fpn` | `two_stage` | 使用 ResNet-50 FPN 骨干网络的 Faster R-CNN 对照；比 MobileNet Faster R-CNN 配方需要更多内存和计算。 |
| `ssdlite320_mobilenet_v3_large` | `one_stage` | 使用 MobileNet V3 骨干网络的单阶段 SSDLite 对照；内置变换使用该检测器的 320 像素配方。 |

三个模型都支持 `weights: none` 与 `weights: imagenet1k_v1`：

- `none` 会把检测器和骨干网络的预训练权重都设为 `None`。模型以随机权重离线构造；
  VOC 数据仍须已经存在于本地。
- `imagenet1k_v1` 仍把完整检测器权重设为 `None`，只请求固定的 torchvision ImageNet
  骨干网络权重。精确文件必须已在 torch hub 缓存中，否则模型构造可能访问网络下载。
  本项目从不声称使用了预训练检测头权重。

修改 `model.params` 前，请阅读[模型选择指南](docs/guides/using-models.zh-CN.md)和
[模型参考](docs/reference/model-zoo.zh-CN.md)。

## 读懂运行产物

一次完整训练目录是一个不可拆散的可复现单元：

| 产物 | 记录内容 |
|---|---|
| `config.yaml` | 该运行实际使用的完整解析类型化配置。 |
| `run.yaml` | 环境、设备、种子、Git 修订版本、有序类别、清单标识和划分哈希。 |
| `metrics.csv` | 每个训练轮次的具名训练损失项和验证指标。 |
| `best.pt` | 截至当时验证 `map_50_95` 最优的自包含检查点。 |
| `last.pt` | 最近完成的训练轮次，以及优化器、调度器、历史和续训状态。 |

评估另外生成 `evaluation.json`、`per_class.csv`、`predictions.json`、`errors.csv` 和
`visualizations/`。单图预测在 JSON 中保留浮点边界框、分数、每个检测对应的类别名称和
清单标识，并单独保存标注 PNG；完整有序类别列表保存在检查点中。不要从文件名猜测来源，
应查询[指标与评估产物参考](docs/reference/metrics.zh-CN.md)和
[检查点结构](docs/reference/checkpoint-schema.zh-CN.md)。

清单标识绑定源图像、XML 标注、类别、坐标和划分成员。标识不同的运行不是同一数据实验。
使用一个明确指标比较兼容的验证运行：

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

`compare-runs` 会验证清单标识一致，并报告有语义的配置差异。指定 `--output` 时，如果 CSV
已存在，命令会拒绝替换；请改用新的输出路径。使用验证集选择配方和检查点，然后冻结协议，
最后只在保留的测试集上生成一次报告。有样本限制或改动过的划分即使使用 `test` 评估，
仍然只是有界证据。

## 按任务查文档

- **学习完整工作流：** [学习路线](docs/tutorial/learning-path.zh-CN.md)和
  [教程索引](docs/tutorial/README.zh-CN.md)。
- **操作项目：** [模型选择](docs/guides/using-models.zh-CN.md)、
  [受控实验](docs/guides/experiments.zh-CN.md)、[使用自己的 VOC 形状数据](docs/guides/using-your-data.zh-CN.md)
  和[排错指南](docs/guides/troubleshooting.zh-CN.md)。
- **理解系统：** [端到端检测流程](docs/concepts/detection-flow.zh-CN.md)、
  [Faster R-CNN 原理](docs/concepts/how-faster-rcnn-works.zh-CN.md)、
  [配置流程](docs/concepts/configuration-flow.zh-CN.md)和[代码导览](docs/concepts/code-tour.zh-CN.md)。
- **查询契约：** [配置](docs/reference/config-reference.zh-CN.md)、
  [数据集与清单](docs/reference/dataset-format.zh-CN.md)、[指标](docs/reference/metrics.zh-CN.md)、
  [模型注册表](docs/reference/model-zoo.zh-CN.md)和 [VOC 2007 协议](docs/reference/voc2007.zh-CN.md)。
- **审阅证据决策：** [架构决策 0001](docs/architecture/0001-reproducible-voc-detection-contracts.zh-CN.md)
  和[运行记录发布门槛](docs/recorded-run/README.zh-CN.md)。
- **浏览可运行材料：** [示例](examples/README.zh-CN.md)、[配置](configs/README.zh-CN.md)、
  [脚本](scripts/README.zh-CN.md)和[测试](tests/README.zh-CN.md)。

完整地图见[文档导航](docs/README.zh-CN.md)。

## 开发与项目政策

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
```

测试只使用本地合成数据或临时测试数据，不会下载 VOC 或预训练权重。提交修改前请阅读
[贡献指南](CONTRIBUTING.zh-CN.md)，报告安全漏洞前请阅读[安全政策](SECURITY.md)。项目采用
[MIT License](LICENSE)。

<!-- Documentation contract: download -> prepare -> inspect -> dry run -> train -> evaluate -> predict | recorded full-VOC score 0.322312 -->
