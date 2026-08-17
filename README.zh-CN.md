# PyTorch 目标检测实验室

[English](README.md) | [中文文档](docs/README.zh-CN.md) | [示例](examples/README.zh-CN.md)

这是一个面向初学者、可复现的 Pascal VOC 2007 目标检测实验室，基于 PyTorch 与 torchvision。它适合希望看清成熟检测器外围每个数据和工程边界，而不是复制黑盒训练脚本的学习者。

完整学习路径严格为：`download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`。

## 安装

支持 Python 3.10-3.12，并使用 [uv](https://docs.astral.sh/uv/) 管理锁定环境。

```bash
uv sync --locked --extra dev
uv run detect --version
```

## 七阶段工作流

1. 下载两份 VOC 2007 官方归档，并校验发布的 MD5。

   ```bash
   uv run python scripts/download_data.py --data-dir data/raw
   ```

2. 校验官方划分并生成确定性 manifest。

   ```bash
   detect prepare-data --data-dir data/raw --manifest-dir data/manifests
   ```

3. 训练前检查图像、坐标转换、类别与 difficult 标注。

   ```bash
   uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --output artifacts/dataset_preview.png
   ```

4. 离线执行一次前向、反向与参数更新；默认学习配置不请求预训练权重。

   ```bash
   detect train --config configs/learning_minimal.yaml --dry-run --device cpu
   ```

5. 运行有界学习实验；先理解完整流程，再移除样本上限。

   ```bash
   detect train --config configs/learning_minimal.yaml --set train.epochs 2 --device cpu
   ```

6. 使用自包含 checkpoint 评估，输出 COCO 风格指标、预测、确定性错误和证据图。

   ```bash
   detect evaluate --checkpoint artifacts/run/best.pt --split test --output-dir artifacts/evaluation --device cpu
   ```

7. 无需 YAML 或下载权重，对单图或目录预测。

   ```bash
   detect predict --checkpoint artifacts/run/best.pt --image data/raw/VOCdevkit/VOC2007/JPEGImages/000001.jpg --output-dir artifacts/prediction --device cpu
   ```

## 产物

训练目录包含解析后的 `config.yaml`、记录环境与 manifest 来源的 `run.yaml`、`metrics.csv`，以及自包含的 `best.pt`/`last.pt`。评估还生成 `evaluation.json`、`per_class.csv`、`predictions.json`、`errors.csv` 与标注图。预测 JSON 保留浮点框、有序类别和 manifest identity。

## 模型与结果边界

模型注册表包含默认的 Faster R-CNN MobileNet V3 Large 320 FPN、Faster R-CNN ResNet-50 FPN 和 SSDLite 320 MobileNet V3 Large。`weights: none` 完全离线；`weights: imagenet1k_v1` 明确要求本地缓存或网络。

参考配置目前 **no published full-VOC score**，即没有发布完整 VOC 指标。仓库测试和示例只使用合成或有界数据，不宣称可与完整数据集结果比较。VOC 2007 test 划分只用于最终评估，不用于选模。

## 仓库结构

- `configs/`：学习、参考和对比配方。
- `src/object_detector/`：类型化的数据、模型、训练、评估与推理模块。
- `scripts/`：下载、预览与指标绘图工具。
- `examples/`：五个渐进式本地示例。
- `docs/`：教程、原理、指南与参考。
- `tests/`：离线单元、集成、打包与验收测试。

## 开发

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest -W error::DeprecationWarning
```

参见 [CONTRIBUTING.zh-CN.md](CONTRIBUTING.zh-CN.md)。项目采用 [MIT License](LICENSE)。
