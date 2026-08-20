# 训练配置

[English](README.md) | [配置字段参考](../docs/reference/config-reference.zh-CN.md)

配置按“默认值 -> YAML -> `--set KEY VALUE`”的顺序合并。开始训练前可以查看最终结果：

```bash
uv run detect show-config --config configs/reference_fasterrcnn.yaml
```

这条命令只打印配置，不加载数据、不构造模型，也不会开始训练。

## 项目提供的配置

| 文件 | 用途 | 权重与网络 |
|---|---|---|
| `reference_fasterrcnn.yaml` | Kaggle 主训练：Faster R-CNN MobileNet V3、26 轮、完整 VOC | `imagenet1k_v1`；需要联网下载或已有缓存 |
| `learning_minimal.yaml` | 本地 dry run 或少量样本代码检查 | `none`；模型随机初始化，不下载权重 |
| `fasterrcnn_resnet50_fpn.yaml` | 尝试更大的 Faster R-CNN backbone | `none`；默认不下载权重 |
| `ssdlite320_mobilenet_v3.yaml` | 尝试单阶段 SSDLite | `none`；默认不下载权重 |

项目发布的完整训练结果只来自 `reference_fasterrcnn.yaml` 的已完成 Kaggle 训练。其他配置
没有发布完整 VOC 成绩。

## 推荐选择

第一次训练直接使用 Kaggle runner，它会加载 `reference_fasterrcnn.yaml` 并覆盖设备、AMP、
worker 数量和 Kaggle 路径。步骤见 [Kaggle 指南](../docs/guides/kaggle.zh-CN.md)。

只想在本地确认数据和模型能完成一次更新时，使用：

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

想比较模型时，为每次运行设置不同名称，并保持数据、随机种子、训练轮次和优化器一致：

```bash
uv run detect train --config configs/ssdlite320_mobilenet_v3.yaml --set run_name ssdlite-check --dry-run --device cpu
```

dry run 不保存 checkpoint，也不代表模型已经训练完成。

## 运行后保存什么

正常训练目录包含 `config.yaml`、`run.yaml`、`metrics.csv`、`best.pt` 和 `last.pt`。保存
`config.yaml` 很重要，因为它记录默认值、YAML 和命令行覆盖合并后的实际设置。

比较两个兼容运行：

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

先使用验证集比较设置，最终测试集只在选择完成后评估。
