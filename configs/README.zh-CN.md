# 配置配方

[English](README.md) | [配置参考](../docs/reference/config-reference.zh-CN.md)

配置按“类型化默认值、YAML、重复的 `--set KEY VALUE` 覆盖”解析；支持 `--device` 的命令最后再应用该运行时覆盖。构造模型前先检查结果：

```bash
uv run detect show-config --config configs/learning_minimal.yaml
```

该命令输出解析后的 YAML 与每个值的来源，不读取数据、不构造模型、不访问网络，也不写训练产物。

## 仓库内配方

| 文件 | 作用与范围 | 网络行为 | 预期训练产物 |
|---|---|---|---|
| `learning_minimal.yaml` | 默认学习路线，使用 Faster R-CNN MobileNet V3 Large 320 FPN、2 个 epoch，train/valid/test 上限为 32/16/16 | `weights: none`，模型构造离线；源数据必须已经位于本地 | 未覆盖 `run_name` 时，在 `artifacts/run` 写入 `config.yaml`、`run.yaml`、`metrics.csv`、`best.pt`、`last.pt` |
| `fasterrcnn_resnet50_fpn.yaml` | Faster R-CNN ResNet-50 FPN 的短期无样本上限对比配方；省略字段继承类型化默认值 | `weights: none`，模型构造离线，也不会下载数据集 | 标准运行产物；它虽然只有 2 个 epoch，但既不是有界学习配方，也不是证据完整的参考运行 |
| `ssdlite320_mobilenet_v3.yaml` | SSDLite 320 MobileNet V3 Large 的短期无样本上限对比配方；省略字段继承类型化默认值 | `weights: none`，模型构造离线，也不会下载数据集 | 标准运行产物；设置唯一 `run_name` 后可用于受控的模型家族对比 |
| `reference_fasterrcnn.yaml` | 完整 VOC 参考配方，使用 Faster R-CNN MobileNet V3 Large 320 FPN、26 个 epoch、step scheduler，并取消样本上限；仓库默认值可在 CPU 上执行 | `weights: imagenet1k_v1`；torch cache 中必须已有固定 backbone 权重，否则需要网络下载 | `artifacts/reference-fasterrcnn` 中的标准运行文件；已记录 Kaggle 运行另行保存 CUDA/AMP 覆盖项与评估产物 |

## 如何选择

使用 `learning_minimal.yaml` 学习 `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`。它带有样本上限，属于有界学习运行。两个短期模型家族配方应在检查解析后默认值并设置唯一运行名之后使用，例如：

```bash
uv run detect train --config configs/ssdlite320_mobilenet_v3.yaml --set run_name ssdlite-check --dry-run --device cpu
```

dry run 的预期输出是 batch 诊断、值均为有限数的各项具名 loss，以及 `dry-run OK`；它不会创建运行目录或 checkpoint。

只有在官方数据准备完成且算力预算明确时，才使用 `reference_fasterrcnn.yaml`。一次证据完整的执行已经发布在[实测运行](../docs/recorded-run/README.zh-CN.md)中：Kaggle runner 把操作字段改成 CUDA、AMP、两个 worker 与 Kaggle 路径，同时保持模型和优化配方不变。单独一份 YAML 仍然不构成结果证据。

## 产物与比较规则

正常训练保存的是解析后配置，而不是简单复制输入 YAML。请把 `config.yaml`、`run.yaml`、`metrics.csv` 和两个 checkpoint 一起保留，并用不同 `run_name` 隔离实验。`detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95` 只接受 manifest identity 相同的兼容运行目录：

```bash
uv run detect compare-runs artifacts/experiment-a artifacts/experiment-b --metric valid_map_50_95 --output artifacts/comparison.csv
```

该命令输出对比表格并可选写入 CSV；它不会训练，不会评估保留的 test 划分，也不能让两份不同准备的数据变得可比较。
