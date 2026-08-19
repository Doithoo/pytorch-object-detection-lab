# 按边界排查故障

[English](troubleshooting.md) | [配置参考](../reference/config-reference.zh-CN.md)

先运行能够跨越故障边界的最小命令。在原因明确之前，应保留清单、检查点和已有输出目录。

## 安装或命令解析失败

```bash
uv run detect --version
uv run detect show-config --config configs/learning_minimal.yaml
```

- `unknown configuration field: ...`：修正对应 YAML 或 `--set` 路径；未知键不会被透传。
- `... must be ...`：YAML 把值解析成了其他类型，或数值超出规定范围。`null` 与 `~` 会变成 Python `None`，而 `none` 会保留为字符串。
- `invalid override`：`--set` 的值按 YAML 解析，错误 YAML 会在训练前失败。
- Argparse 打印 `usage:` 并以 2 退出：参数属于其他子命令，或者缺失/无效。训练命令可运行 `uv run detect train --help` 检查。

`show-config` 不读取数据集、不构造模型、不访问网络，也不写产物。输出中的 `sources` 映射会说明每个叶字段来自 `default`、`yaml` 还是 `cli`。

## 数据准备或加载失败

- `... split has ... images; expected ...`：目录不是完整官方 VOC 2007。应重新下载或修复；只有有意构造 VOC 形状夹具时才使用 `--allow-nonstandard-counts`。
- `split contains duplicate image IDs` 或 `split overlap`：修复划分文件；错误不会替换已有清单的一部分。
- 缺少图像或标注、文件名不匹配、尺寸不一致、XML 错误、未知类别或非正检测框：修复错误中指出的源样本，再重新准备。
- 训练时才报告图像尺寸不一致：源内容可能在准备后被修改，应重新准备并使用新标识。

同时检查结构和像素：

```bash
uv run detect inspect-data --manifest-dir data/manifests --data-dir data/raw --split train --limit 16
uv run python scripts/preview_dataset.py data/manifests --data-dir data/raw --split valid --limit 4 --output artifacts/dataset-preview.png
```

准备和结构检查不会创建 `dataset-preview.png`，只有预览脚本会创建。若 `--output` 指向的 PNG 已存在，脚本会直接覆盖且不询问；需要保留旧证据时应换一个输出路径。解析成功也不能证明自定义坐标约定在图像上正确。

## 预检查或模型构造失败

- `missing train.csv, ...`：`data.manifest_dir` 缺少必须的准备文件。
- `expected 21, dataset requires ...`：`model.expected_num_classes` 必须等于背景加元数据类别数。
- 请求 CUDA 或 MPS 但不可用：使用可用设备，或修复环境。
- `unsupported device`：使用 `auto`、`cpu`、`mps` 或有效的 `cuda...` 字符串。
- `cannot write below ...`：选择可写的 `output_dir`。
- 提示权重未缓存：除非提供对应的 torchvision 缓存文件，否则构造时需要网络。要保证模型路径离线，请选择 `model.weights=none`。
- 未知模型或关键字参数错误：运行 `list-models` 和 `model-info`，修正 `model.name` 或 `model.params`。

## 试运行或训练失败

```bash
uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu
```

- `non-finite <loss> for image IDs [...]`：检查报告样本、坐标、类别、学习率和全精度 CPU 路径，不要静默跳过批次。
- 后端内存不足：降低 `train.batch_size`，通过文档支持的模型参数缩小模型自有输入，或更换模型。这会改变实验，必须创建新运行。
- 已存在的运行目录被拒绝：使用新的 `run_name`；新运行绝不会追加到旧文件。
- 混合精度与预期不符：缩放器只在 CUDA 启用，自动类型转换只用于 CPU/CUDA，因此解析为 MPS 时始终使用全精度。当前只有配置的 `device` 明确设为 `mps` 时预检查才提示；`device=auto` 可以解析为 MPS 而不显示该提示。

有限损失和 `dry-run OK` 只证明一次更新链路连通，不能证明收敛或检测质量。

## 续训、评估或预测失败

- `resume identity mismatch`：模型名、有序类别、清单标识或严格预处理契约发生变化。应开始新运行，或恢复匹配输入。
- `resume configuration changes training semantics`：只有总轮次、工作进程数、设备、输出目录和运行名可以变化。
- 请求轮次不大于检查点轮次：提高 `train.epochs`。
- 续训目标与检查点目录无关且非空：让 `run_name` 指向检查点父目录，或使用新的空目录。
- 历史最佳检查点不可用或不兼容：从 `last.pt` 续训到另一个空运行时，恢复与它匹配的同级 `best.pt`；也可以从有效的 `best.pt` 续训到新的空运行目录，或仅在原目录缺少 `last.pt` 时用其原始精确路径原位恢复。
- 续训报告指标历史、历史最佳 `lineage_id`、严格最佳历史或 CUDA RNG 不匹配：恢复同一 lineage 中未经修改的检查点。配置验证指标的每个值都必须有限，且 `best_metric` 必须等于完整历史最大值。CUDA 元数据必须包含 `cuda:0` 这类显式设备索引，其 RNG 条目按该检查点自身的 `run_metadata.cuda_device_count` 校验，而不是与 `last.pt` 比较。
- 不支持的 `schema_version`、受限的张量与容器安全加载失败或预处理契约错误：文件损坏、不受信任或不是版本 1。版本 1 有意包含安全的基础值、列表、映射和张量；不要回退到不受限制的 pickle 加载。
- 评估报告清单不匹配：恢复与检查点匹配的准备数据。预测不声明数据集指标，因此仍可在没有清单时运行。
- 评估或预测输出已存在：保留原输出并换路径；只有明确要替换时才使用 `--overwrite`。

## 指标看起来异常

AP 使用模型的原始预测。评估命令的 `--score-threshold` 只影响序列化预测和图像，不影响 AP/AR。错误分类使用检查点配置中的 `evaluation.error_score_threshold` 和 `evaluation.error_iou_threshold`。困难目标不计入普通目标数量和漏检；与其匹配的预测标为 `ignored`。

随机初始化和两轮有界运行可能得到接近零的指标。先检查 `evaluation.json`、`per_class.csv`、`errors.csv` 和 `visualizations/`，再提出假设。参考配方本身不是证据；应与[实测运行](../recorded-run/README.zh-CN.md)的产物结构比较，而不是只比较分数。

报告故障时，请附上完整命令、简洁错误、解析配置、清单标识，以及相关的检查点版本或哈希、框架版本、设备和 Git 修订，并移除私人路径与数据。模块职责见[代码导览](../concepts/code-tour.zh-CN.md)。
