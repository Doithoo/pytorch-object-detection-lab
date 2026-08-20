# 排错指南

[English](troubleshooting.md) | [Kaggle 指南](kaggle.zh-CN.md)

先按日志中最早出现的错误查找。后面的 nbconvert 警告通常只是 Kaggle 在生成结果页面，
不是训练失败原因。

## Kaggle 提交后立即失败

### `expected one project archive, found []`

使用了旧 runner，它期待手动附加源码压缩包。当前
[`run_kaggle.py`](../recorded-run/kaggle/run_kaggle.py) 已内嵌源码。确认 metadata 的
`code_file` 指向这个文件并重新提交。

### `New Datasets cannot be attached in non-interactive sessions`

使用了运行时调用 `kagglehub.dataset_download` 的旧 runner。当前 runner 不需要 Dataset 或
`kagglehub`，`dataset_sources` 应为空。

### `no kernel image is available for execution on the device`

任务分配了 Tesla P100。当前 Kaggle PyTorch 不支持 P100 的 `sm_60`。在 Settings 中选择
T4 或更新 GPU，再重新提交任务。

### 页面显示 T4 x2，但只有一张卡工作

这是正常现象。本项目是单 GPU 训练，只使用 `cuda:0`。不要因此停止正在输出心跳的任务。

## Kaggle 一直显示 Running

完整训练约需 50-60 分钟。日志每 60 秒出现
`{"phase": "training", "status": "running"}` 时，任务仍然正常。只有心跳停止且页面
报告错误时，才根据最早的 traceback 处理。

## Kaggle CLI 无法使用

- `kaggle: command not found`：运行 `uv tool install kaggle`，检查 uv 工具目录是否在 PATH。
- API 返回未授权：运行 `kaggle auth login --force`。
- kernel ID 不存在：检查 metadata 中用户名和查询命令完全一致。
- 无法申请 GPU：完成 Kaggle 账户验证，并检查本周 GPU 配额。

## VOC 下载或准备失败

- 下载连接失败：确认 Kaggle Internet 已开启；官方主机也可能暂时不可用。
- MD5 不匹配：不要手动跳过检查，重新下载对应压缩包。
- 划分数量不是 `2501 / 2510 / 4952`：确认使用完整官方 VOC 2007。
- 图像或 XML 缺失：重新运行下载和准备，不要直接编辑生成的 CSV 掩盖问题。
- 自定义数据数量不同：使用 `--allow-nonstandard-counts`，并明确它不是官方 VOC 结果。

## 本地环境问题

- `uv sync --locked` 失败：使用 Python 3.10-3.12，并保留已提交的 `uv.lock`。
- 导入了另一份 `object_detector`：检查 `uv run python -c "import object_detector; print(object_detector.__file__)"`。
- 本地 CUDA 不可用：改用 Kaggle；CPU 只建议运行示例和 dry run。
- MPS 某个操作失败：先用 `--device cpu` 和 `data.num_workers=0` 确认是否是设备问题。

## 训练失败

- 预训练权重下载失败：确认 Internet，或把对应权重放进 Torch Hub cache。
- loss 是 NaN / Inf：检查日志报告的图像 ID、框坐标、类别和学习率；先用全精度重现。
- 显存不足：降低 `train.batch_size` 或图像尺寸；更换设置后创建新的运行名称。
- `best.pt` 与 `last.pt` 不同：最后一轮不一定是验证指标最好的轮次，这是正常的。
- 输出目录已存在：给新训练换 `run_name`；不要把两次训练混进同一目录。

## 续训失败

- 优先从同一运行的 `last.pt` 继续，它包含最近的优化器、调度器和随机状态。
- 模型、类别、数据标识、batch size、学习率或数据增强不同：这些变化应开始新运行。
- 请求总轮数不大于 checkpoint 已完成轮数：把 `train.epochs` 设置为更大的目标值。
- checkpoint 版本不支持或文件损坏：不要关闭安全加载；使用本项目生成的有效文件。

## 评估或预测失败

- 评估时数据标识不同：使用训练该 checkpoint 的同一份准备数据。
- 预测不需要 VOC，但需要 checkpoint 中支持的模型名、类别和预处理信息。
- 输出目录已有文件：换新目录；确认不再需要旧结果后才使用 `--overwrite`。
- JSON 中的预测比 PNG 多：`--display-limit` 只限制图片绘制数量。

## 指标看起来很低

随机权重或少量样本 dry run 接近零很正常，它们不是训练成绩。对于完整训练，依次查看
`metrics.csv`、`evaluation.json`、`per_class.csv`、`errors.csv` 和预测图。先确认训练完成
轮数、最佳轮次和数据数量，再判断模型问题。可以与
[已完成的 Kaggle 记录](../recorded-run/README.zh-CN.md)对照文件结构和数值。
