# 测试指南

[English](README.md) | [参与贡献](../CONTRIBUTING.zh-CN.md)

仓库测试必须全部离线运行。测试使用合成张量、临时构造的 VOC 格式测试数据、fake detector，或者以 `weights: none` 构造的模型。请先通过 `uv sync --locked --extra dev` 安装开发环境。

## 选择最小测试层

| 层级 | 适合验证的改动 | 代表文件 | 预期输出 |
|---|---|---|---|
| 单元 | 单个 schema、变换、解析器、指标或 checkpoint 规则 | `test_config.py`、`test_voc.py`、`test_transforms.py`、`test_metrics.py`、`test_checkpoint.py` | 快速检查一项行为，产物只存在于临时目录 |
| 数据集成 | 准备流程、manifest identity、加载、检查或 preflight | `test_manifest.py`、`test_dataset.py`、`test_inspection.py`、`test_preflight.py` | pytest 临时目录中的合成 VOC manifests、targets、摘要与预览图 |
| 模型与优化 | 注册表、权重策略、一次 detector 更新或 trainer 行为 | `test_models.py`、`test_model_smoke.py`、`test_trainer.py`、`test_training.py` | 离线构造，以及使用合成数据或 fixture 的参数更新，且各项 loss 值均为有限数；不宣称 benchmark 指标 |
| 评估与推理 | AP 规则、错误分析、运行比较、报告或预测 | `test_evaluation.py`、`test_errors.py`、`test_comparison.py`、`test_inference.py` | 临时目录中的确定性 JSON、CSV、checkpoint 和图片产物 |
| 端到端 | 改动同时影响 CLI、数据、训练、评估与预测 | `test_end_to_end.py`、`test_cli.py` | 一条完整离线合成流程，以及命令行解析器的实际行为 |
| 发布与示例 | 文档、package metadata、脚本或可运行示例 | `test_documentation.py`、`test_examples.py`、`test_packaging.py`、`test_scripts.py`、`test_download_data.py` | 有效链接和命令、声明的 package 文件、可离线执行的帮助命令，以及脚本产物规则 |

## 聚焦命令

修改单一规则时，只运行一个测试：

```bash
uv run --no-sync pytest tests/test_config.py::test_yaml_then_cli_override_precedence -q
```

相关模块一起变化时，运行对应测试组：

```bash
uv run --no-sync pytest tests/test_manifest.py tests/test_dataset.py tests/test_inspection.py -q
uv run --no-sync pytest tests/test_models.py tests/test_model_smoke.py tests/test_trainer.py -q
uv run --no-sync pytest tests/test_evaluation.py tests/test_inference.py tests/test_comparison.py -q
uv run --no-sync pytest tests/test_documentation.py tests/test_examples.py tests/test_packaging.py -q
```

聚焦测试通过只代表选中的行为通过，不代表其他层或完整流程已经通过。`test_model_smoke.py` 会构造真实 torchvision 检测器，因此可能比 fake-detector 单元测试慢，但不会下载权重。

## 完整验证

提交影响多个模块的改动前，运行完整套件，并把弃用警告提升为错误：

```bash
uv run --no-sync pytest -W error::DeprecationWarning
```

然后执行[参与贡献](../CONTRIBUTING.zh-CN.md)列出的静态检查。完整测试应把生成文件留在 pytest 临时目录，不改动仓库的数据或产物目录。

## 适用范围

测试通过表示被覆盖的软件行为正常，其中包括一条离线合成端到端路径。它不能说明检测器已经收敛，也不能建立完整 VOC 成绩。单独的 [Kaggle 训练记录](../docs/recorded-run/README.zh-CN.md)来自官方数据和真实 GPU 运行，测试不会重新生成它。
