# Checkpoint Schema

schema version `1` 包含解析配置、模型名称/参数、显式权重策略、有序类别、预处理契约、manifest identity 与划分 hash、模型/优化器/scheduler 状态、epoch、最佳指标、指标历史和运行元数据。评估与预测用 `weights=none` 重建模型，不需要 YAML 或下载。

恢复训练要求模型名、类别、预处理、manifest identity 和语义配置一致。允许的操作性覆盖是 `train.epochs`、`data.num_workers`、`device`、`output_dir` 与 `run_name`，且 epoch 必须延长已有运行。其他修改应创建新运行而不是 resume。
