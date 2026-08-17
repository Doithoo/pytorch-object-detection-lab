# 实验管理

从 `learning_minimal.yaml` 开始，设置唯一 `run_name`，并通过 `--set KEY VALUE` 每次只改变一个假设。把解析配置、运行元数据、manifest identity、指标与 checkpoint 一起保留。使用 validation AP 选模，选择固定后才评估保留的 test 划分。

```bash
detect train --config configs/learning_minimal.yaml --set run_name experiment-01 --set train.epochs 2 --device cpu
```
