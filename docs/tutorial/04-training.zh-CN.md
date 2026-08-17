# 04 - 训练

训练会解析配置、校验数据集/模型 identity、设置随机种子并原子写 checkpoint。`--dry-run` 只执行一次更新并输出尺寸、目标数和损失。恢复训练可以修改 epochs、workers、device、输出目录或 run name，但不能改变训练语义。

运行：`uv run detect train --config configs/learning_minimal.yaml --dry-run --device cpu`

预期：诊断以 `dry-run OK` 结束，不写运行目录或 checkpoint。
