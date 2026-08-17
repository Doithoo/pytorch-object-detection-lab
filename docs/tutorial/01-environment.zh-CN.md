# 01 - 环境

uv 为 Python 3.10-3.12 安装锁定的运行与开发依赖。学习配置使用随机初始化检测器并保持离线；ImageNet backbone 策略是显式的，本地无缓存时可能需要网络。

运行：`uv run detect show-config --config configs/learning_minimal.yaml`

预期：解析后的 YAML 显示 `weights: none`、默认检测器与有界样本上限。
