# 代码导览

`config.py` 解析类型化设置；`data/` 解析 VOC、生成 manifest、构造样本并同步变换；`models/` 注册显式构造器与权重策略；`training/` 负责更新、checkpoint 和运行产物；`evaluation/` 负责 AP/AR、错误与证据；`inference/` 从自包含 checkpoint 恢复并处理本地图像；`cli.py` 只做 argparse 适配。
