# 添加模型

在 `models/registry.py` 注册稳定名称、构造器、支持的权重策略与默认参数。构造器接收 `num_classes`、权重策略和模型参数；`none` 必须避免下载。添加注册校验、forward smoke、checkpoint 恢复与离线 preflight 测试，不要重写 torchvision 已维护的检测器内部实现。
