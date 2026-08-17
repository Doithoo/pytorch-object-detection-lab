# 检测数据流

图像与 target 字典进入可变长度 collate batch。训练模式下 torchvision 检测器返回具名标量损失，trainer 校验有限性、求和并更新参数；评估模式下返回框、类别和分数。这些 CPU 预测不经阈值过滤进入 torchmetrics，同时供确定性错误匹配、JSON 序列化和渲染使用。
