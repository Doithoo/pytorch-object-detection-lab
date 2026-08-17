# 模型列表

| 注册名称 | 角色 | 权重策略 |
|---|---|---|
| `fasterrcnn_mobilenet_v3_large_320_fpn` | 默认轻量学习检测器 | `none`, `imagenet1k_v1` |
| `fasterrcnn_resnet50_fpn` | 较大 Faster R-CNN 对比 | `none`, `imagenet1k_v1` |
| `ssdlite320_mobilenet_v3_large` | 单阶段对比 | `none`, `imagenet1k_v1` |

`none` 构造时不下载。ImageNet 策略只初始化支持的 backbone，preflight 会报告是否需要缓存/网络。checkpoint 恢复始终先以 `none` 构造，再加载已保存张量。
