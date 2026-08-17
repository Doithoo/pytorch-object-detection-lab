# 配置参考

优先级是 dataclass 默认值、YAML、重复的 CLI `--set KEY VALUE`；支持 `--device` 的命令再应用最终运行时覆盖。未知字段、错误类型和越界值会被拒绝。

| 区域 | 字段 | 含义 |
|---|---|---|
| `data` | `name` | 数据集 provider 名称 |
| `data` | `data_dir`, `manifest_dir` | 源数据与已准备 manifest 根目录 |
| `data` | `num_workers` | 非负 loader workers |
| `data` | `horizontal_flip` | 训练水平翻转概率 |
| `data` | `max_train_samples`, `max_valid_samples`, `max_test_samples` | 可选正数划分上限 |
| `model` | `name` | 注册表键 |
| `model` | `weights` | `none` 或 `imagenet1k_v1` |
| `model` | `expected_num_classes` | 背景加目标类别数 |
| `model` | `params` | 构造器专用 mapping |
| `train` | `epochs`, `batch_size` | 正数轮次与 batch 大小 |
| `train` | `lr`, `momentum`, `weight_decay` | 优化器数值 |
| `train` | `optimizer`, `scheduler` | `sgd`/`adamw` 与 `none`/`step` |
| `train` | `seed`, `amp`, `grad_clip` | 可复现、混合精度与裁剪 |
| `train` | `best_metric` | checkpoint 选择指标 |
| `evaluation` | `score_threshold` | 序列化/渲染预测阈值 |
| `evaluation` | `error_score_threshold`, `error_iou_threshold` | 错误过滤与匹配 IoU |
| `evaluation` | `max_detections` | 每图后端上限 |
| 顶层 | `device` | `auto`、`cpu`、`cuda` 或 `mps` |
| 顶层 | `output_dir`, `run_name` | 产物根目录与可选运行目录名 |

```bash
detect show-config --config configs/learning_minimal.yaml --set train.epochs 1
```
