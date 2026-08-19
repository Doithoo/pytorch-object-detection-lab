# 文档导航

[English](README.md)

第一次使用请从[教程](tutorial/README.zh-CN.md)开始。教程严格遵循 `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`，并说明每一步能够证明什么、不能证明什么。基础前提是 Python 3.10-3.12、uv 和本地仓库副本；涉及数据的阶段还需要本地 Pascal VOC 2007 数据。

## 按任务选择入口

| 目标 | 从这里开始 | 适用场景 | 预期结果 |
|---|---|---|---|
| 学习完整流程 | [教程索引](tutorial/README.zh-CN.md)与[学习路径](tutorial/learning-path.zh-CN.md) | 希望按依赖顺序执行命令 | manifests、数据预览、dry run 诊断、有界训练产物、评估报告和预测文件 |
| 选择并检查模型 | [使用模型](guides/using-models.zh-CN.md)与[模型目录](reference/model-zoo.zh-CN.md) | 需要了解权重策略、模型元数据或对比起点 | `detect list-models` 或 `detect model-info fasterrcnn_mobilenet_v3_large_320_fpn` 输出的注册表信息；这些命令不会训练模型 |
| 配置一次运行 | [配置流](concepts/configuration-flow.zh-CN.md)、[配置参考](reference/config-reference.zh-CN.md)与[配置索引](../configs/README.zh-CN.md) | 需要确认优先级、校验规则或选择仓库配方 | `detect show-config` 输出解析后的 YAML，并标明每个值的来源 |
| 准备或替换数据 | [使用自己的数据](guides/using-your-data.zh-CN.md)、[数据格式](reference/dataset-format.zh-CN.md)与[VOC 2007 参考](reference/voc2007.zh-CN.md) | 需要处理划分、坐标、difficult 目标或 manifest | 经过校验的 CSV manifests 与 `dataset.yaml`；准备成功本身不代表模型有效 |
| 理解检测器行为 | [检测数据流](concepts/detection-flow.zh-CN.md)与[Faster R-CNN 原理](concepts/how-faster-rcnn-works.zh-CN.md) | batch、loss 字典、预测或指标不符合预期 | 从源标注到 checkpoint 输出的可追踪契约 |
| 运行和比较实验 | [实验指南](guides/experiments.zh-CN.md)、[指标参考](reference/metrics.zh-CN.md)与[checkpoint schema](reference/checkpoint-schema.zh-CN.md) | 一次只验证一个假设，或比较兼容运行 | 完整的运行溯源信息、validation 指标、checkpoint，以及可选的比较 CSV |
| 在 Kaggle 上训练 | [Kaggle 指南](guides/kaggle.zh-CN.md)与[实测运行](recorded-run/README.zh-CN.md) | 本地 CPU 执行完整参考配方太慢 | 一次 T4 训练及可下载的训练与评估产物 |
| 定位故障 | [排错指南](guides/troubleshooting.zh-CN.md)与[代码导览](concepts/code-tour.zh-CN.md) | 命令失败或产物异常 | 定位故障所需的最小命令及对应模块或测试层 |
| 扩展项目 | [添加数据集](guides/adding-datasets.zh-CN.md)或[添加模型](guides/adding-models.zh-CN.md) | 修改 provider 或模型注册契约 | 聚焦的离线测试，以及同步更新的中英文文档 |
| 审查可复现性决策 | [架构决策 0001](architecture/0001-reproducible-voc-detection-contracts.zh-CN.md) | 需要理解数据、权重、checkpoint 和证据边界背后的原因 | 架构依据，而不是 benchmark 成绩 |

## 证据边界

示例和大多数测试使用合成张量、临时构造的 VOC 格式测试数据或 fake detector。它们验证 API、几何、序列化和编排契约。学习配方带有样本上限，属于有界训练；它可以证明学习流程已经连通，但其中的指标不能当作完整数据集 benchmark。

仓库已经发布一次证据完整的参考运行：官方划分上训练 26 轮，由验证集选择第 18 轮，并在全部 4,952 张测试图像上得到 `map_50_95 = 0.322312`。[实测运行](recorded-run/README.zh-CN.md)保存了适用范围、环境、指标、checkpoint 哈希和失败案例图。合成示例和有界学习运行仍然只是教学证据，不能冒充这次完整结果。

## 辅助索引

- [示例](../examples/README.zh-CN.md)：从框到 checkpoint 预测的渐进式可执行契约。
- [配置](../configs/README.zh-CN.md)：每份仓库配方的作用、网络策略与产物范围。
- [脚本](../scripts/README.zh-CN.md)：下载、可视化、绘图和文档资源工具。
- [测试](../tests/README.zh-CN.md)：聚焦测试层与完整离线测试套件。
