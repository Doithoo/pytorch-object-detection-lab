# 在 Kaggle 上训练

[English](kaggle.md) | [实测运行](../recorded-run/README.zh-CN.md)

本地 CPU 运行完整 VOC 参考配方太慢时，可以使用 Kaggle。官方 CLI 负责认证、提交、
状态、日志和结果下载。本次实测 runner 直接下载两份官方 VOC 压缩包，并嵌入源码快照，
因此不需要 `kagglehub`。

## 安装与认证

把 CLI 安装为用户工具，不加入项目运行依赖：

```bash
uv tool install kaggle
kaggle auth login
kaggle --version
```

如果过期 OAuth 缓存一边显示已经登录、一边被 API 拒绝，用
`kaggle auth login --force` 刷新。

## 使用兼容 GPU

请求 `NvidiaTeslaT4` 或更新 GPU。当前 Kaggle PyTorch 2.10 CUDA 12.8 构建支持计算能力
7.0 及以上；Tesla P100 是 `sm_60`，会报
`no kernel image is available for execution on the device`。本次分配显示 T4 x2，但只用
`cuda:0`；项目没有实现多 GPU 训练。

应保持网络开启，让 runner 下载官方 VOC 2007 和固定 ImageNet backbone 权重。本流程
不需要附加 Kaggle dataset。

## 提交与监控

[`../recorded-run/kaggle/run_kaggle.py`](../recorded-run/kaggle/run_kaggle.py)
是实际执行的自包含 v7 runner，相邻 metadata 请求 T4 和网络。换到其他账户发布副本前，
先修改其中的 `id`，然后运行：

```bash
kaggle kernels push -p docs/recorded-run/kaggle
kaggle kernels status yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7
```

runner 每 60 秒输出一次心跳。第二张 T4 空闲并不代表故障，不要因此修改或重新提交正在
运行的 kernel。

## 只下载结果

状态变成 `KernelWorkerStatus.COMPLETE` 后，避免下载临时生成的 1.7 GB VOC 目录：

```bash
kaggle kernels output yashowhoo/pytorch-object-detection-lab-voc2007-gpu-run-v7 --file-pattern 'artifacts/.*' -p kaggle-output
```

引用指标前，应核对 `completed_epochs`、划分数量、最佳验证轮次、测试图像数和 `best.pt`
的 SHA-256。除非项目明确通过 release 或模型托管发布，否则不要把大型 checkpoint 放进
Git。
