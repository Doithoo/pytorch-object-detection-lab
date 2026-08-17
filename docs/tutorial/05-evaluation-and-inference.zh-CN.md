# 05 - 评估与推理

评估从 checkpoint 重建模型，在推理前校验 manifest identity，计算 COCO 风格 AP/AR、确定性排序错误并渲染证据。预测也以 `weights=none` 重建，不需要 YAML；JSON 保留全部有效检测，之后才应用可视化显示上限。

运行：`uv run detect predict --checkpoint artifacts/run/best.pt --image image.jpg --output-dir artifacts/prediction --device cpu`

预期：写入 `image.json` 与 `image.png`，或以退出码 2 返回简洁的 checkpoint/图像错误。
