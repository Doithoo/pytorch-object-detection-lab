# 学习路径

顺序是 `download -> prepare -> inspect -> dry run -> train -> evaluate -> predict`。准备阶段生成稳定的 manifest identity，后续 checkpoint 与报告都携带它；dry run 在较长的有界训练前证明一次更新可行。

运行：`uv run detect --version`

预期：无需加载数据集或模型，输出 `0.1.0`。
