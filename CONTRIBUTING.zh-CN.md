# 参与贡献

安装 Python 3.10-3.12 与 uv，然后运行 `uv sync --locked --extra dev`。保持改动聚焦，并在修改生产行为前先写失败测试。

提交前运行 `uv run ruff check .`、`uv run ruff format --check .`、`uv run mypy` 与 `uv run pytest -W error::DeprecationWarning`。测试必须离线，禁止在测试中下载数据集、模型权重、字体或其他资源；请使用合成 VOC fixture 与 fake detector。

提交信息使用英文 ASCII Conventional Commits，例如 `feat(evaluation): Add report export`。主题采用祈使语气、不超过 72 字符，并在正文解释改动与原因。新数据集应实现 manifest 数据边界，新模型应注册显式构造器与权重策略；同时更新对应的英文和中文文档。
