"""自动生成的模型目录数据。

由 ``scripts/generate_models.py`` 从 pi 模型目录同步生成。
当前为占位空数据，运行同步脚本后填充 ~200 个热门模型。
"""

from __future__ import annotations

from pimo.ai.types import Model

# 结构: { provider: { model_id: Model } }
# 示例:
#   MODELS = {
#       "anthropic": {
#           "claude-sonnet-4-20250514": Model(id=..., name=..., ...),
#       },
#       "openai": {
#           "gpt-4o": Model(id=..., name=..., ...),
#       },
#   }
MODELS: dict[str, dict[str, Model]] = {}
