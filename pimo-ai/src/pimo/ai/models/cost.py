"""pimo-ai 成本计算。

根据模型的 token 单价和使用量计算一次 LLM 调用的货币成本。
"""

from __future__ import annotations

from pimo.ai.types import CostInfo, Model, Usage

_MILLION = 1_000_000


def calculate_cost(model: Model, usage: Usage) -> CostInfo:
    """根据 token 用量计算成本并回填到 usage.cost。

    model.cost 各字段单位为 **美元/百万 token**，usage 中各字段为
    **token 数**。计算后原地修改 usage.cost 并返回。

    Args:
        model: 包含 cost 单价的模型描述。
        usage: 包含 input/output/cache_read/cache_write token 数的用量对象。

    Returns:
        CostInfo: 计算后的 usage.cost（与传入的 usage.cost 是同一对象）。
    """
    usage.cost.input = (model.cost.input / _MILLION) * usage.input
    usage.cost.output = (model.cost.output / _MILLION) * usage.output
    usage.cost.cache_read = (model.cost.cache_read / _MILLION) * usage.cache_read
    usage.cost.cache_write = (model.cost.cache_write / _MILLION) * usage.cache_write
    usage.cost.total = (
        usage.cost.input
        + usage.cost.output
        + usage.cost.cache_read
        + usage.cost.cache_write
    )
    return usage.cost
