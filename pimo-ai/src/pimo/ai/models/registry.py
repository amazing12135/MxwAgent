"""pimo-ai 模型注册表。

从静态模型数据（``models/generated.py`` -> ``MODELS`` dict）构建内存索引，
提供按 provider + model_id 查询、按 provider 枚举、思考级别校验等功能。
"""

from __future__ import annotations

from pimo.ai.types import Model, ThinkingLevel

# =============================================================================
# 公共函数
# =============================================================================


def get_model(provider: str, model_id: str) -> Model | None:
    """按厂商和模型 ID 查询 Model 对象。

    Args:
        provider: 厂商名，如 ``"anthropic"``。
        model_id: 模型 ID，如 ``"claude-sonnet-4-20250514"``。

    Returns:
        匹配的 Model，未找到返回 None。
    """
    ...


def get_providers() -> list[str]:
    """获取所有已注册的厂商名列表。

    Returns:
        厂商名字符串列表，按注册顺序排列。
    """
    ...


def get_models(provider: str) -> list[Model]:
    """获取指定厂商的所有模型。

    Args:
        provider: 厂商名。

    Returns:
        该厂商的 Model 列表。厂商不存在时返回空列表。
    """
    ...


def get_supported_thinking_levels(model: Model) -> list[ThinkingLevel]:
    """查询模型支持的思考级别。

    不支持 reasoning 的模型只返回 ``["off"]``；
    支持 reasoning 的模型返回所有有效级别。

    Args:
        model: 目标模型。

    Returns:
        可用的 ThinkingLevel 列表。
    """
    ...


def clamp_thinking_level(model: Model, level: ThinkingLevel) -> ThinkingLevel:
    """将请求的思考级别夹紧到模型实际支持的范围内。

    若 model 不支持请求的级别，自动回退到最近的高级别，
    无高级别时回退到最近的低级别。

    Args:
        model: 目标模型。
        level: 用户请求的思考级别。

    Returns:
        调整后的有效 ThinkingLevel。
    """
    ...
