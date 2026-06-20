"""pimo-ai 模型注册表。

从静态模型数据（``models/generated.py`` -> ``MODELS`` dict）构建内存索引，
提供按 provider + model_id 查询、按 provider 枚举、思考级别校验等功能。
"""

from __future__ import annotations

from pimo.ai.models.generated import MODELS
from pimo.ai.types import Model, ThinkingLevel

# =============================================================================
# 思考级别定义
# =============================================================================

# 规范思考级别全集（按强度递增）
_EXTENDED_THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

# =============================================================================
# 内存索引
# =============================================================================

# provider → { model_id → Model }，模块加载时从 MODELS 构建
_registry: dict[str, dict[str, Model]] = {}

for _provider, _models in MODELS.items():
    _registry[_provider] = dict(_models)


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
    provider_models = _registry.get(provider)
    if provider_models is None:
        return None
    return provider_models.get(model_id)


def get_providers() -> list[str]:
    """获取所有已注册的厂商名列表。

    Returns:
        厂商名字符串列表，按注册顺序排列。
    """
    return list(_registry.keys())


def get_models(provider: str) -> list[Model]:
    """获取指定厂商的所有模型。

    Args:
        provider: 厂商名。

    Returns:
        该厂商的 Model 列表。厂商不存在时返回空列表。
    """
    provider_models = _registry.get(provider)
    if provider_models is None:
        return []
    return list(provider_models.values())


def get_supported_thinking_levels(model: Model) -> list[ThinkingLevel]:
    """查询模型支持的思考级别。

    不支持 reasoning 的模型只返回 ``["off"]``；
    支持 reasoning 的模型根据 ``thinking_level_map`` 过滤无效级别，
    ``"xhigh"`` 需要显式映射才视为支持。

    Args:
        model: 目标模型。

    Returns:
        可用的 ThinkingLevel 列表。
    """
    if not model.reasoning:
        return ["off"]

    level_map = model.thinking_level_map or {}
    supported: list[ThinkingLevel] = []
    for level in _EXTENDED_THINKING_LEVELS:
        mapped = level_map.get(level)
        # None 明确表示不支持该级别
        if mapped is None and level in level_map:
            continue
        # "xhigh" 需要显式映射（不能只有 key 没有 value）
        if level == "xhigh" and mapped is None:
            continue
        supported.append(level)
    return supported


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
    available = get_supported_thinking_levels(model)
    if level in available:
        return level

    requested_idx = _EXTENDED_THINKING_LEVELS.index(level) if level in _EXTENDED_THINKING_LEVELS else -1
    if requested_idx == -1:
        return available[0] if available else "off"

    # 优先向高级别方向夹紧
    for i in range(requested_idx, len(_EXTENDED_THINKING_LEVELS)):
        candidate = _EXTENDED_THINKING_LEVELS[i]
        if candidate in available:
            return candidate

    # 无高级别时向低级别回落
    for i in range(requested_idx - 1, -1, -1):
        candidate = _EXTENDED_THINKING_LEVELS[i]
        if candidate in available:
            return candidate

    return available[0] if available else "off"
