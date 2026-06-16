"""pimo-ai Provider 注册表。

全局注册表将 API 协议名（如 ``"anthropic-messages"``）映射到对应的 Provider
实现。各 Provider 模块加载时通过 ``register_api_provider()`` 自注册，
调用方通过 ``get_api_provider()`` 按 Model.api 字段路由到正确的 Provider。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pimo.ai.event_stream import AssistantMessageEventStream
    from pimo.ai.types import Context, Model, SimpleStreamOptions, StreamOptions

# =============================================================================
# ApiProvider 抽象基类 — 所有 Provider 必须实现
# =============================================================================


class ApiProvider(ABC):
    """LLM Provider 抽象基类。

    每个 Provider 封装一种 API 协议（如 Anthropic Messages、OpenAI Completions），
    负责：统一 Context → 厂商原生格式 → 调用 SDK → 原生事件 → 统一事件。

    子类必须定义:
    - ``api: str`` — 协议名，作为注册键。如 ``"anthropic-messages"``。
    - ``stream()`` — 流式调用 LLM。
    - ``stream_simple()`` — 带 reasoning 支持的流式调用。
    """

    api: str
    """协议名，作为注册键。子类必须覆盖此属性。"""

    @abstractmethod
    async def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """流式调用 LLM，返回归一化事件流。

        Args:
            model: 目标模型描述。
            context: 统一上下文（system_prompt + messages + tools）。
            options: 流式选项（api_key、signal、transport 等）。

        Returns:
            AssistantMessageEventStream: 归一化后的异步事件流。
        """
        ...

    @abstractmethod
    async def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """stream 的简化版本，额外支持 reasoning 参数。

        Args:
            model: 目标模型描述。
            context: 统一上下文。
            options: 包含 reasoning、session_id、thinking_budgets 的选项。

        Returns:
            AssistantMessageEventStream: 归一化后的异步事件流。
        """
        ...


# =============================================================================
# 内部结构
# =============================================================================


class _RegistryEntry:
    """注册表条目：Provider + 来源标识。

    source_id 用于扩展生命周期管理——卸载扩展时按 source_id 批量移除。
    """

    __slots__ = ("provider", "source_id")

    def __init__(self, provider: ApiProvider, source_id: str | None = None) -> None:
        self.provider = provider
        self.source_id = source_id


# =============================================================================
# 全局注册表
# =============================================================================

_api_provider_registry: dict[str, _RegistryEntry] = {}
"""内部注册表。键为 API 协议名，值为 Provider + 来源封装。"""


# =============================================================================
# 注册 / 查询 / 管理函数
# =============================================================================


def register_api_provider(
    provider: ApiProvider, source_id: str | None = None
) -> None:
    """注册一个 ApiProvider 到全局注册表。

    各 Provider 模块（``providers/anthropic.py`` 等）在模块加载时调用此函数
    完成自注册。注册键为 ``provider.api``，同名注册会覆盖旧的 Provider。

    Args:
        provider: 实现了 ApiProvider 协议的 Provider 实例。
        source_id: 可选的来源标识（如扩展模块路径），用于后续按源卸载。
    """
    _api_provider_registry[provider.api] = _RegistryEntry(provider, source_id)


def get_api_provider(api: str) -> ApiProvider | None:
    """按 API 协议名查询已注册的 Provider。

    Args:
        api: API 协议名，如 ``"anthropic-messages"``。

    Returns:
        注册的 Provider 实例，未注册时返回 None。
    """
    entry = _api_provider_registry.get(api)
    return entry.provider if entry else None


def get_api_providers() -> list[ApiProvider]:
    """获取所有已注册的 Provider 列表。

    Returns:
        按注册顺序排列的 Provider 实例列表。
    """
    return [entry.provider for entry in _api_provider_registry.values()]


def unregister_api_providers(source_id: str) -> int:
    """按来源标识批量移除已注册的 Provider。

    扩展卸载时使用，避免残留已卸载扩展注册的 Provider。

    Args:
        source_id: ``register_api_provider`` 时传入的来源标识。

    Returns:
        被移除的 Provider 数量。
    """
    to_remove = [
        api
        for api, entry in _api_provider_registry.items()
        if entry.source_id == source_id
    ]
    for api in to_remove:
        del _api_provider_registry[api]
    return len(to_remove)


def clear_api_providers() -> None:
    """清空所有已注册的 Provider。

    主要用于测试环境重置。
    """
    _api_provider_registry.clear()
