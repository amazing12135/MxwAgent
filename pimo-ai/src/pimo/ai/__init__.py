"""pimo-ai — LLM 抽象层。

提供多 Provider 统一 API、流式事件、模型注册表和辅助工具。

公共 API:
- ``stream()`` / ``complete()`` — 流式/完整调用（自动解析 API Key）
- ``stream_simple()`` / ``complete_simple()`` — 简化版（含 reasoning）
- ``get_model()`` / ``get_providers()`` — 模型目录查询
"""

from __future__ import annotations

from pimo.ai.api_registry import get_api_provider
from pimo.ai.env_api_keys import get_env_api_key
from pimo.ai.models.registry import (
    clamp_thinking_level,
    get_model,
    get_models,
    get_providers,
    get_supported_thinking_levels,
)
from pimo.ai.types import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)

# 触发内置 Provider 自动注册
from pimo.ai.providers.register_builtins import (  # noqa: F401
    register_builtin_api_providers,
    reset_api_providers,
)


def stream(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> "AssistantMessageEventStream":
    """流式调用 LLM，自动解析 API Key 并路由到对应 Provider。

    Args:
        model: 目标模型描述。
        context: 统一上下文。
        options: 流式选项。api_key 为 None 时从环境变量自动解析。

    Returns:
        AssistantMessageEventStream: 归一化后的异步事件流。

    Raises:
        RuntimeError: model.api 对应的 Provider 未注册。
    """
    from pimo.ai.event_stream import AssistantMessageEventStream

    provider = get_api_provider(model.api)
    if not provider:
        raise RuntimeError(
            f"No API provider registered for api: {model.api}"
        )
    resolved = _with_env_api_key(model, options)
    return provider.stream(model, context, resolved)


async def complete(
    model: Model,
    context: Context,
    options: StreamOptions | None = None,
) -> AssistantMessage:
    """完整调用 LLM，等待流结束后返回最终 AssistantMessage。

    Args:
        model: 目标模型描述。
        context: 统一上下文。
        options: 流式选项。

    Returns:
        最终的 AssistantMessage。
    """
    s = stream(model, context, options)
    return await s.result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> "AssistantMessageEventStream":
    """stream 的简化版本，额外支持 reasoning 参数。

    Args:
        model: 目标模型描述。
        context: 统一上下文。
        options: 简化选项（含 reasoning）。

    Returns:
        AssistantMessageEventStream。
    """
    from pimo.ai.event_stream import AssistantMessageEventStream

    provider = get_api_provider(model.api)
    if not provider:
        raise RuntimeError(
            f"No API provider registered for api: {model.api}"
        )
    resolved = _with_env_api_key(model, options)
    return provider.stream_simple(model, context, resolved)


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """complete 的简化版本。

    Args:
        model: 目标模型描述。
        context: 统一上下文。
        options: 简化选项。

    Returns:
        最终的 AssistantMessage。
    """
    s = stream_simple(model, context, options)
    return await s.result()


def _with_env_api_key(
    model: Model,
    options: StreamOptions | SimpleStreamOptions | None,
) -> StreamOptions | SimpleStreamOptions | None:
    """如果 options 中未提供 api_key，则从环境变量自动解析。"""
    if options and options.api_key and options.api_key.strip():
        return options
    env_key = get_env_api_key(model.provider)
    if not env_key:
        return options
    if options is None:
        return StreamOptions(api_key=env_key)
    return type(options)(**{**options.__dict__, "api_key": env_key})
