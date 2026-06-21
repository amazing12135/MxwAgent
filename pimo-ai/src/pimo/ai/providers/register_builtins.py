"""内置 Provider 注册。

模块导入时自动将 AnthropicProvider、OpenAICompletionsProvider、
OpenAIResponsesProvider 注册到全局 Provider 注册表。
"""

from __future__ import annotations

from pimo.ai.api_registry import (
    clear_api_providers,
    register_api_provider,
)
from pimo.ai.providers.anthropic import AnthropicProvider
from pimo.ai.providers.openai_completions import OpenAICompletionsProvider
from pimo.ai.providers.openai_responses import OpenAIResponsesProvider


def register_builtin_api_providers() -> None:
    """注册所有内置 Provider 到全局注册表。

    可重复调用——同名 Provider 会覆盖旧注册。
    """
    register_api_provider(AnthropicProvider())
    register_api_provider(OpenAICompletionsProvider())
    register_api_provider(OpenAIResponsesProvider())


def reset_api_providers() -> None:
    """清空并重新注册所有 Provider（用于测试环境重置）。"""
    clear_api_providers()
    register_builtin_api_providers()


# 模块导入时自动注册
register_builtin_api_providers()
