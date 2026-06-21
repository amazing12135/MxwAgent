"""OpenAI Responses API 共享模块（占位）。

提供 convert_responses_messages、convert_responses_tools、process_responses_stream
等函数，被 openai_responses.py 和 openai_codex_responses.py 复用。

TODO: 完整实现将在后续 /gen 任务中完成。
"""

from __future__ import annotations

from typing import Any

from pimo.ai.types import AssistantMessage, Context, Model


async def process_responses_stream(
    openai_stream: Any,
    output: AssistantMessage,
    stream: Any,
    model: Model,
    options: dict[str, Any] | None = None,
) -> None:
    """消费 OpenAI Responses SSE 事件流，转换为归一化事件。

    Args:
        openai_stream: OpenAI SDK 返回的 ResponseStreamEvent 异步迭代器。
        output: 正在构建的 AssistantMessage。
        stream: AssistantMessageEventStream 生产者。
        model: 目标模型。
        options: 流选项（service_tier 等）。
    """
    ...


def convert_responses_messages(
    model: Model,
    context: Context,
    allowed_tool_call_providers: set[str] | None = None,
    *,
    include_system_prompt: bool = True,
) -> list[dict[str, Any]]:
    """将统一 Message 列表转换为 OpenAI Responses Input 格式。

    Args:
        model: 目标模型。
        context: 统一上下文。
        allowed_tool_call_providers: 允许直传 tool call ID 的 provider 集合。
        include_system_prompt: 是否包含 system prompt。

    Returns:
        OpenAI ResponseInput 列表。
    """
    ...


def convert_responses_tools(
    tools: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """将统一 Tool JSON Schema 列表转换为 OpenAI Responses Tool 格式。

    Args:
        tools: JSON Schema 格式的工具定义列表。
        strict: 是否启用 strict 模式。

    Returns:
        OpenAI Tool 数组。
    """
    ...


def _map_stop_reason(status: str | None) -> str:
    """将 OpenAI Responses response.status 映射为 pimo stop_reason。

    Args:
        status: response.status 值。

    Returns:
        标准化 stop_reason。
    """
    ...
